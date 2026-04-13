import hashlib
import json
import os.path
import sys
import threading
from datetime import datetime
from queue import Empty, Full, Queue

import PIL
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from store.config import ConfigStore
from utils.common import singleton
from utils.vision import VisionTools, draw_rectangle, draw_text
from utils.mediapipe.mp import MediaPipe as MP
import multiprocessing
import requests
from time import sleep

from service.recognition_result import select_display_name

FACE_SERVICE_BASE_URL = os.getenv('FACE_SERVICE_BASE_URL', 'http://127.0.0.1:18000').rstrip('/')

try:
    import mindspore as ms
    from mindspore import Tensor, context
    from mindspore.train.serialization import load_checkpoint, load_param_into_net
    from utils.mindface.detection.models import RetinaFace, resnet50, mobilenet025
    from utils.mindface.detection.runner import DetectionEngine, read_yaml
    from utils.mindface.detection.utils import prior_box
    from utils.mindface.recognition.models import iresnet50, iresnet100, get_mbf, vit_t, vit_s, vit_b, vit_l

    MINDSPORE_AVAILABLE = True
except ImportError:
    ms = None
    Tensor = None
    context = None
    load_checkpoint = None
    load_param_into_net = None
    RetinaFace = None
    resnet50 = None
    mobilenet025 = None
    DetectionEngine = None
    read_yaml = None
    prior_box = None
    iresnet50 = None
    iresnet100 = None
    get_mbf = None
    vit_t = None
    vit_s = None
    vit_b = None
    vit_l = None
    MINDSPORE_AVAILABLE = False


class VisionService(QThread):
    """
    摄像头相关服务
    """
    resultSignal = pyqtSignal(str)  # 识别结果信号状态
    registerResSignal = pyqtSignal(str)  # 注册结果信号状态
    cameraStatusSignal = pyqtSignal(bool, str)  # 摄像头状态

    def __init__(self, store):
        super(VisionService, self).__init__()
        self.all_queues = {
            'display': Queue(maxsize=1),
            'video': Queue(maxsize=1),
        }
        self.isRunning = True
        self.config = store
        self.classicFaceRecognizer = ClassicFaceRecognizer()
        self.video_stream_in = None
        self.recogThread = RecogThread(self.config.recogQueue, self.resultSignal,
                                       self.classicFaceRecognizer,
                                       None)
        self.recogThread.start()
        self.registerThread = RegisterThread(self.registerResSignal)
        self.registerThread.start()

    def close_camera(self):
        """
        close camera and release resources
        """
        if self.video_stream_in is None:
            self.config.set_config('camera_on', False)
            self.cameraStatusSignal.emit(False, '摄像头已关闭')
            return
        if not self.video_stream_in.is_alive():
            self.video_stream_in = None
            self.config.set_config('camera_on', False)
            self.cameraStatusSignal.emit(False, '摄像头已关闭')
            return
        self.video_stream_in.stop_read_camera()

    def start_camera(self):
        """
        start camera
        """
        if self.video_stream_in is not None and self.video_stream_in.is_alive():
            return False
        self.video_stream_in = ReadCameraThread(
            self.all_queues,
            self.resultSignal,
            self.registerResSignal,
            self.cameraStatusSignal,
            0,
        )
        self.video_stream_in.start()
        return True

    def vision_face_register(self, name):
        if self.video_stream_in is None:
            self.registerResSignal.emit('注册失败')
            return
        self.video_stream_in.register_face_inner(name)


def classicFaceRecognize(faces, classicPredictor):
    return classicPredictor.predict(faces)


def encode_face_image(image):
    if image is None or image.size == 0:
        raise ValueError('empty face image')
    success, encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not success:
        raise ValueError('failed to encode face image')
    return encoded.tobytes()


class ReadCameraThread(threading.Thread):
    """
    摄像头读取线程
    """

    def __init__(self, video_queue, resultSignal, registerResultSignal, cameraStatusSignal, camera_id=0):
        super(ReadCameraThread).__init__()
        threading.Thread.__init__(self)
        self.camera_id = camera_id
        self.read_capture = None
        self.video_queue = video_queue
        self.resultSignal = resultSignal
        self.registerResultSignal = registerResultSignal
        self.cameraStatusSignal = cameraStatusSignal
        self.vision_tools = VisionTools()
        self.running = True
        self.config = ConfigStore()
        self.recogQueue = self.config.recogQueue
        self.registerQueue = self.config.registerQueue
        self.mindface = None
        self.mindface_failed = False
        self.detect_config = {}
        self.classicFaceRecognizer = ClassicFaceRecognizer()
        self.mp = MP()
        self.current_face = None
        self.current_frame = None
        self.last_detected_face = None
        self.missed_face_frames = 0
        self.max_missed_face_frames = 8
        self.mp_failed = False

    def run(self):
        self.read_capture, status_message = self.open_capture()
        if self.read_capture is None or not self.read_capture.isOpened():
            self.cameraStatusSignal.emit(False, status_message)
            return

        self.cameraStatusSignal.emit(True, status_message)

        while self.read_capture.isOpened() and self.running:
            ret, frame = self.read_capture.read()
            res = None
            faces = []
            if not ret or frame is None:
                if self.running:
                    self.cameraStatusSignal.emit(False, '摄像头读取失败，请重试')
                break
            self.current_frame = frame
            if self.config.get_config('face_detect'):
                res = frame.copy()
                if self.config.get_config('detect_method') == self.config.detect_methods_mapper['classic']:
                    detected_faces = self.classicFaceRecognizer.detect_face(frame)
                    faces = self.normalize_faces(detected_faces, frame.shape)
                if self.config.get_config('detect_method') == self.config.detect_methods_mapper['mediapipe']:
                    try:
                        detected_faces = self.mp.transform_result(self.mp.detect_face(frame))
                        faces = self.normalize_faces(detected_faces, frame.shape)
                        self.mp_failed = False
                    except Exception:
                        self.mp_failed = True
                        draw_text(res, '轻量检测暂不可用', 10, 30)
                        faces = []
                if self.config.get_config('detect_method') == self.config.detect_methods_mapper['mindspore']:
                    if self.mindface is None and MINDSPORE_AVAILABLE and not self.mindface_failed:
                        try:
                            self.mindfaceInit()
                        except Exception:
                            self.mindface_failed = True
                    if self.mindface is None:
                        draw_text(res, 'MindSpore 不可用', 10, 30)
                        faces = []
                    else:
                        detected_faces = self.mindfaceFaceDetect(frame)
                        faces = self.normalize_faces(detected_faces, frame.shape)

                faces = self.stabilize_faces(faces)
                for face in faces:
                    draw_rectangle(res, face)
            else:
                self.current_face = None
                self.last_detected_face = None
                self.missed_face_frames = 0

            if self.config.get_config('recog_open') and faces is not None and len(faces) > 0:
                face_img = self.vision_tools.cut_face(frame, faces[0])
                if self.recogQueue.empty():
                    self.recogQueue.put(face_img)
            if frame is not None:
                self.push_latest_frame('video', frame)
            if res is not None:
                self.push_latest_frame('display', res)
            else:
                self.push_latest_frame('display', frame)

        if self.read_capture is not None:
            self.read_capture.release()
            self.read_capture = None
        self.mp.close()
        cv2.destroyAllWindows()
        self.current_frame = None
        self.current_face = None
        if not self.running:
            self.cameraStatusSignal.emit(False, '摄像头已关闭')

    def open_capture(self):
        candidate_ids = []
        for camera_id in (self.camera_id, 0, 1, 2, 3):
            if camera_id not in candidate_ids:
                candidate_ids.append(camera_id)

        candidate_backends = []
        if sys.platform == 'darwin':
            for backend_name in ('CAP_AVFOUNDATION', 'CAP_ANY'):
                backend = getattr(cv2, backend_name, None)
                if backend is not None and backend not in candidate_backends:
                    candidate_backends.append(backend)
        elif sys.platform.startswith('win'):
            for backend_name in ('CAP_MSMF', 'CAP_DSHOW', 'CAP_ANY'):
                backend = getattr(cv2, backend_name, None)
                if backend is not None and backend not in candidate_backends:
                    candidate_backends.append(backend)
        else:
            for backend_name in ('CAP_V4L2', 'CAP_ANY'):
                backend = getattr(cv2, backend_name, None)
                if backend is not None and backend not in candidate_backends:
                    candidate_backends.append(backend)

        attempts = []
        for camera_id in candidate_ids:
            for backend in candidate_backends:
                capture = cv2.VideoCapture(camera_id, backend)
                backend_name = self.backend_name(backend)
                if not capture.isOpened():
                    attempts.append(f'{camera_id}:{backend_name}:open-failed')
                    capture.release()
                    continue

                capture.set(cv2.CAP_PROP_FPS, 30)
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                frame = None
                frame_ready = False
                for _ in range(8):
                    ret, frame = capture.read()
                    if ret and frame is not None:
                        frame_ready = True
                        break
                    sleep(0.08)

                if frame_ready:
                    self.push_latest_frame('display', frame)
                    self.push_latest_frame('video', frame)
                    return capture, f'摄像头已连接：设备 {camera_id} / 后端 {backend_name}'

                attempts.append(f'{camera_id}:{backend_name}:read-failed')
                capture.release()

        attempts_text = '，'.join(attempts[:6]) if attempts else '没有可尝试的摄像头后端'
        return None, f'未检测到可用摄像头，已尝试 {attempts_text}'

    def backend_name(self, backend):
        backend_names = {
            getattr(cv2, 'CAP_ANY', -1): '默认',
            getattr(cv2, 'CAP_AVFOUNDATION', -2): 'AVFoundation',
            getattr(cv2, 'CAP_MSMF', -3): 'MSMF',
            getattr(cv2, 'CAP_DSHOW', -4): 'DirectShow',
            getattr(cv2, 'CAP_V4L2', -5): 'V4L2',
        }
        return backend_names.get(backend, str(backend))

    def push_latest_frame(self, queue_name, frame):
        target_queue = self.video_queue[queue_name]
        if target_queue.full():
            try:
                target_queue.get_nowait()
            except Empty:
                pass
        try:
            target_queue.put_nowait(frame)
        except Full:
            pass

    def stop_read_camera(self):
        self.running = False

    def normalize_face_rect(self, face, frame_shape, expand_ratio=0.12):
        if face is None:
            return None

        try:
            x, y, w, h = [int(value) for value in face]
        except (TypeError, ValueError):
            return None

        if w <= 0 or h <= 0:
            return None

        frame_height, frame_width = frame_shape[:2]
        pad_x = int(w * expand_ratio)
        pad_y = int(h * expand_ratio)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame_width, x + w + pad_x)
        y2 = min(frame_height, y + h + pad_y)

        clipped_w = x2 - x1
        clipped_h = y2 - y1
        if clipped_w <= 0 or clipped_h <= 0:
            return None

        return x1, y1, clipped_w, clipped_h

    def select_best_face(self, faces, frame_shape):
        normalized_faces = []
        face_iterable = faces if faces is not None else []
        for face in face_iterable:
            normalized_face = self.normalize_face_rect(face, frame_shape)
            if normalized_face is not None:
                normalized_faces.append(normalized_face)

        if not normalized_faces:
            return None

        return max(normalized_faces, key=lambda item: item[2] * item[3])

    def normalize_faces(self, faces, frame_shape):
        normalized_faces = []
        face_iterable = faces if faces is not None else []
        for face in face_iterable:
            normalized_face = self.normalize_face_rect(face, frame_shape)
            if normalized_face is not None:
                normalized_faces.append(normalized_face)

        normalized_faces.sort(key=lambda item: item[2] * item[3], reverse=True)
        return normalized_faces

    def stabilize_faces(self, faces):
        if faces:
            self.current_face = faces[0]
            self.last_detected_face = faces[0]
            self.missed_face_frames = 0
            return faces

        if self.last_detected_face is not None and self.missed_face_frames < self.max_missed_face_frames:
            self.missed_face_frames += 1
            self.current_face = self.last_detected_face
            return [self.last_detected_face]

        self.current_face = None
        self.last_detected_face = None
        self.missed_face_frames = 0
        return []

    def detect_faces_by_method(self, frame, method_name):
        if method_name == 'classic':
            faces = self.classicFaceRecognizer.detect_face(frame)
            if faces is None or len(faces) == 0:
                return []
            return [tuple(face) for face in faces]

        if method_name == 'mediapipe':
            try:
                detection_result = self.mp.detect_face(frame)
            except Exception:
                return []
            return self.mp.transform_result(detection_result)

        if method_name == 'mindspore':
            if not MINDSPORE_AVAILABLE or self.mindface_failed:
                return []
            if self.mindface is None:
                try:
                    self.mindfaceInit()
                except Exception:
                    self.mindface_failed = True
                    return []
            if self.mindface is None:
                return []
            try:
                return self.mindfaceFaceDetect(frame)
            except Exception:
                self.mindface_failed = True
                return []

        return []

    def locate_face_for_capture(self, frame):
        face = self.select_best_face([self.current_face], frame.shape)
        if face is not None:
            return face

        methods = []
        if self.config.get_config('face_detect'):
            current_method = self.config.get_config('detect_method')
            detect_methods = self.config.detect_methods_mapper
            if current_method == detect_methods['classic']:
                methods.append('classic')
            elif current_method == detect_methods['mediapipe']:
                methods.append('mediapipe')
            elif current_method == detect_methods['mindspore']:
                methods.append('mindspore')

        for fallback_method in ('mediapipe', 'classic', 'mindspore'):
            if fallback_method not in methods:
                methods.append(fallback_method)

        for method_name in methods:
            face = self.select_best_face(self.detect_faces_by_method(frame, method_name), frame.shape)
            if face is not None:
                return face

        return None

    def register_face_inner(self, name):
        clean_name = name.strip()
        if not clean_name:
            self.registerResultSignal.emit('注册失败：姓名不能为空')
            return

        if self.current_frame is None:
            self.registerResultSignal.emit('注册失败：当前没有可用画面')
            return

        target_face = self.locate_face_for_capture(self.current_frame)
        if target_face is None:
            self.registerResultSignal.emit('注册失败：当前没有锁定到人脸，请正对镜头或先开启轻量检测')
            return

        face_image = self.vision_tools.cut_face(self.current_frame, target_face)
        if face_image is None or face_image.size == 0:
            self.registerResultSignal.emit('注册失败：人脸裁剪失败，请调整位置后重试')
            return

        self.current_face = target_face
        self.registerQueue.put((face_image, clean_name))

    def mindfaceInit(self):
        if not MINDSPORE_AVAILABLE:
            self.mindface = None
            self.detect_config = {}
            return
        detect_config = 'utils/mindface/detection/configs/RetinaFace_mobilenet025.yaml'
        self.detect_config = read_yaml(detect_config)
        self.detect_config['val_model'] = 'utils/mindface/detection/pretrained/RetinaFace_MobileNet025.ckpt'
        self.detect_config['conf'] = 0.8
        self.mindface = MindFaceService(self.detect_config)

    def mindfaceFaceDetect(self, frame):
        boxes = self.mindface.face_detect(self.detect_config, frame)
        faces = []
        for box in boxes:
            if box[4] > self.detect_config['conf']:
                faces.append((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
        return faces


@singleton
class ClassicFaceRecognizer:
    """
    传统面部识别能力
    """
    DEFAULT_UNKNOWN_THRESHOLD = 95

    def __init__(self, method='lbph'):
        self.method = method
        self.vision = VisionTools()
        self.method = method
        self.trained = False
        self.training = False
        self.predicting = False
        self.label_name_map = {}
        self.training_data_path = os.path.join('dataset', 'full')
        self.training_logger = None
        self.unknown_threshold = self.DEFAULT_UNKNOWN_THRESHOLD
        self.model_path = os.path.join('pretrained', f'classic_{self.method}.yml')
        self.model_meta_path = os.path.join('pretrained', f'classic_{self.method}.json')
        self.face_recognizer = self.create_face_recognizer()
        self.load_model()

    def create_face_recognizer(self):
        if self.method == 'lbph':
            return cv2.face.LBPHFaceRecognizer.create()
        if self.method == 'elgenface':
            return cv2.face.EigenFaceRecognizer.create()
        if self.method == 'fisherface':
            return cv2.face.FisherFaceRecognizer.create()
        raise ValueError(f'unsupported recognizer method: {self.method}')

    def delete_saved_model_files(self):
        for model_file in (self.model_path, self.model_meta_path):
            try:
                if os.path.exists(model_file):
                    os.remove(model_file)
            except OSError:
                continue

    def invalidate_model(self, remove_persisted=False):
        self.trained = False
        self.predicting = False
        self.label_name_map = {}
        self.face_recognizer = self.create_face_recognizer()
        if remove_persisted:
            self.delete_saved_model_files()

    def dataset_signature(self):
        if not os.path.isdir(self.training_data_path):
            return None

        signature_items = []
        for root, dirs, files in os.walk(self.training_data_path):
            dirs[:] = sorted(dir_name for dir_name in dirs if not dir_name.startswith('.'))
            visible_files = sorted(file_name for file_name in files if not file_name.startswith('.'))
            for file_name in visible_files:
                file_path = os.path.join(root, file_name)
                try:
                    file_stat = os.stat(file_path)
                except OSError:
                    continue
                relative_path = os.path.relpath(file_path, self.training_data_path)
                modified_ns = getattr(file_stat, 'st_mtime_ns', int(file_stat.st_mtime * 1_000_000_000))
                signature_items.append(f'{relative_path}:{file_stat.st_size}:{modified_ns}')

        if not signature_items:
            return None

        return hashlib.sha256('\n'.join(signature_items).encode('utf-8')).hexdigest()

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.face_recognizer.write(self.model_path)
        metadata = {
            'method': self.method,
            'unknown_threshold': self.unknown_threshold,
            'label_name_map': {str(key): value for key, value in self.label_name_map.items()},
            'dataset_signature': self.dataset_signature(),
        }
        with open(self.model_meta_path, 'w', encoding='utf-8') as meta_file:
            json.dump(metadata, meta_file, ensure_ascii=False, indent=2)

    def load_model(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.model_meta_path):
            self.invalidate_model(remove_persisted=False)
            return False

        try:
            with open(self.model_meta_path, 'r', encoding='utf-8') as meta_file:
                metadata = json.load(meta_file)
        except (OSError, json.JSONDecodeError):
            self.invalidate_model(remove_persisted=True)
            return False

        if metadata.get('method') != self.method:
            self.invalidate_model(remove_persisted=True)
            return False

        current_signature = self.dataset_signature()
        saved_signature = metadata.get('dataset_signature')
        if current_signature != saved_signature:
            self.invalidate_model(remove_persisted=True)
            return False

        try:
            self.face_recognizer = self.create_face_recognizer()
            self.face_recognizer.read(self.model_path)
        except cv2.error:
            self.invalidate_model(remove_persisted=True)
            return False

        self.label_name_map = {
            int(key): value for key, value in metadata.get('label_name_map', {}).items()
        }
        self.unknown_threshold = self.DEFAULT_UNKNOWN_THRESHOLD
        self.trained = bool(self.label_name_map)
        if not self.trained:
            self.invalidate_model(remove_persisted=True)
            return False
        return True

    def train(self, putLog):
        if self.training:
            putLog("经典模型正在训练中，请稍候")
            return False

        if not os.path.isdir(self.training_data_path):
            putLog("训练失败：未找到本地样本目录 dataset/full")
            self.invalidate_model(remove_persisted=True)
            return False

        sample_dirs = [
            dir_name for dir_name in os.listdir(self.training_data_path)
            if os.path.isdir(os.path.join(self.training_data_path, dir_name)) and not dir_name.startswith(".")
        ]
        if not sample_dirs:
            putLog("训练失败：dataset/full 中还没有可用样本")
            self.invalidate_model(remove_persisted=True)
            return False

        self.training = True
        self.invalidate_model(remove_persisted=True)
        self.training_logger = putLog
        self.training_thread = ClassicTrainingDataThread(self.vision, self.training_data_path, self.method)
        self.training_thread.log_signal.connect(putLog)
        self.training_thread.finish_signal.connect(self.finish_training)
        self.training_thread.start()
        putLog("开始训练经典模型")
        return True

    def finish_training(self):
        self.training = False

        if not self.training_thread.faces or len(self.training_thread.labels) == 0:
            self.invalidate_model(remove_persisted=True)
            if self.training_logger:
                self.training_logger("训练失败：没有检测到可用于训练的人脸样本")
            return

        try:
            self.face_recognizer.train(self.training_thread.faces, self.training_thread.labels)
        except cv2.error as exc:
            self.invalidate_model(remove_persisted=True)
            if self.training_logger:
                self.training_logger(f"训练失败：{exc}")
            return

        self.label_name_map = dict(self.training_thread.label_name_map)
        self.trained = True
        try:
            self.save_model()
        except (cv2.error, OSError, TypeError, ValueError) as exc:
            self.invalidate_model(remove_persisted=True)
            if self.training_logger:
                self.training_logger(f"训练失败：无法保存经典模型 ({exc})")
            return
        if self.training_logger:
            self.training_logger(f"训练完成：已载入 {len(self.label_name_map)} 个身份样本，并已保存本地模型")

    def detect_face(self, frame):
        """
        人脸检测
        :param frame: 一帧图像
        :return: 人脸坐标
        """
        return self.vision.lbp_detect_face(frame)

    def predict(self, faces):
        """
        识别测试图像中的人脸
        :param input_img: 输入的预测图像
        :return: 预测完成的图像、对应的id和概率
        """
        if not self.trained:
            raise RuntimeError("经典模型尚未训练")

        label_text = (None, None)
        # for face in faces:
        #     face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        #     # 使用我们的脸部识别器预测图像
        #     label = self.face_recognizer.predict(cv2.cvtColor(cv2.resize(face, (160, 160)), cv2.COLOR_BGR2GRAY))
        #     # 获取由人脸识别器返回的相应标签的名称
        #     label_text = label
        #     # 画预计人的名字
        #     # draw_text(img, f'B{label_text[0]} possibility: {label_text[1]}', face[0], face[1] - 5)
        # 使用我们的脸部识别器预测图像
        label = self.face_recognizer.predict(cv2.cvtColor(cv2.resize(faces, (160, 160)), cv2.COLOR_BGR2GRAY))
        label_text = label
        label_id = int(label_text[0])
        label_name = self.label_name_map.get(label_id)
        confidence = float(label_text[1])
        matched = label_name is not None and confidence <= self.unknown_threshold
        return {
            'name': label_name if matched else '未知',
            'matched': matched,
            'score': confidence,
            'candidate_name': label_name if label_name is not None else '未知',
            'candidate_score': confidence,
        }


class ClassicTrainingDataThread(QThread):
    """
    传统方法下的比如说LBPH和PCA方法的训练线程
    """
    log_signal = pyqtSignal(str)
    finish_signal = pyqtSignal()

    def __init__(self, vision_tools, data_folder_path, method='lbph'):
        super(ClassicTrainingDataThread, self).__init__()
        self.vision_tools = vision_tools
        self.data_folder_path = data_folder_path
        self.method = method
        self.faces = []
        self.labels = []
        self.label_name_map = {}

    def run(self):
        self.log_signal.emit("正在准备训练数据...")
        faces, labels, label_name_map = self.vision_tools.prepare_training_data(
            self.data_folder_path,
            self.log_signal.emit,
            self.method,
        )
        self.faces = faces
        self.labels = labels
        self.label_name_map = label_name_map
        self.log_signal.emit("训练数据准备完成")
        self.finish_signal.emit()


class MindFaceService:
    def __init__(self, detectCfg):
        if not MINDSPORE_AVAILABLE:
            raise RuntimeError('MindSpore is not available in the current environment.')
        self.detect_options = {}
        self.recog_options = {}
        self.face_detect_prepare(detectCfg)
        self.recog_prepare()

    def face_detect_prepare(self, cfg):
        if cfg['mode'] == 'Graph':
            context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
        else:
            context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
        if cfg['name'] == 'ResNet50':
            backbone = resnet50(1001)
        elif cfg['name'] == 'MobileNet025':
            backbone = mobilenet025(1000)
        self.detect_options['network'] = RetinaFace(phase='predict', backbone=backbone,
                                                    in_channel=cfg['in_channel'],
                                                    out_channel=cfg['out_channel'])
        backbone.set_train(False)
        self.detect_options['network'].set_train(False)
        # load checkpoint
        assert cfg['val_model'] is not None, 'val_model is None.'
        param_dict = load_checkpoint(cfg['val_model'])
        print(f"Load trained model done. {cfg['val_model']}")
        self.detect_options['network'].init_parameters_data()
        load_param_into_net(self.detect_options['network'], param_dict)
        # testing image
        self.detect_options['conf_test'] = cfg['conf']
        self.detect_options['detection'] = DetectionEngine(nms_thresh=cfg['val_nms_threshold'],
                                                           conf_thresh=cfg['val_confidence_threshold'],
                                                           iou_thresh=cfg['val_iou_threshold'], var=cfg['variance'])
        self.detect_options['target_size'] = 1600
        self.detect_options['max_size'] = 2176
        self.detect_options['priors'] = prior_box(
            image_sizes=(self.detect_options['max_size'], self.detect_options['max_size']),
            min_sizes=[[16, 32], [64, 128], [256, 512]],
            steps=[8, 16, 32],
            clip=False)

    def face_detect(self, cfg, frame):
        """
        基于retinaface做的面部检测
        """
        img_raw = frame.copy()
        img = np.float32(img_raw)
        im_size_min = np.min(img.shape[0:2])
        im_size_max = np.max(img.shape[0:2])
        resize = float(self.detect_options['target_size']) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > self.detect_options['max_size']:
            resize = float(self.detect_options['max_size']) / float(im_size_max)
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        assert img.shape[0] <= self.detect_options['max_size'] and img.shape[1] <= self.detect_options['max_size']
        image_t = np.empty((self.detect_options['max_size'], self.detect_options['max_size'], 3), dtype=img.dtype)
        image_t[:, :] = (104.0, 117.0, 123.0)
        image_t[0:img.shape[0], 0:img.shape[1]] = img
        img = image_t
        scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]], dtype=img.dtype)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        img = Tensor(img)
        boxes, confs, _ = self.detect_options['network'](img)
        boxes = self.detect_options['detection'].infer(boxes, confs, resize, scale, self.detect_options['priors'])
        return boxes

    def recog_prepare(self, backbone="mobilefacenet", num_features=512,
                      pretrained='utils/mindface/recognition/pretrained/mobile_casia_ArcFace.ckpt'):
        if backbone == 'iresnet50':
            self.recog_options['recog_model'] = iresnet50(num_features=num_features)
            print("Finish loading iresnet50")
        elif backbone == 'iresnet100':
            self.recog_options['recog_model'] = iresnet100(num_features=num_features)
            print("Finish loading iresnet100")
        elif backbone == 'mobilefacenet':
            self.recog_options['recog_model'] = get_mbf(num_features=num_features)
            print("Finish loading mobilefacenet")
        elif backbone == 'vit_t':
            self.recog_options['recog_model'] = vit_t(num_features=num_features)
            print("Finish loading vit_t")
        elif backbone == 'vit_s':
            self.recog_options['recog_model'] = vit_s(num_features=num_features)
            print("Finish loading vit_s")
        elif backbone == 'vit_b':
            self.recog_options['recog_model'] = vit_b(num_features=num_features)
            print("Finish loading vit_b")
        elif backbone == 'vit_l':
            self.recog_options['recog_model'] = vit_l(num_features=num_features)
            print("Finish loading vit_l")
        else:
            raise NotImplementedError
        if pretrained:
            param_dict = load_checkpoint(pretrained)
            load_param_into_net(self.recog_options['recog_model'], param_dict)

    def recog_infer(self, img):
        """
        The inference of arcface.

        Args:
            img (NumPy): The input image.
            backbone (Object): Arcface model without loss function. Default: "iresnet50".
            pretrained (Bool): Pretrain. Default: False.
        """
        assert (img.shape[-1] == 112 and img.shape[-2] == 112)
        img = ((img / 255) - 0.5) / 0.5
        img = ms.Tensor(img, ms.float32)
        if len(img.shape) == 4:
            pass
        elif len(img.shape) == 3:
            img = img.expand_dims(axis=0)
        net_out = self.recog_options['recog_model'](img)
        embeddings = net_out.asnumpy()
        return embeddings

    def compare_embedding(self, emb1, emb2):
        """
        计算特征向量差异
        :param emb1: 特征向量1
        :param emb2: 特征向量2
        :return:
        """
        return np.linalg.norm(emb1 - emb2, axis=1)


class RecogThread(QThread):
    """
    识别线程
    """

    def __init__(self, recogQueue, resultSignal, classsicFaceRecognizer, mindface):
        super(RecogThread, self).__init__()
        self.vision_tools = VisionTools()
        self.recogQueue = recogQueue
        self.resultSignal = resultSignal
        self.classicFaceRecognizer = classsicFaceRecognizer
        self.mindface = mindface
        self.config = ConfigStore()
        self.faces = []
        self.labels = []
        self.http = requests.Session()

    def run(self):
        while True:
            try:
                faces = self.recogQueue.get(timeout=0.2)
            except Empty:
                continue

            try:
                if self.config.get_config('recog_method') == self.config.recog_methods_mapper['classic']:
                    if not self.classicFaceRecognizer.trained:
                        continue
                    payload = classicFaceRecognize(faces, self.classicFaceRecognizer)
                    display_name = select_display_name(payload)
                    print(
                        f"score: {payload.get('score')} "
                        f"name: {payload.get('name')} "
                        f"candidate: {payload.get('candidate_name')}"
                    )
                    if display_name:
                        self.resultSignal.emit(str(display_name))
                if self.config.get_config('recog_method') in (
                    self.config.recog_methods_mapper['mindspore'],
                    self.config.recog_methods_mapper['cnn_classifier'],
                ):
                    photo_bytes = encode_face_image(faces)
                    url = f"{FACE_SERVICE_BASE_URL}/recognize"
                    payload_data = None
                    if (
                        self.config.get_config('recog_method')
                        == self.config.recog_methods_mapper['cnn_classifier']
                    ):
                        payload_data = {'backend': 'cnn_classifier'}
                    response = self.http.post(
                        url,
                        data=payload_data,
                        files={'photo': ('recog.jpg', photo_bytes, 'image/jpeg')},
                        timeout=8,
                    )
                    print(response)
                    if response.status_code == 200:
                        payload = response.json()
                        self.resultSignal.emit(select_display_name(payload))
            except (RuntimeError, ValueError, cv2.error, requests.RequestException) as exc:
                print(f"recognition error: {exc}")
                continue
            except Exception as exc:
                print(f"unexpected recognition error: {exc}")
                continue

class RegisterThread(QThread):
    def __init__(self, registerResultSignal):
        super(RegisterThread, self).__init__()
        self.config = ConfigStore()
        self.registerQueue = self.config.registerQueue
        self.registerResultSignal = registerResultSignal
        self.http = requests.Session()

    def run(self):
        while True:
            try:
                faces, name = self.registerQueue.get(timeout=0.2)
            except Empty:
                continue

            clean_name = name.strip()
            if not clean_name:
                self.registerResultSignal.emit("注册失败：姓名不能为空")
                continue

            local_dir = os.path.join('dataset', 'full', clean_name)

            try:
                photo_bytes = encode_face_image(faces)
                os.makedirs(local_dir, exist_ok=True)

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                local_path = os.path.join(local_dir, f'{timestamp}.jpg')
                with open(local_path, 'wb') as sample_file:
                    sample_file.write(photo_bytes)

                url = f"{FACE_SERVICE_BASE_URL}/register"
                response = self.http.post(
                    url,
                    data={'name': clean_name},
                    files={'photo': ('register.jpg', photo_bytes, 'image/jpeg')},
                    timeout=8,
                )

                print(response)
                if response.status_code == 200:
                    self.registerResultSignal.emit("注册成功：已保存本地样本并同步到服务")
                else:
                    self.registerResultSignal.emit(
                        f"本地采集成功：远端服务返回 {response.status_code}"
                    )
            except requests.RequestException:
                self.registerResultSignal.emit("本地采集成功：远端注册服务不可用")
            except (OSError, ValueError):
                self.registerResultSignal.emit("注册失败：读写注册文件时出错")
