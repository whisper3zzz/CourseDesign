from utils.common import singleton
from queue import Queue


@singleton
class ConfigStore:
    recogQueue = Queue(maxsize=1)  # 识别队列
    registerQueue = Queue()  # 注册队列

    def __init__(self):
        self.config = {
            'camera_on': False,
            'recog_open_disabled': False,
            'face_detect': False,
            'detect_method': 0,
            'recog_open': False,
            'recog_method': 0,
        }
        self.recog_methods_mapper = {
            'classic': 0,
            'mindspore': 1,
            'cnn_classifier': 2,
        }
        self.detect_methods_mapper = {
            'classic': 0,
            'mediapipe': 1,
            'mindspore': 2
        }

    def get_config(self, key):
        return self.config[key]

    def set_config(self, key, value):
        self.config[key] = value
        return True

    def get_all_config(self):
        return self.config
