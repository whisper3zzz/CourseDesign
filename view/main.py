import os
import shutil
import threading
from datetime import datetime
from queue import Queue

import cv2
from PyQt5.QtCore import QPoint, QPropertyAnimation, QEasingCurve, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPixmap, QTextCursor
from PyQt5.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
)

from store.config import ConfigStore
from view.mainDisplay import Ui_System


class MainPage(QMainWindow, Ui_System):
    logQueue = Queue()  # 日志队列
    receiveLogSignal = pyqtSignal(str)  # LOG 信号状态

    def __init__(self, all_queues, classicPredictor, visionService, config):
        super(MainPage, self).__init__()
        self.setupUi(self)
        self.visionService = visionService
        self.config = config
        self.all_queues = all_queues
        self.classicPredictor = classicPredictor
        self._animations = []
        self.camera_ready = False
        self.camera_connecting = False
        self.current_display_frame = None
        self.current_result_frame = None
        self.sample_root = os.path.join("dataset", "full")
        self.sample_identity_count = 0
        self.sample_file_count = 0

        self.setup_chrome()

        # 摄像头开关设置
        self.switchButton.clicked.connect(self.switch_camera)

        # 摄像头帧轮询
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # 人脸检测设置
        self.openDetection.stateChanged.connect(self.open_detection)
        self.detectMethod.currentIndexChanged.connect(self.change_detect_method)

        # 人脸识别设置
        self.openRecognition.stateChanged.connect(self.open_recognition)
        self.recogMethod.currentIndexChanged.connect(self.change_recog_method)

        # 日志系统
        self.receiveLogSignal.connect(lambda log: self.logOutput(log))
        self.logOutputThread = threading.Thread(target=self.receiveLog, daemon=True)
        self.logOutputThread.start()

        # 训练按钮
        self.startTrainButton.clicked.connect(self.start_train)
        self.deleteSampleButton.clicked.connect(self.delete_current_sample)
        self.clearSamplesButton.clicked.connect(self.clear_all_samples)

        # 人脸注册按钮
        self.imgRegister.clicked.connect(self.register_face)
        self.nameInput.textChanged.connect(self.handle_name_input_changed)
        self.visionService.registerResSignal.connect(lambda res: self.handleRegister(res))
        self.loadMindSporeFace.clicked.connect(self.show_mindspore_status)
        self.visionService.cameraStatusSignal.connect(self.handle_camera_status)

        # 识别结果显示
        self.visionService.resultSignal.connect(lambda result: self.recogOutput(result))

        self.refresh_sample_overview()
        self.update_status_labels()
        self.animate_intro()

        if self.classicPredictor.trained:
            self.putLog(
                f"已加载本地经典模型：{len(self.classicPredictor.label_name_map)} 个身份，可直接开启经典识别"
            )

    def setup_chrome(self):
        self.setWindowTitle("神经扫描终端")
        self.setMaximumSize(16777215, 16777215)
        self.setMinimumSize(1024, 640)
        self.resize(1280, 760)

        self.leftCard = QFrame(self)
        self.leftCard.setObjectName("leftCard")
        self.leftCard.setGeometry(12, 12, 210, 486)

        self.previewCard = QFrame(self)
        self.previewCard.setObjectName("previewCard")
        self.previewCard.setGeometry(230, 12, 680, 486)

        self.rightCard = QFrame(self)
        self.rightCard.setObjectName("rightCard")
        self.rightCard.setGeometry(926, 12, 242, 486)

        self.accentOrbOne = QFrame(self)
        self.accentOrbOne.setObjectName("accentOrbOne")
        self.accentOrbOne.setGeometry(780, 12, 120, 120)

        self.accentOrbTwo = QFrame(self)
        self.accentOrbTwo.setObjectName("accentOrbTwo")
        self.accentOrbTwo.setGeometry(1048, 392, 96, 96)

        self.subtitleLabel = QLabel(self)
        self.subtitleLabel.setObjectName("subtitleLabel")
        self.subtitleLabel.setGeometry(28, 60, 170, 20)
        self.subtitleLabel.setText("实时生物识别终端")

        self.logSectionLabel = QLabel(self)
        self.logSectionLabel.setObjectName("logSectionLabel")
        self.logSectionLabel.setGeometry(28, 92, 170, 18)
        self.logSectionLabel.setText("系统事件")

        self.previewTitleLabel = QLabel(self)
        self.previewTitleLabel.setObjectName("previewTitleLabel")
        self.previewTitleLabel.setGeometry(256, 34, 240, 28)
        self.previewTitleLabel.setText("神经视觉画面")

        self.previewSubtitleLabel = QLabel(self)
        self.previewSubtitleLabel.setObjectName("previewSubtitleLabel")
        self.previewSubtitleLabel.setGeometry(256, 64, 420, 18)
        self.previewSubtitleLabel.setText("连接摄像头后，可直接开始识别")

        self.cameraBadge = QLabel(self)
        self.cameraBadge.setObjectName("cameraBadge")
        self.cameraBadge.setGeometry(736, 32, 152, 30)
        self.cameraBadge.setAlignment(Qt.AlignCenter)

        self.controlSubtitleLabel = QLabel(self)
        self.controlSubtitleLabel.setObjectName("controlSubtitleLabel")
        self.controlSubtitleLabel.setGeometry(950, 62, 180, 18)
        self.controlSubtitleLabel.setText("开始识别后会自动锁定人脸")

        self.resultHintLabel = QLabel(self)
        self.resultHintLabel.setObjectName("resultHintLabel")
        self.resultHintLabel.setGeometry(950, 324, 180, 18)
        self.resultHintLabel.setText("连接摄像头后，可直接开始识别")

        self.leftNodeLabel = QLabel(self)
        self.leftNodeLabel.setObjectName("leftNodeLabel")
        self.leftNodeLabel.setGeometry(28, 290, 176, 16)
        self.leftNodeLabel.setText("步骤 1 / 连接摄像头")

        self.sampleStatsLabel = QLabel(self)
        self.sampleStatsLabel.setObjectName("sampleStatsLabel")
        self.sampleStatsLabel.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.sampleStatsLabel.setWordWrap(True)

        self.deleteSampleButton = QPushButton(self)
        self.deleteSampleButton.setObjectName("deleteSampleButton")
        self.deleteSampleButton.setText("删除当前样本")

        self.clearSamplesButton = QPushButton(self)
        self.clearSamplesButton.setObjectName("clearSamplesButton")
        self.clearSamplesButton.setText("清空全部样本")

        self.previewMetaLabel = QLabel(self)
        self.previewMetaLabel.setObjectName("previewMetaLabel")
        self.previewMetaLabel.setGeometry(256, 34, 632, 28)
        self.previewMetaLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.previewMetaLabel.setText("画面 640x480  /  实时视频流")

        self.previewBeamTop = QFrame(self)
        self.previewBeamTop.setObjectName("previewBeamTop")
        self.previewBeamTop.setGeometry(274, 88, 596, 2)

        self.previewBeamBottom = QFrame(self)
        self.previewBeamBottom.setObjectName("previewBeamBottom")
        self.previewBeamBottom.setGeometry(274, 480, 596, 2)

        self.label.setGeometry(28, 28, 170, 30)
        self.logText.setGeometry(28, 114, 176, 176)
        self.nameInput.setGeometry(28, 312, 176, 40)
        self.sampleStatsLabel.setGeometry(28, 358, 176, 34)
        self.imgRegister.setGeometry(28, 362, 176, 36)
        self.startTrainButton.setGeometry(28, 406, 176, 36)
        self.deleteSampleButton.setGeometry(28, 450, 84, 36)
        self.clearSamplesButton.setGeometry(120, 450, 84, 36)
        self.switchButton.setGeometry(28, 494, 176, 36)

        self.displayField.setGeometry(252, 104, 636, 366)
        self.label_2.setGeometry(950, 36, 120, 24)
        self.openDetection.setGeometry(950, 94, 160, 22)
        self.detectMethod.setGeometry(950, 122, 194, 38)
        self.openRecognition.setGeometry(950, 176, 160, 22)
        self.recogMethod.setGeometry(950, 204, 194, 38)
        self.loadMindSporeFace.setGeometry(950, 256, 194, 38)
        self.label_3.setGeometry(950, 296, 120, 24)
        self.resultFace.setGeometry(986, 354, 122, 96)
        self.resultLabel.setGeometry(968, 458, 160, 26)

        self.label.setText("扫描核心")
        self.label_2.setText("识别控制")
        self.label_3.setText("识别目标")
        self.switchButton.setText("连接摄像头")
        self.startTrainButton.setText("用本地样本训练")
        self.imgRegister.setText("采集当前人脸")
        self.loadMindSporeFace.setText("查看深度服务状态")
        self.nameInput.setPlaceholderText("输入姓名后采集当前人脸")
        self.displayField.setText("实时扫描画面将在这里启动")
        self.displayField.setAlignment(Qt.AlignCenter)
        self.displayField.setWordWrap(True)
        self.resultFace.setText("等待识别")
        self.resultFace.setAlignment(Qt.AlignCenter)
        self.resultFace.setWordWrap(True)
        self.resultLabel.setText("等待目标")
        self.resultLabel.setAlignment(Qt.AlignCenter)
        self.leftNodeLabel.setWordWrap(True)
        self.leftNodeLabel.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.logText.setReadOnly(True)
        self.logText.setAcceptRichText(False)
        self.recogMethod.setItemText(0, "经典识别")
        self.recogMethod.setItemText(1, "深度识别")
        self.openDetection.setText("显示锁脸框")
        self.openRecognition.setText("开始自动识别")
        self.detectMethod.setItemText(0, "经典锁脸")
        self.detectMethod.setItemText(1, "轻量锁脸")
        self.detectMethod.setItemText(2, "深度锁脸")

        self.setStyleSheet(
            """
            QMainWindow#System {
                background-color: #07111d;
            }
            QFrame#leftCard {
                background-color: #0a1726;
                border: 1px solid #123049;
                border-radius: 28px;
            }
            QFrame#previewCard {
                background-color: #091724;
                border: 1px solid #143850;
                border-radius: 30px;
            }
            QFrame#rightCard {
                background-color: #0b1828;
                border: 1px solid #14354d;
                border-radius: 28px;
            }
            QFrame#accentOrbOne {
                background-color: rgba(31, 206, 255, 42);
                border-radius: 60px;
            }
            QFrame#accentOrbTwo {
                background-color: rgba(0, 255, 170, 32);
                border-radius: 48px;
            }
            QFrame#previewBeamTop,
            QFrame#previewBeamBottom {
                background-color: rgba(41, 235, 255, 184);
                border-radius: 1px;
            }
            QLabel#label {
                color: #d7efff;
                font-size: 26px;
                font-weight: 700;
                font-family: "Avenir Next Condensed", "Avenir Next", "PingFang SC";
                letter-spacing: 2px;
            }
            QLabel#subtitleLabel {
                color: rgba(92, 214, 255, 190);
                font-size: 12px;
                font-weight: 600;
                font-family: "Menlo";
            }
            QLabel#logSectionLabel {
                color: rgba(210, 238, 255, 215);
                font-size: 13px;
                font-weight: 600;
                font-family: "Menlo";
            }
            QLabel#previewTitleLabel {
                color: #d4f4ff;
                font-size: 24px;
                font-weight: 700;
                font-family: "Avenir Next Condensed", "Avenir Next", "PingFang SC";
                letter-spacing: 1px;
            }
            QLabel#previewSubtitleLabel {
                color: #7fbad0;
                font-size: 12px;
                font-weight: 500;
                font-family: "Menlo";
            }
            QLabel#cameraBadge {
                border-radius: 15px;
                font-size: 11px;
                font-weight: 700;
                font-family: "Menlo";
                padding: 0 12px;
                letter-spacing: 1px;
            }
            QLabel#label_2,
            QLabel#label_3 {
                color: #d8f2ff;
                font-size: 18px;
                font-weight: 700;
                font-family: "Avenir Next Condensed", "Avenir Next", "PingFang SC";
                letter-spacing: 1px;
            }
            QLabel#controlSubtitleLabel,
            QLabel#resultHintLabel {
                color: #7cb1c9;
                font-size: 12px;
                font-weight: 500;
                font-family: "Menlo";
            }
            QLabel#previewMetaLabel {
                color: rgba(90, 229, 255, 205);
                font-size: 11px;
                font-weight: 600;
                font-family: "Menlo";
                letter-spacing: 1px;
            }
            QLabel#leftNodeLabel {
                color: #d7f6ff;
                background-color: rgba(8, 27, 45, 224);
                border: 1px solid rgba(47, 149, 191, 122);
                border-radius: 18px;
                font-size: 12px;
                font-weight: 600;
                font-family: "Avenir Next", "PingFang SC";
                padding: 10px 12px;
            }
            QLabel#sampleStatsLabel {
                color: #a9dff2;
                background-color: rgba(7, 24, 38, 196);
                border: 1px solid rgba(42, 117, 150, 128);
                border-radius: 14px;
                font-size: 12px;
                font-weight: 600;
                font-family: "Menlo";
                padding: 8px 10px;
            }
            QLabel#displayField {
                background-color: #04111c;
                border: 1px solid #1b607d;
                border-radius: 24px;
                color: #70dfff;
                font-size: 18px;
                font-weight: 600;
                font-family: "Avenir Next", "PingFang SC";
                padding: 18px;
            }
            QLabel#resultFace {
                background-color: #06131d;
                border: 1px solid #16546f;
                border-radius: 24px;
                color: #74d8ff;
                font-size: 14px;
                font-weight: 600;
                font-family: "Menlo";
                padding: 10px;
            }
            QLabel#resultLabel {
                color: #e0f8ff;
                font-size: 18px;
                font-weight: 700;
                font-family: "Avenir Next Condensed", "Avenir Next", "PingFang SC";
            }
            QTextEdit#logText {
                background-color: rgba(5, 18, 31, 240);
                border: 1px solid rgba(39, 117, 153, 140);
                border-radius: 18px;
                color: #86dbff;
                padding: 12px;
                selection-background-color: #1b90c2;
                font-size: 12px;
                font-family: "Menlo";
            }
            QTextEdit#logText QScrollBar:vertical {
                width: 10px;
                margin: 6px 2px 6px 0;
                background: transparent;
            }
            QTextEdit#logText QScrollBar::handle:vertical {
                background: rgba(84, 225, 255, 97);
                border-radius: 5px;
                min-height: 28px;
            }
            QTextEdit#logText QScrollBar::add-line:vertical,
            QTextEdit#logText QScrollBar::sub-line:vertical {
                height: 0;
            }
            QLineEdit#nameInput {
                background-color: rgba(5, 20, 33, 235);
                border: 1px solid rgba(36, 116, 152, 148);
                border-radius: 16px;
                color: #dff7ff;
                padding: 0 14px;
                font-size: 13px;
                font-weight: 500;
                font-family: "Menlo";
            }
            QLineEdit#nameInput:focus {
                border: 1px solid rgba(63, 233, 255, 242);
            }
            QPushButton {
                border: none;
                border-radius: 16px;
                font-size: 14px;
                font-weight: 700;
                font-family: "Menlo";
                letter-spacing: 1px;
            }
            QPushButton#switchButton {
                background-color: #0d5a7b;
                color: #dff9ff;
            }
            QPushButton#switchButton:hover {
                background-color: #11709b;
            }
            QPushButton#startTrainButton {
                background-color: #123d76;
                color: #dff5ff;
            }
            QPushButton#startTrainButton:hover {
                background-color: #175191;
            }
            QPushButton#imgRegister {
                background-color: #0e7a63;
                color: #e6fff8;
            }
            QPushButton#imgRegister:hover {
                background-color: #12957a;
            }
            QPushButton#deleteSampleButton {
                background-color: #75402d;
                color: #fff0e7;
                font-size: 12px;
            }
            QPushButton#deleteSampleButton:hover {
                background-color: #8e5039;
            }
            QPushButton#clearSamplesButton {
                background-color: #5f2430;
                color: #ffe6ec;
                font-size: 12px;
            }
            QPushButton#clearSamplesButton:hover {
                background-color: #7c3040;
            }
            QPushButton#loadMindSporeFace {
                background-color: #081b2d;
                border: 1px solid #1a5f7f;
                color: #b8f0ff;
            }
            QPushButton#loadMindSporeFace:hover {
                background-color: #0b2238;
            }
            QPushButton:disabled {
                background-color: #183042;
                color: #6f8ea0;
            }
            QCheckBox {
                color: #daf4ff;
                font-size: 13px;
                font-weight: 600;
                font-family: "Avenir Next", "PingFang SC";
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid #2a7597;
                background-color: #081421;
            }
            QCheckBox::indicator:checked {
                background-color: #1be0ff;
                border: 4px solid #081421;
            }
            QCheckBox::indicator:disabled {
                background-color: #122433;
                border: 1px solid #27455d;
            }
            QComboBox {
                background-color: #071625;
                border: 1px solid #215f7d;
                border-radius: 16px;
                color: #dff5ff;
                padding: 0 14px;
                font-size: 13px;
                font-weight: 600;
                font-family: "Menlo";
            }
            QComboBox::drop-down {
                border: none;
                width: 28px;
            }
            QComboBox QAbstractItemView {
                background-color: #091b2b;
                border: 1px solid #215f7d;
                color: #d7f6ff;
                selection-background-color: #0a8dba;
                selection-color: #f0fdff;
                padding: 6px;
            }
            """
        )

        for widget, blur, offset_y in (
            (self.leftCard, 34, 12),
            (self.previewCard, 30, 10),
            (self.rightCard, 30, 10),
            (self.displayField, 20, 6),
            (self.resultFace, 18, 6),
        ):
            self.apply_shadow(widget, blur, offset_y)

        for widget in (
            self.leftCard,
            self.previewCard,
            self.rightCard,
            self.accentOrbOne,
            self.accentOrbTwo,
            self.previewBeamTop,
            self.previewBeamBottom,
        ):
            widget.lower()

        for widget in (
            self.label,
            self.subtitleLabel,
            self.logSectionLabel,
            self.logText,
            self.nameInput,
            self.sampleStatsLabel,
            self.imgRegister,
            self.startTrainButton,
            self.deleteSampleButton,
            self.clearSamplesButton,
            self.switchButton,
            self.previewTitleLabel,
            self.previewSubtitleLabel,
            self.previewMetaLabel,
            self.cameraBadge,
            self.displayField,
            self.label_2,
            self.controlSubtitleLabel,
            self.openDetection,
            self.detectMethod,
            self.openRecognition,
            self.recogMethod,
            self.loadMindSporeFace,
            self.label_3,
            self.resultHintLabel,
            self.resultFace,
            self.resultLabel,
            self.leftNodeLabel,
        ):
            widget.raise_()

        self.layout_chrome()

    def clamp(self, value, minimum, maximum):
        return max(minimum, min(maximum, value))

    def layout_chrome(self):
        width = max(self.width(), self.minimumWidth())
        height = max(self.height(), self.minimumHeight())

        margin = self.clamp(int(min(width, height) * 0.02), 12, 24)
        gap = self.clamp(int(width * 0.014), 12, 20)
        card_height = height - margin * 2

        left_width = self.clamp(int(width * 0.19), 210, 270)
        right_width = self.clamp(int(width * 0.22), 240, 310)
        preview_width = width - margin * 2 - gap * 2 - left_width - right_width

        if preview_width < 460:
            shortage = 460 - preview_width
            shrink_left = min(shortage // 2, left_width - 190)
            left_width -= max(0, shrink_left)
            shortage -= max(0, shrink_left)
            shrink_right = min(shortage, right_width - 220)
            right_width -= max(0, shrink_right)
            preview_width = width - margin * 2 - gap * 2 - left_width - right_width

        left_x = margin
        preview_x = left_x + left_width + gap
        right_x = preview_x + preview_width + gap
        top_y = margin

        self.leftCard.setGeometry(left_x, top_y, left_width, card_height)
        self.previewCard.setGeometry(preview_x, top_y, preview_width, card_height)
        self.rightCard.setGeometry(right_x, top_y, right_width, card_height)

        orb_one_size = self.clamp(int(preview_width * 0.16), 96, 136)
        self.accentOrbOne.setGeometry(
            preview_x + preview_width - orb_one_size + 24,
            top_y - 18,
            orb_one_size,
            orb_one_size,
        )

        orb_two_size = self.clamp(int(right_width * 0.36), 86, 116)
        self.accentOrbTwo.setGeometry(
            right_x + right_width - orb_two_size + 18,
            top_y + card_height - orb_two_size + 14,
            orb_two_size,
            orb_two_size,
        )

        left_pad = self.clamp(int(left_width * 0.1), 18, 26)
        left_inner_width = left_width - left_pad * 2
        button_height = self.clamp(int(card_height * 0.075), 36, 42)
        field_height = self.clamp(int(card_height * 0.08), 40, 46)
        hint_height = self.clamp(int(card_height * 0.08), 44, 56)
        stats_height = self.clamp(int(card_height * 0.055), 34, 42)
        control_gap = self.clamp(int(card_height * 0.016), 10, 14)
        section_gap = self.clamp(int(card_height * 0.022), 14, 22)

        self.label.setGeometry(left_x + left_pad, top_y + 18, left_inner_width, 30)
        self.subtitleLabel.setGeometry(left_x + left_pad, top_y + 52, left_inner_width, 20)
        self.logSectionLabel.setGeometry(left_x + left_pad, top_y + 86, left_inner_width, 18)

        switch_y = top_y + card_height - left_pad - button_height
        manage_y = switch_y - control_gap - button_height
        train_y = manage_y - control_gap - button_height
        register_y = train_y - control_gap - button_height
        name_y = register_y - control_gap - field_height
        stats_y = name_y - control_gap - stats_height
        hint_y = stats_y - control_gap - hint_height

        self.leftNodeLabel.setGeometry(left_x + left_pad, hint_y, left_inner_width, hint_height)
        self.sampleStatsLabel.setGeometry(left_x + left_pad, stats_y, left_inner_width, stats_height)
        self.nameInput.setGeometry(left_x + left_pad, name_y, left_inner_width, field_height)
        self.imgRegister.setGeometry(left_x + left_pad, register_y, left_inner_width, button_height)
        self.startTrainButton.setGeometry(left_x + left_pad, train_y, left_inner_width, button_height)
        manage_gap = self.clamp(int(left_inner_width * 0.05), 8, 12)
        manage_width = (left_inner_width - manage_gap) // 2
        self.deleteSampleButton.setGeometry(left_x + left_pad, manage_y, manage_width, button_height)
        self.clearSamplesButton.setGeometry(
            left_x + left_pad + manage_width + manage_gap,
            manage_y,
            left_inner_width - manage_width - manage_gap,
            button_height,
        )
        self.switchButton.setGeometry(left_x + left_pad, switch_y, left_inner_width, button_height)

        log_top = top_y + 114
        log_bottom = hint_y - section_gap
        log_height = max(0, log_bottom - log_top)
        self.logText.setGeometry(left_x + left_pad, log_top, left_inner_width, log_height)

        preview_pad = self.clamp(int(preview_width * 0.04), 22, 34)
        preview_inner_width = preview_width - preview_pad * 2
        header_top = top_y + 20
        title_width = self.clamp(int(preview_inner_width * 0.36), 200, 260)
        badge_width = self.clamp(int(preview_width * 0.24), 164, 196)
        badge_height = 32
        badge_x = preview_x + preview_width - preview_pad - badge_width

        self.previewTitleLabel.setGeometry(preview_x + preview_pad, header_top, title_width, 28)
        self.cameraBadge.setGeometry(badge_x, header_top - 2, badge_width, badge_height)

        meta_x = preview_x + preview_pad + title_width + 18
        meta_width = max(0, badge_x - meta_x - 20)
        self.previewMetaLabel.setVisible(meta_width > 150)
        if meta_width > 150:
            self.previewMetaLabel.setGeometry(meta_x, header_top, meta_width, 28)

        self.previewSubtitleLabel.setGeometry(preview_x + preview_pad, header_top + 34, preview_inner_width, 18)

        display_x = preview_x + preview_pad
        display_y = top_y + 92
        display_height = max(260, card_height - 122)
        self.displayField.setGeometry(display_x, display_y, preview_inner_width, display_height)

        beam_x = display_x + 18
        beam_width = max(60, preview_inner_width - 36)
        self.previewBeamTop.setGeometry(beam_x, display_y - 14, beam_width, 2)
        self.previewBeamBottom.setGeometry(beam_x, display_y + display_height + 12, beam_width, 2)
        self.previewMetaLabel.setText(f"画面 {preview_inner_width}×{display_height}  /  实时视频流")

        right_pad = self.clamp(int(right_width * 0.1), 20, 28)
        right_inner_width = right_width - right_pad * 2
        combo_height = 38

        self.label_2.setGeometry(right_x + right_pad, top_y + 24, right_inner_width, 24)
        self.controlSubtitleLabel.setGeometry(right_x + right_pad, top_y + 52, right_inner_width, 18)
        self.openDetection.setGeometry(right_x + right_pad, top_y + 88, right_inner_width, 22)
        self.detectMethod.setGeometry(right_x + right_pad, top_y + 116, right_inner_width, combo_height)
        self.openRecognition.setGeometry(right_x + right_pad, top_y + 170, right_inner_width, 22)
        self.recogMethod.setGeometry(right_x + right_pad, top_y + 198, right_inner_width, combo_height)
        self.loadMindSporeFace.setGeometry(right_x + right_pad, top_y + 248, right_inner_width, combo_height)

        result_face_size = self.clamp(right_inner_width - 34, 118, 156)
        result_face_x = right_x + (right_width - result_face_size) // 2
        result_face_y = top_y + card_height - right_pad - result_face_size - 34
        self.label_3.setGeometry(right_x + right_pad, result_face_y - 54, right_inner_width, 24)
        self.resultHintLabel.setGeometry(right_x + right_pad, result_face_y - 26, right_inner_width, 18)
        self.resultFace.setGeometry(result_face_x, result_face_y, result_face_size, result_face_size)
        self.resultLabel.setGeometry(right_x + right_pad, result_face_y + result_face_size + 10, right_inner_width, 26)

    def apply_shadow(self, widget, blur_radius, offset_y):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(blur_radius)
        shadow.setOffset(0, offset_y)
        shadow.setColor(QColor(15, 204, 255, 34))
        widget.setGraphicsEffect(shadow)

    def animate_intro(self):
        self._animations.clear()
        self.setWindowOpacity(0.0)

        fade_animation = QPropertyAnimation(self, b"windowOpacity", self)
        fade_animation.setDuration(420)
        fade_animation.setStartValue(0.0)
        fade_animation.setEndValue(1.0)
        fade_animation.setEasingCurve(QEasingCurve.OutCubic)
        self._animations.append(fade_animation)
        fade_animation.start()

        for widget, offset, duration, delay in (
            (self.label, 14, 420, 0),
            (self.previewTitleLabel, 18, 520, 80),
            (self.displayField, 24, 580, 120),
            (self.resultFace, 18, 480, 180),
        ):
            end_pos = widget.pos()
            start_pos = end_pos + QPoint(0, offset)
            widget.move(start_pos)
            animation = QPropertyAnimation(widget, b"pos", self)
            animation.setDuration(duration)
            animation.setStartValue(start_pos)
            animation.setEndValue(end_pos)
            animation.setEasingCurve(QEasingCurve.OutCubic)
            self._animations.append(animation)
            QTimer.singleShot(delay, animation.start)

    def update_status_labels(self):
        self.refresh_sample_overview()
        detection_state = "开启" if self.config.get_config("face_detect") else "关闭"
        recognition_state = "开启" if self.config.get_config("recog_open") else "关闭"
        classic_ready = self.classicPredictor.trained and not getattr(self.classicPredictor, "training", False)
        sample_ready = self.sample_identity_count > 0
        self.previewSubtitleLabel.setText(
            f"锁脸 {detection_state} / {self.detectMethod.currentText()}    "
            f"识别 {recognition_state} / {self.recogMethod.currentText()}"
        )

        if self.camera_connecting:
            self.leftNodeLabel.setText("步骤 1 / 正在连接摄像头\n请稍候...")
        elif not self.camera_ready:
            self.leftNodeLabel.setText("步骤 1 / 先连接摄像头\n连接后可直接开始识别或采集")
        elif not self.openDetection.isChecked():
            self.leftNodeLabel.setText("步骤 2 / 可直接开始识别\n系统会自动锁定人脸，也可先手动显示锁脸框")
        elif not self.openRecognition.isChecked():
            if classic_ready:
                self.leftNodeLabel.setText("步骤 3 / 经典模型已就绪\n现在可以直接开始识别，或继续采集新样本")
            else:
                self.leftNodeLabel.setText("步骤 3 / 可采集当前人脸\n如需认出姓名，直接点开始自动识别")
        else:
            self.leftNodeLabel.setText("流程已就绪\n系统正在自动锁脸并实时识别")

        if self.camera_connecting:
            self.cameraBadge.setText("摄像头连接中")
            self.cameraBadge.setStyleSheet(
                "background-color: #0c2d42; color: #d9fbff; border: 1px solid #64edff; border-radius: 15px;"
            )
        elif self.camera_ready:
            self.cameraBadge.setText("摄像头在线")
            self.cameraBadge.setStyleSheet(
                "background-color: #0a4058; color: #bff7ff; border: 1px solid #2fdfff; border-radius: 15px;"
            )
        else:
            self.cameraBadge.setText("摄像头离线")
            self.cameraBadge.setStyleSheet(
                "background-color: #081b2d; color: #7baec7; border: 1px solid #214f68; border-radius: 15px;"
            )

        self.controlSubtitleLabel.setText(
            f"当前锁脸 {self.detectMethod.currentText()} / 当前识别 {self.recogMethod.currentText()}"
        )

        if self.current_result_frame is None:
            if self.camera_connecting:
                self.resultHintLabel.setText("摄像头连接中，请稍候")
            elif not self.camera_ready:
                self.resultHintLabel.setText("连接摄像头后，可直接开始识别")
            elif not self.openDetection.isChecked():
                self.resultHintLabel.setText("点击“开始自动识别”后，系统会自动锁定人脸")
            elif self.recogMethod.currentIndex() == self.config.recog_methods_mapper["classic"] and not classic_ready:
                if getattr(self.classicPredictor, "training", False):
                    self.resultHintLabel.setText("经典模型训练中，完成后即可开始识别")
                elif sample_ready:
                    self.resultHintLabel.setText("本地样本已更新，点击“用本地样本训练”后即可识别")
                else:
                    self.resultHintLabel.setText("经典识别需要先采集样本并训练")
            elif (
                self.recogMethod.currentIndex() == self.config.recog_methods_mapper["classic"]
                and classic_ready
                and not self.openRecognition.isChecked()
            ):
                self.resultHintLabel.setText("经典模型已就绪，可直接开始识别")
            elif not self.openRecognition.isChecked():
                self.resultHintLabel.setText("可直接采集当前人脸，或点击“开始自动识别”")
            else:
                self.resultHintLabel.setText("识别命中后，会在这里显示目标快照")

        self.sync_controls_state()

    def sync_controls_state(self):
        camera_active = self.camera_ready
        camera_busy = self.camera_connecting
        detection_active = self.openDetection.isChecked()
        recognition_active = self.openRecognition.isChecked()
        training_active = getattr(self.classicPredictor, "training", False)
        classic_ready = self.classicPredictor.trained and not getattr(self.classicPredictor, "training", False)
        classic_selected = self.recogMethod.currentIndex() == self.config.recog_methods_mapper["classic"]
        has_name = bool(self.nameInput.text().strip())
        has_samples = self.sample_identity_count > 0

        self.switchButton.setDisabled(camera_busy)
        self.startTrainButton.setDisabled(training_active or not has_samples)
        if training_active:
            self.startTrainButton.setText("训练中...")
        elif classic_ready:
            self.startTrainButton.setText("重新训练本地样本")
        else:
            self.startTrainButton.setText("用本地样本训练")
        self.nameInput.setDisabled(camera_busy or training_active)
        self.imgRegister.setDisabled(not camera_active or camera_busy or training_active)
        self.deleteSampleButton.setDisabled(training_active or not has_name)
        self.clearSamplesButton.setDisabled(training_active or not has_samples)
        self.openDetection.setDisabled(not camera_active or camera_busy)
        self.detectMethod.setDisabled(not camera_active or camera_busy)

        recog_available = camera_active and not camera_busy
        if classic_selected and not classic_ready:
            recog_available = False
        if not recog_available and recognition_active:
            self.openRecognition.blockSignals(True)
            self.openRecognition.setChecked(False)
            self.openRecognition.blockSignals(False)
            self.config.set_config("recog_open", False)
            recognition_active = False

        self.openRecognition.setDisabled(not recog_available)
        self.recogMethod.setDisabled(not camera_active or camera_busy)
        self.loadMindSporeFace.setDisabled(camera_busy)

    def switch_camera(self):
        if self.camera_ready or self.camera_connecting:
            self.switchButton.setDisabled(True)
            self.visionService.close_camera()
        else:
            started = self.visionService.start_camera()
            if not started:
                self.putLog("摄像头正在运行中")
                return
            self.camera_connecting = True
            self.switchButton.setText("连接中...")
            self.switchButton.setDisabled(True)
            self.putLog("正在连接摄像头...")
        self.update_status_labels()

    def handle_camera_status(self, active, message):
        self.camera_ready = active
        self.camera_connecting = False
        self.config.set_config("camera_on", active)

        if message:
            self.putLog(message)

        if active:
            if not self.timer.isActive():
                self.timer.start(33)
            self.switchButton.setText("关闭摄像头")
        else:
            if self.timer.isActive():
                self.timer.stop()
            self.switchButton.setText("连接摄像头")
            self.current_display_frame = None
            self.current_result_frame = None
            self.displayField.clear()
            self.displayField.setText("实时扫描画面将在这里启动")
            self.displayField.setAlignment(Qt.AlignCenter)
            self.resultFace.clear()
            self.resultFace.setText("等待识别")
            self.resultLabel.setText("等待目标")

        self.switchButton.setDisabled(False)
        self.update_status_labels()

    def update_frame(self):
        config = ConfigStore()
        if self.classicPredictor.trained:
            config.set_config("recog_open_disabled", False)
        self.sync_controls_state()
        if not self.all_queues["display"].empty():
            frame = self.all_queues["display"].get()
            self.displayImage(frame, self.displayField)

    def displayImage(self, img, qlabel):
        frame_copy = img.copy()
        if qlabel is self.displayField:
            self.current_display_frame = frame_copy
        elif qlabel is self.resultFace:
            self.current_result_frame = frame_copy

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qformat = QImage.Format_Indexed8

        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        pixmap = QPixmap.fromImage(outImage)
        scaled = pixmap.scaled(
            qlabel.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        qlabel.setPixmap(scaled)
        qlabel.setAlignment(Qt.AlignCenter)

    def resizeEvent(self, event):
        super(MainPage, self).resizeEvent(event)
        self.layout_chrome()
        if self.current_display_frame is not None:
            self.displayImage(self.current_display_frame, self.displayField)
        if self.current_result_frame is not None:
            self.displayImage(self.current_result_frame, self.resultFace)

    def collect_sample_stats(self):
        identity_count = 0
        file_count = 0

        if not os.path.isdir(self.sample_root):
            return identity_count, file_count

        for entry in os.scandir(self.sample_root):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            visible_files = [
                file_entry for file_entry in os.scandir(entry.path)
                if file_entry.is_file() and not file_entry.name.startswith(".")
            ]
            if visible_files:
                identity_count += 1
                file_count += len(visible_files)

        return identity_count, file_count

    def refresh_sample_overview(self):
        self.sample_identity_count, self.sample_file_count = self.collect_sample_stats()
        current_name = self.nameInput.text().strip()
        current_sample_path = os.path.join(self.sample_root, current_name) if current_name else ""
        current_state = "当前姓名未填写"
        if current_name:
            current_state = "当前姓名已有样本" if os.path.isdir(current_sample_path) else "当前姓名暂无样本"
        self.sampleStatsLabel.setText(
            f"本地样本 {self.sample_identity_count} 人 / {self.sample_file_count} 张\n{current_state}"
        )

    def reset_result_panel(self, label_text="等待目标", face_text="等待识别", hint_text=None):
        self.current_result_frame = None
        self.resultFace.clear()
        self.resultFace.setText(face_text)
        self.resultFace.setAlignment(Qt.AlignCenter)
        self.resultFace.setWordWrap(True)
        self.resultLabel.setText(label_text)
        if hint_text is not None:
            self.resultHintLabel.setText(hint_text)

    def handle_name_input_changed(self):
        self.refresh_sample_overview()
        self.sync_controls_state()

    def delete_current_sample(self):
        name = self.nameInput.text().strip()
        if not name:
            self.putLog("请输入要删除样本的姓名")
            return

        sample_path = os.path.join(self.sample_root, name)
        if not os.path.isdir(sample_path):
            self.putLog(f"未找到姓名 {name} 的本地样本")
            self.refresh_sample_overview()
            self.sync_controls_state()
            return

        try:
            shutil.rmtree(sample_path)
        except OSError as exc:
            self.putLog(f"删除样本失败：{exc}")
            return

        self.classicPredictor.invalidate_model(remove_persisted=True)
        self.putLog(f"已删除姓名 {name} 的本地样本")
        self.putLog("样本已变更，请重新训练经典模型")
        if self.resultLabel.text() == name:
            self.reset_result_panel(
                hint_text="对应样本已删除，请重新采集或重新训练后再识别"
            )
        self.refresh_sample_overview()
        self.update_status_labels()

    def clear_all_samples(self):
        if self.sample_identity_count == 0:
            self.putLog("当前没有可清空的本地样本")
            self.refresh_sample_overview()
            self.sync_controls_state()
            return

        confirmed = QMessageBox.question(
            self,
            "清空全部样本",
            "这会删除 dataset/full 下的全部本地样本，并使经典模型失效。是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirmed != QMessageBox.Yes:
            return

        os.makedirs(self.sample_root, exist_ok=True)
        try:
            for entry in os.scandir(self.sample_root):
                if entry.name.startswith("."):
                    continue
                if entry.is_dir():
                    shutil.rmtree(entry.path)
                elif entry.is_file():
                    os.remove(entry.path)
        except OSError as exc:
            self.putLog(f"清空样本失败：{exc}")
            return

        self.classicPredictor.invalidate_model(remove_persisted=True)
        self.putLog("已清空全部本地样本")
        self.putLog("样本已变更，请重新采集并训练经典模型")
        self.reset_result_panel(hint_text="本地样本已清空，请先采集后再训练或识别")
        self.refresh_sample_overview()
        self.update_status_labels()

    def open_detection(self):
        if self.openDetection.isChecked():
            self.putLog("✅️锁脸辅助已开启")
            self.config.set_config("face_detect", True)
        else:
            self.putLog("🛑锁脸辅助已关闭")
            self.config.set_config("face_detect", False)
            if self.openRecognition.isChecked():
                self.openRecognition.setChecked(False)
                self.putLog("已同步关闭自动识别")
        self.update_status_labels()

    def change_detect_method(self):
        self.putLog(f"切换锁脸方式: {self.detectMethod.currentText()}")
        self.config.set_config("detect_method", self.detectMethod.currentIndex())
        self.update_status_labels()

    def open_recognition(self):
        if self.openRecognition.isChecked():
            if (
                self.recogMethod.currentIndex() == self.config.recog_methods_mapper["classic"]
                and (not self.classicPredictor.trained or getattr(self.classicPredictor, "training", False))
            ):
                self.openRecognition.blockSignals(True)
                self.openRecognition.setChecked(False)
                self.openRecognition.blockSignals(False)
                if getattr(self.classicPredictor, "training", False):
                    self.putLog("经典模型训练中，请等待训练完成后再开启识别")
                else:
                    self.putLog("经典模型尚未训练，请先采集样本并点击“用本地样本训练”")
                self.update_status_labels()
                return
            if not self.openDetection.isChecked():
                self.openDetection.setChecked(True)
                self.putLog("已自动开启锁脸辅助")
            recommended_method = self.config.detect_methods_mapper["mediapipe"]
            if self.detectMethod.currentIndex() != recommended_method:
                self.detectMethod.setCurrentIndex(recommended_method)
                self.putLog("已自动切换到轻量锁脸，保证实时识别更稳定")
            self.putLog("✅️自动识别已开启")
            self.config.set_config("recog_open", True)
        else:
            self.putLog("🛑自动识别已关闭")
            self.config.set_config("recog_open", False)
        self.update_status_labels()

    def change_recog_method(self):
        self.putLog(f"切换人脸识别算法: {self.recogMethod.currentText()}")
        self.config.set_config("recog_method", self.recogMethod.currentIndex())
        if (
            self.recogMethod.currentIndex() == self.config.recog_methods_mapper["classic"]
            and not self.classicPredictor.trained
            and not getattr(self.classicPredictor, "training", False)
        ):
            self.putLog("提示：经典识别需要先采集样本并完成训练")
        self.update_status_labels()

    def putLog(self, log):
        self.logQueue.put(log)

    def receiveLog(self):
        """
        系统日志服务常驻，接收并处理系统日志
        :return:
        """
        while True:
            try:
                data = self.logQueue.get()
            except (EOFError, OSError):
                break
            if data:
                self.receiveLogSignal.emit(data)

    def recogOutput(self, result):
        self.resultLabel.setText(result)

        if result == "未知":
            self.reset_result_panel(
                label_text="未知",
                face_text="未匹配到本地样本",
                hint_text=f"当前人脸与本地样本差异较大，超过阈值 {self.classicPredictor.unknown_threshold:.0f}",
            )
            return

        self.resultHintLabel.setText("已更新目标快照")

        result_dir = os.path.join("dataset", "full", result)
        if not os.path.isdir(result_dir):
            self.reset_result_panel(label_text=result, face_text="未找到目标数据")
            return

        picList = [
            pic for pic in os.listdir(result_dir)
            if not pic.startswith(".")
        ]
        if not picList:
            self.reset_result_panel(label_text=result, face_text="目标目录为空")
            return

        resPic = cv2.imread(os.path.join(result_dir, picList[0]))
        if resPic is None:
            self.reset_result_panel(label_text=result, face_text="图像读取失败")
            return

        self.displayImage(resPic, self.resultFace)

    def logOutput(self, log):
        """
        LOG输出
        :param log:
        :return:
        """
        time = datetime.now().strftime("[%Y/%m/%d %H:%M:%S]")
        log = time + " " + log + "\n"

        self.logText.moveCursor(QTextCursor.End)
        self.logText.insertPlainText(log)
        self.logText.ensureCursorVisible()

    def start_train(self):
        self.refresh_sample_overview()
        self.classicPredictor.train(self.putLog)
        self.update_status_labels()

    def register_face(self):
        """
        异步请求深度识别服务的 /register 接口
        接口参数为 name, photo
        """
        if self.camera_connecting:
            self.putLog("请等待摄像头连接完成")
            return
        if not self.camera_ready:
            self.putLog("请先连接摄像头")
            return
        name = self.nameInput.text()
        if not name:
            self.putLog("请输入姓名")
            return
        self.putLog("开始采集当前人脸")
        self.putLog(f"姓名: {name}")
        self.putLog("请稍后...")
        self.imgRegister.setDisabled(True)
        self.visionService.vision_face_register(name)

    def handleRegister(self, res):
        self.putLog(res)
        if res.startswith("注册成功") or res.startswith("本地采集成功"):
            self.classicPredictor.invalidate_model(remove_persisted=True)
            self.putLog("样本已更新，请重新训练经典模型")
            self.resultHintLabel.setText("当前样本已处理，可继续采集或开始训练")
            self.refresh_sample_overview()
        self.imgRegister.setDisabled(False)
        self.update_status_labels()

    def show_mindspore_status(self):
        service_url = os.getenv("FACE_SERVICE_BASE_URL", "http://127.0.0.1:18000")
        self.putLog(f"当前深度识别服务地址: {service_url}")
        self.putLog("说明：默认地址是本机 SSH 隧道 127.0.0.1:18000，深度识别走远端 embedding 服务")
