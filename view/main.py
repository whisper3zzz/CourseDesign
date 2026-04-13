import os
import shutil
import threading
from datetime import datetime
from queue import Queue

import cv2
import requests
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

from service.remote_sync import reconcile_remote_samples
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
        self.startup_remote_sync_thread = None
        self.startup_remote_sync_running = False

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
        QTimer.singleShot(0, self.start_startup_remote_sync)

        self.refresh_recog_button_label()

        if self.classicPredictor.trained:
            self.putLog(
                f"已加载本地经典模型：{len(self.classicPredictor.label_name_map)} 个身份，可直接开启经典识别"
            )

    def setup_chrome(self):
        self.setWindowTitle("人脸识别调度台")
        self.setMaximumSize(16777215, 16777215)
        self.setMinimumSize(1120, 700)
        self.resize(1360, 820)

        self.leftCard = QFrame(self)
        self.leftCard.setObjectName("leftCard")
        self.leftCard.setGeometry(16, 16, 280, 640)

        self.previewCard = QFrame(self)
        self.previewCard.setObjectName("previewCard")
        self.previewCard.setGeometry(312, 16, 706, 640)

        self.rightCard = QFrame(self)
        self.rightCard.setObjectName("rightCard")
        self.rightCard.setGeometry(1034, 16, 310, 640)

        self.samplePanel = QFrame(self)
        self.samplePanel.setObjectName("samplePanel")

        self.actionPanel = QFrame(self)
        self.actionPanel.setObjectName("actionPanel")

        self.logPanel = QFrame(self)
        self.logPanel.setObjectName("logPanel")

        self.controlPanel = QFrame(self)
        self.controlPanel.setObjectName("controlPanel")

        self.resultPanel = QFrame(self)
        self.resultPanel.setObjectName("resultPanel")

        self.previewFooterPanel = QFrame(self)
        self.previewFooterPanel.setObjectName("previewFooterPanel")

        self.accentOrbOne = QFrame(self)
        self.accentOrbOne.setObjectName("accentOrbOne")
        self.accentOrbOne.setGeometry(866, 14, 132, 132)

        self.accentOrbTwo = QFrame(self)
        self.accentOrbTwo.setObjectName("accentOrbTwo")
        self.accentOrbTwo.setGeometry(1206, 500, 112, 112)

        self.subtitleLabel = QLabel(self)
        self.subtitleLabel.setObjectName("subtitleLabel")
        self.subtitleLabel.setGeometry(40, 68, 190, 20)
        self.subtitleLabel.setText("采集 / 训练 / 同步")

        self.logSectionLabel = QLabel(self)
        self.logSectionLabel.setObjectName("logSectionLabel")
        self.logSectionLabel.setGeometry(40, 400, 170, 18)
        self.logSectionLabel.setText("事件流")

        self.previewTitleLabel = QLabel(self)
        self.previewTitleLabel.setObjectName("previewTitleLabel")
        self.previewTitleLabel.setGeometry(344, 40, 260, 30)
        self.previewTitleLabel.setText("实时侦测画面")

        self.previewSubtitleLabel = QLabel(self)
        self.previewSubtitleLabel.setObjectName("previewSubtitleLabel")
        self.previewSubtitleLabel.setGeometry(344, 612, 520, 18)
        self.previewSubtitleLabel.setText("等待视频流、远端检查与识别状态更新")

        self.cameraBadge = QLabel(self)
        self.cameraBadge.setObjectName("cameraBadge")
        self.cameraBadge.setGeometry(822, 38, 164, 30)
        self.cameraBadge.setAlignment(Qt.AlignCenter)

        self.controlSubtitleLabel = QLabel(self)
        self.controlSubtitleLabel.setObjectName("controlSubtitleLabel")
        self.controlSubtitleLabel.setGeometry(1062, 66, 232, 36)
        self.controlSubtitleLabel.setText("锁脸方式、识别方式和远端状态都在这里统一调度")
        self.controlSubtitleLabel.setWordWrap(True)

        self.resultHintLabel = QLabel(self)
        self.resultHintLabel.setObjectName("resultHintLabel")
        self.resultHintLabel.setGeometry(1062, 418, 232, 34)
        self.resultHintLabel.setText("识别命中后，这里显示最近样本")
        self.resultHintLabel.setWordWrap(True)

        self.leftNodeLabel = QLabel(self)
        self.leftNodeLabel.setObjectName("leftNodeLabel")
        self.leftNodeLabel.setGeometry(40, 118, 220, 58)
        self.leftNodeLabel.setText("启动检查\n连接摄像头后进入采集或识别流程")

        self.sampleStatsLabel = QLabel(self)
        self.sampleStatsLabel.setObjectName("sampleStatsLabel")
        self.sampleStatsLabel.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.sampleStatsLabel.setWordWrap(True)

        self.deleteSampleButton = QPushButton(self)
        self.deleteSampleButton.setObjectName("deleteSampleButton")
        self.deleteSampleButton.setText("删除当前样本")

        self.clearSamplesButton = QPushButton(self)
        self.clearSamplesButton.setObjectName("clearSamplesButton")
        self.clearSamplesButton.setText("清空全部样本")

        self.previewMetaLabel = QLabel(self)
        self.previewMetaLabel.setObjectName("previewMetaLabel")
        self.previewMetaLabel.setGeometry(586, 40, 400, 28)
        self.previewMetaLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.previewMetaLabel.setText("画面 640x480  /  主视频流")

        self.previewBeamTop = QFrame(self)
        self.previewBeamTop.setObjectName("previewBeamTop")
        self.previewBeamTop.setGeometry(338, 98, 628, 2)

        self.previewBeamBottom = QFrame(self)
        self.previewBeamBottom.setObjectName("previewBeamBottom")
        self.previewBeamBottom.setGeometry(338, 574, 628, 2)

        self.label.setGeometry(40, 34, 200, 32)
        self.logText.setGeometry(40, 424, 220, 208)
        self.nameInput.setGeometry(40, 242, 220, 44)
        self.sampleStatsLabel.setGeometry(40, 186, 220, 44)
        self.imgRegister.setGeometry(40, 316, 220, 40)
        self.startTrainButton.setGeometry(40, 366, 220, 40)
        self.deleteSampleButton.setGeometry(40, 416, 106, 40)
        self.clearSamplesButton.setGeometry(154, 416, 106, 40)
        self.switchButton.setGeometry(40, 466, 220, 40)

        self.displayField.setGeometry(344, 118, 642, 438)
        self.label_2.setGeometry(1062, 40, 150, 26)
        self.openDetection.setGeometry(1062, 126, 210, 24)
        self.detectMethod.setGeometry(1062, 156, 232, 40)
        self.openRecognition.setGeometry(1062, 214, 210, 24)
        self.recogMethod.setGeometry(1062, 244, 232, 40)
        self.loadMindSporeFace.setGeometry(1062, 308, 232, 40)
        self.label_3.setGeometry(1062, 390, 150, 24)
        self.resultFace.setGeometry(1114, 468, 128, 128)
        self.resultLabel.setGeometry(1062, 614, 232, 30)

        self.label.setText("样本中枢")
        self.label_2.setText("识别调度")
        self.label_3.setText("目标画像")
        self.switchButton.setText("连接摄像头")
        self.startTrainButton.setText("用本地样本训练")
        self.imgRegister.setText("采集当前人脸")
        self.loadMindSporeFace.setText("检查远端深度服务")
        self.nameInput.setPlaceholderText("输入姓名，准备采集或管理样本")
        self.displayField.setText("实时扫描画面将在这里启动")
        self.displayField.setAlignment(Qt.AlignCenter)
        self.displayField.setWordWrap(True)
        self.resultFace.setText("待命中")
        self.resultFace.setAlignment(Qt.AlignCenter)
        self.resultFace.setWordWrap(True)
        self.resultLabel.setText("等待目标")
        self.resultLabel.setAlignment(Qt.AlignCenter)
        self.resultLabel.setWordWrap(True)
        self.leftNodeLabel.setWordWrap(True)
        self.leftNodeLabel.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.logText.setReadOnly(True)
        self.logText.setAcceptRichText(False)
        self.previewSubtitleLabel.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.recogMethod.setItemText(0, "经典识别")
        self.recogMethod.setItemText(1, "深度识别")
        self.recogMethod.setItemText(2, "CNN分类识别")
        self.openDetection.setText("显示锁脸框")
        self.openRecognition.setText("开始自动识别")
        self.detectMethod.setItemText(0, "经典锁脸")
        self.detectMethod.setItemText(1, "轻量锁脸")
        self.detectMethod.setItemText(2, "深度锁脸")

        self.setStyleSheet(
            """
            QMainWindow#System {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #050c11,
                    stop: 0.42 #0b141a,
                    stop: 1 #10181f
                );
            }
            QFrame#leftCard {
                background-color: #101820;
                border: 1px solid #273845;
                border-radius: 32px;
            }
            QFrame#previewCard {
                background-color: #0d161d;
                border: 1px solid #304451;
                border-radius: 36px;
            }
            QFrame#rightCard {
                background-color: #111a22;
                border: 1px solid #2b404d;
                border-radius: 32px;
            }
            QFrame#samplePanel,
            QFrame#actionPanel,
            QFrame#logPanel,
            QFrame#controlPanel,
            QFrame#resultPanel,
            QFrame#previewFooterPanel {
                background-color: rgba(10, 18, 24, 220);
                border: 1px solid rgba(79, 110, 124, 120);
                border-radius: 22px;
            }
            QFrame#accentOrbOne {
                background-color: rgba(72, 210, 189, 34);
                border-radius: 66px;
            }
            QFrame#accentOrbTwo {
                background-color: rgba(255, 181, 92, 28);
                border-radius: 56px;
            }
            QFrame#previewBeamTop,
            QFrame#previewBeamBottom {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 rgba(84, 214, 197, 0),
                    stop: 0.18 rgba(84, 214, 197, 140),
                    stop: 0.82 rgba(255, 184, 87, 140),
                    stop: 1 rgba(255, 184, 87, 0)
                );
                border-radius: 1px;
            }
            QLabel#label {
                color: #f5f7f8;
                font-size: 30px;
                font-weight: 700;
                font-family: "Avenir Next Condensed", "Avenir Next", "PingFang SC";
                letter-spacing: 1px;
            }
            QLabel#subtitleLabel {
                color: #83a8b2;
                font-size: 12px;
                font-weight: 600;
                font-family: "Menlo";
                letter-spacing: 1px;
            }
            QLabel#logSectionLabel {
                color: #cfdee4;
                font-size: 13px;
                font-weight: 600;
                font-family: "Menlo";
                letter-spacing: 1px;
            }
            QLabel#previewTitleLabel {
                color: #f6f7f4;
                font-size: 28px;
                font-weight: 700;
                font-family: "Avenir Next Condensed", "Avenir Next", "PingFang SC";
                letter-spacing: 1px;
            }
            QLabel#previewSubtitleLabel {
                color: #d2dde1;
                font-size: 12px;
                font-weight: 500;
                font-family: "Menlo";
                padding: 0 18px;
            }
            QLabel#cameraBadge {
                background-color: rgba(16, 30, 38, 230);
                border: 1px solid rgba(88, 126, 137, 150);
                border-radius: 15px;
                font-size: 11px;
                font-weight: 700;
                font-family: "Menlo";
                padding: 0 12px;
                letter-spacing: 1px;
            }
            QLabel#label_2,
            QLabel#label_3 {
                color: #eef4f6;
                font-size: 18px;
                font-weight: 700;
                font-family: "Avenir Next Condensed", "Avenir Next", "PingFang SC";
                letter-spacing: 1px;
            }
            QLabel#controlSubtitleLabel,
            QLabel#resultHintLabel {
                color: #94adb6;
                font-size: 12px;
                font-weight: 500;
                font-family: "Menlo";
            }
            QLabel#previewMetaLabel {
                color: #90aeb8;
                font-size: 11px;
                font-weight: 600;
                font-family: "Menlo";
                letter-spacing: 1px;
            }
            QLabel#leftNodeLabel {
                color: #f4f7f8;
                background-color: rgba(17, 28, 36, 220);
                border: 1px solid rgba(73, 114, 131, 145);
                border-radius: 18px;
                font-size: 12px;
                font-weight: 600;
                font-family: "Avenir Next", "PingFang SC";
                padding: 8px 10px;
            }
            QLabel#sampleStatsLabel {
                color: #8dc6d0;
                background-color: rgba(10, 19, 26, 212);
                border: 1px solid rgba(57, 85, 98, 145);
                border-radius: 16px;
                font-size: 11px;
                font-weight: 600;
                font-family: "Menlo";
                padding: 6px 10px;
            }
            QLabel#displayField {
                background-color: #081117;
                border: 1px solid #38525f;
                border-radius: 28px;
                color: #88a9b4;
                font-size: 18px;
                font-weight: 600;
                font-family: "Avenir Next", "PingFang SC";
                padding: 20px;
            }
            QLabel#resultFace {
                background-color: #081117;
                border: 1px solid #425864;
                border-radius: 30px;
                color: #d4e0e4;
                font-size: 15px;
                font-weight: 600;
                font-family: "Menlo";
                padding: 12px;
            }
            QLabel#resultLabel {
                color: #f4f7f8;
                font-size: 22px;
                font-weight: 700;
                font-family: "Avenir Next Condensed", "Avenir Next", "PingFang SC";
            }
            QTextEdit#logText {
                background-color: rgba(8, 15, 20, 238);
                border: 1px solid rgba(55, 79, 93, 140);
                border-radius: 18px;
                color: #c4dde4;
                padding: 12px;
                selection-background-color: #4e8f9a;
                font-size: 12px;
                font-family: "Menlo";
            }
            QTextEdit#logText QScrollBar:vertical {
                width: 10px;
                margin: 6px 2px 6px 0;
                background: transparent;
            }
            QTextEdit#logText QScrollBar::handle:vertical {
                background: rgba(132, 177, 187, 110);
                border-radius: 5px;
                min-height: 28px;
            }
            QTextEdit#logText QScrollBar::add-line:vertical,
            QTextEdit#logText QScrollBar::sub-line:vertical {
                height: 0;
            }
            QLineEdit#nameInput {
                background-color: rgba(8, 17, 23, 235);
                border: 1px solid rgba(71, 102, 116, 165);
                border-radius: 18px;
                color: #f4f8f9;
                padding: 0 14px;
                font-size: 13px;
                font-weight: 500;
                font-family: "Menlo";
            }
            QLineEdit#nameInput:focus {
                border: 1px solid rgba(255, 190, 104, 220);
            }
            QPushButton {
                border: none;
                border-radius: 18px;
                font-size: 13px;
                font-weight: 700;
                font-family: "Menlo";
                letter-spacing: 0.5px;
                padding: 0 14px;
            }
            QPushButton#switchButton {
                background-color: #16897f;
                color: #f4fffc;
            }
            QPushButton#switchButton:hover {
                background-color: #1ca093;
            }
            QPushButton#startTrainButton {
                background-color: #355a89;
                color: #edf5ff;
            }
            QPushButton#startTrainButton:hover {
                background-color: #416ca3;
            }
            QPushButton#imgRegister {
                background-color: #d29647;
                color: #1d1305;
            }
            QPushButton#imgRegister:hover {
                background-color: #ebaa54;
            }
            QPushButton#deleteSampleButton {
                background-color: #8a4c36;
                color: #fff3ec;
                font-size: 12px;
            }
            QPushButton#deleteSampleButton:hover {
                background-color: #a45d43;
            }
            QPushButton#clearSamplesButton {
                background-color: #7a3d4f;
                color: #ffe6ec;
                font-size: 12px;
            }
            QPushButton#clearSamplesButton:hover {
                background-color: #92495e;
            }
            QPushButton#loadMindSporeFace {
                background-color: rgba(12, 22, 30, 235);
                border: 1px solid rgba(77, 112, 127, 160);
                color: #dce9ee;
            }
            QPushButton#loadMindSporeFace:hover {
                background-color: rgba(20, 33, 42, 245);
            }
            QPushButton:disabled {
                background-color: #23343d;
                color: #6d8792;
            }
            QCheckBox {
                color: #e0ebef;
                font-size: 13px;
                font-weight: 600;
                font-family: "Avenir Next", "PingFang SC";
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid #45606d;
                background-color: #091218;
            }
            QCheckBox::indicator:checked {
                background-color: #eab15d;
                border: 4px solid #091218;
            }
            QCheckBox::indicator:disabled {
                background-color: #16232c;
                border: 1px solid #31424d;
            }
            QComboBox {
                background-color: #09151b;
                border: 1px solid #425b69;
                border-radius: 18px;
                color: #edf2f3;
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
                background-color: #101920;
                border: 1px solid #425b69;
                color: #edf2f3;
                selection-background-color: #d99d4d;
                selection-color: #1d1407;
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
            self.samplePanel,
            self.actionPanel,
            self.logPanel,
            self.controlPanel,
            self.resultPanel,
            self.previewFooterPanel,
        ):
            widget.raise_()

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

    def wrapped_label_height(self, label, width, horizontal_padding, vertical_padding, minimum_height):
        content_width = max(32, width - horizontal_padding * 2)
        text_rect = label.fontMetrics().boundingRect(
            0,
            0,
            content_width,
            2000,
            Qt.TextWordWrap,
            label.text(),
        )
        return max(minimum_height, text_rect.height() + vertical_padding * 2)

    def layout_chrome(self):
        width = max(self.width(), self.minimumWidth())
        height = max(self.height(), self.minimumHeight())

        margin = self.clamp(int(min(width, height) * 0.022), 16, 28)
        gap = self.clamp(int(width * 0.015), 14, 24)
        card_height = height - margin * 2

        left_width = self.clamp(int(width * 0.24), 250, 330)
        right_width = self.clamp(int(width * 0.26), 280, 350)
        preview_width = width - margin * 2 - gap * 2 - left_width - right_width

        if preview_width < 500:
            shortage = 500 - preview_width
            shrink_right = min(shortage // 2, right_width - 252)
            right_width -= max(0, shrink_right)
            shortage -= max(0, shrink_right)
            shrink_left = min(shortage, left_width - 228)
            left_width -= max(0, shrink_left)
            preview_width = width - margin * 2 - gap * 2 - left_width - right_width

        left_x = margin
        preview_x = left_x + left_width + gap
        right_x = preview_x + preview_width + gap
        top_y = margin

        self.leftCard.setGeometry(left_x, top_y, left_width, card_height)
        self.previewCard.setGeometry(preview_x, top_y, preview_width, card_height)
        self.rightCard.setGeometry(right_x, top_y, right_width, card_height)

        orb_one_size = self.clamp(int(preview_width * 0.16), 108, 144)
        self.accentOrbOne.setGeometry(
            preview_x + preview_width - orb_one_size + 24,
            top_y - 12,
            orb_one_size,
            orb_one_size,
        )

        orb_two_size = self.clamp(int(right_width * 0.34), 96, 124)
        self.accentOrbTwo.setGeometry(
            right_x + right_width - orb_two_size + 18,
            top_y + card_height - orb_two_size + 14,
            orb_two_size,
            orb_two_size,
        )

        left_pad = self.clamp(int(left_width * 0.085), 20, 28)
        left_inner_width = left_width - left_pad * 2
        button_height = self.clamp(int(card_height * 0.055), 36, 40)
        field_height = self.clamp(int(card_height * 0.056), 38, 42)
        section_gap = self.clamp(int(card_height * 0.02), 12, 18)
        panel_pad = 16
        sample_panel_pad = 12

        self.label.setGeometry(left_x + left_pad, top_y + 24, left_inner_width, 32)
        self.subtitleLabel.setGeometry(left_x + left_pad, top_y + 58, left_inner_width, 20)

        panel_stack_top = top_y + 104
        panel_stack_bottom = top_y + card_height - left_pad
        panel_stack_height = panel_stack_bottom - panel_stack_top
        sample_info_width = left_inner_width - sample_panel_pad * 2
        hint_height = self.wrapped_label_height(
            self.leftNodeLabel,
            sample_info_width,
            horizontal_padding=20,
            vertical_padding=8,
            minimum_height=44,
        )
        stats_height = self.wrapped_label_height(
            self.sampleStatsLabel,
            sample_info_width,
            horizontal_padding=20,
            vertical_padding=6,
            minimum_height=32,
        )
        sample_panel_needed = sample_panel_pad * 2 + hint_height + 8 + stats_height + 8 + field_height
        sample_panel_height = self.clamp(max(sample_panel_needed, int(panel_stack_height * 0.33)), 168, 208)
        action_panel_height = self.clamp(int(panel_stack_height * 0.35), 192, 228)
        log_panel_height = panel_stack_height - sample_panel_height - action_panel_height - section_gap * 2
        if log_panel_height < 144:
            shortage = 144 - log_panel_height
            reduce_action = min(shortage, action_panel_height - 192)
            action_panel_height -= reduce_action
            shortage -= reduce_action
            reduce_sample = min(shortage, sample_panel_height - sample_panel_needed)
            sample_panel_height -= reduce_sample
            log_panel_height = panel_stack_height - sample_panel_height - action_panel_height - section_gap * 2

        sample_y = panel_stack_top
        action_y = sample_y + sample_panel_height + section_gap
        log_y = action_y + action_panel_height + section_gap

        panel_x = left_x + left_pad
        self.samplePanel.setGeometry(panel_x, sample_y, left_inner_width, sample_panel_height)
        self.actionPanel.setGeometry(panel_x, action_y, left_inner_width, action_panel_height)
        self.logPanel.setGeometry(panel_x, log_y, left_inner_width, log_panel_height)

        self.leftNodeLabel.setGeometry(
            panel_x + sample_panel_pad,
            sample_y + sample_panel_pad,
            sample_info_width,
            hint_height,
        )
        self.sampleStatsLabel.setGeometry(
            panel_x + sample_panel_pad,
            sample_y + sample_panel_pad + hint_height + 8,
            sample_info_width,
            stats_height,
        )
        self.nameInput.setGeometry(
            panel_x + sample_panel_pad,
            sample_y + sample_panel_height - sample_panel_pad - field_height,
            sample_info_width,
            field_height,
        )

        register_y = action_y + panel_pad
        train_y = register_y + button_height + 10
        manage_y = train_y + button_height + 10
        switch_y = action_y + action_panel_height - panel_pad - button_height
        self.imgRegister.setGeometry(panel_x + panel_pad, register_y, left_inner_width - panel_pad * 2, button_height)
        self.startTrainButton.setGeometry(panel_x + panel_pad, train_y, left_inner_width - panel_pad * 2, button_height)
        manage_gap = self.clamp(int(left_inner_width * 0.06), 10, 14)
        manage_width = (left_inner_width - panel_pad * 2 - manage_gap) // 2
        self.deleteSampleButton.setGeometry(panel_x + panel_pad, manage_y, manage_width, button_height)
        self.clearSamplesButton.setGeometry(
            panel_x + panel_pad + manage_width + manage_gap,
            manage_y,
            left_inner_width - panel_pad * 2 - manage_width - manage_gap,
            button_height,
        )
        self.switchButton.setGeometry(panel_x + panel_pad, switch_y, left_inner_width - panel_pad * 2, button_height)

        self.logSectionLabel.setGeometry(panel_x + panel_pad, log_y + panel_pad, left_inner_width - panel_pad * 2, 18)
        self.logText.setGeometry(
            panel_x + panel_pad,
            log_y + panel_pad + 26,
            left_inner_width - panel_pad * 2,
            log_panel_height - panel_pad * 2 - 26,
        )

        preview_pad = self.clamp(int(preview_width * 0.04), 24, 36)
        preview_inner_width = preview_width - preview_pad * 2
        header_top = top_y + 24
        title_width = self.clamp(int(preview_inner_width * 0.38), 224, 296)
        badge_width = self.clamp(int(preview_width * 0.23), 156, 196)
        badge_height = 32
        badge_x = preview_x + preview_width - preview_pad - badge_width

        self.previewTitleLabel.setGeometry(preview_x + preview_pad, header_top, title_width, 28)
        self.cameraBadge.setGeometry(badge_x, header_top - 2, badge_width, badge_height)

        meta_x = preview_x + preview_pad + title_width + 18
        meta_width = max(0, badge_x - meta_x - 20)
        self.previewMetaLabel.setVisible(meta_width > 140)
        if meta_width > 140:
            self.previewMetaLabel.setGeometry(meta_x, header_top, meta_width, 28)

        display_x = preview_x + preview_pad
        display_y = top_y + 102
        footer_height = self.clamp(int(card_height * 0.085), 52, 62)
        footer_y = top_y + card_height - preview_pad - footer_height
        display_height = max(280, footer_y - display_y - 20)
        self.displayField.setGeometry(display_x, display_y, preview_inner_width, display_height)

        self.previewFooterPanel.setGeometry(display_x, footer_y, preview_inner_width, footer_height)
        self.previewSubtitleLabel.setGeometry(
            display_x + 18,
            footer_y + (footer_height - 18) // 2,
            preview_inner_width - 36,
            18,
        )

        beam_x = display_x + 18
        beam_width = max(60, preview_inner_width - 36)
        self.previewBeamTop.setGeometry(beam_x, display_y - 12, beam_width, 2)
        self.previewBeamBottom.setGeometry(beam_x, display_y + display_height + 12, beam_width, 2)
        self.previewMetaLabel.setText(f"画面 {preview_inner_width}×{display_height}  /  实时视频流")

        right_pad = self.clamp(int(right_width * 0.09), 22, 30)
        right_inner_width = right_width - right_pad * 2
        combo_height = 40

        self.label_2.setGeometry(right_x + right_pad, top_y + 24, right_inner_width, 24)
        self.controlSubtitleLabel.setGeometry(right_x + right_pad, top_y + 56, right_inner_width, 34)

        control_panel_y = top_y + 104
        control_panel_height = self.clamp(int(card_height * 0.34), 214, 248)
        self.controlPanel.setGeometry(right_x + right_pad, control_panel_y, right_inner_width, control_panel_height)
        self.openDetection.setGeometry(right_x + right_pad + 18, control_panel_y + 18, right_inner_width - 36, 22)
        self.detectMethod.setGeometry(right_x + right_pad + 18, control_panel_y + 48, right_inner_width - 36, combo_height)
        self.openRecognition.setGeometry(right_x + right_pad + 18, control_panel_y + 108, right_inner_width - 36, 22)
        self.recogMethod.setGeometry(right_x + right_pad + 18, control_panel_y + 138, right_inner_width - 36, combo_height)
        self.loadMindSporeFace.setGeometry(
            right_x + right_pad + 18,
            control_panel_y + control_panel_height - 18 - combo_height,
            right_inner_width - 36,
            combo_height,
        )

        result_panel_y = control_panel_y + control_panel_height + section_gap
        result_panel_height = top_y + card_height - right_pad - result_panel_y
        self.resultPanel.setGeometry(right_x + right_pad, result_panel_y, right_inner_width, result_panel_height)
        self.label_3.setGeometry(right_x + right_pad + 18, result_panel_y + 18, right_inner_width - 36, 24)
        self.resultHintLabel.setGeometry(right_x + right_pad + 18, result_panel_y + 48, right_inner_width - 36, 34)

        result_label_height = 42
        result_content_bottom = result_panel_y + result_panel_height - 18
        result_face_max_height = result_content_bottom - (result_panel_y + 96) - 14 - result_label_height
        result_face_size = self.clamp(int(min(right_inner_width - 52, result_face_max_height)), 142, 198)
        result_face_x = right_x + right_pad + (right_inner_width - result_face_size) // 2
        result_face_y = result_panel_y + 96
        self.resultFace.setGeometry(result_face_x, result_face_y, result_face_size, result_face_size)
        self.resultLabel.setGeometry(
            right_x + right_pad + 18,
            result_face_y + result_face_size + 14,
            right_inner_width - 36,
            result_label_height,
        )

    def apply_shadow(self, widget, blur_radius, offset_y):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(blur_radius)
        shadow.setOffset(0, offset_y)
        shadow.setColor(QColor(15, 204, 255, 34))
        widget.setGraphicsEffect(shadow)

    def env_flag(self, name, default):
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def start_startup_remote_sync(self):
        if not self.env_flag("FACE_SERVICE_AUTO_SYNC_ON_START", True):
            self.putLog("启动检查：已关闭远端样本自动同步")
            return
        if self.startup_remote_sync_running:
            return

        self.startup_remote_sync_running = True
        self.putLog("启动检查：正在检查远端服务并同步样本...")
        self.startup_remote_sync_thread = threading.Thread(
            target=self.run_startup_remote_sync,
            daemon=True,
        )
        self.startup_remote_sync_thread.start()

    def run_startup_remote_sync(self):
        try:
            summary = reconcile_remote_samples(
                dataset_root=self.sample_root,
                base_url=os.getenv("FACE_SERVICE_BASE_URL", "http://127.0.0.1:18000"),
                manifest_path=os.path.join("temp", "remote_sync_manifest.json"),
                timeout=float(os.getenv("FACE_SERVICE_AUTO_SYNC_TIMEOUT", "8")),
                keep_remote_missing=self.env_flag("FACE_SERVICE_AUTO_SYNC_KEEP_REMOTE_MISSING", False),
                progress=self.putLog,
            )
            if summary.ok:
                if summary.changed:
                    self.putLog(f"启动检查完成：远端样本已同步，{summary.summary_line()}")
                else:
                    self.putLog(f"启动检查完成：远端样本已是最新，{summary.summary_line()}")
            else:
                self.putLog(f"启动检查失败：{summary.error}")
        except Exception as exc:
            self.putLog(f"启动检查失败：{exc}")
        finally:
            self.startup_remote_sync_running = False

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
            self.leftNodeLabel.setText("步骤 1 / 先连接摄像头\n随后可开始识别或采集")
        elif not self.openDetection.isChecked():
            self.leftNodeLabel.setText("步骤 2 / 可开始识别\n系统会自动锁定人脸")
        elif not self.openRecognition.isChecked():
            if classic_ready:
                self.leftNodeLabel.setText("步骤 3 / 经典模型已就绪\n现在可直接识别或继续采集")
            else:
                self.leftNodeLabel.setText("步骤 3 / 可采集当前人脸\n也可直接开启自动识别")
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
            elif self.recogMethod.currentIndex() == self.config.recog_methods_mapper["cnn_classifier"]:
                self.resultHintLabel.setText("CNN分类识别需要先训练/更新分类器")
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

        self.layout_chrome()
        self.sync_controls_state()

    def refresh_recog_button_label(self):
        if self.recogMethod.currentIndex() == self.config.recog_methods_mapper["cnn_classifier"]:
            self.loadMindSporeFace.setText("训练/更新分类器")
        else:
            self.loadMindSporeFace.setText("检查远端深度服务")

    def sync_controls_state(self):
        camera_active = self.camera_ready
        camera_busy = self.camera_connecting
        detection_active = self.openDetection.isChecked()
        recognition_active = self.openRecognition.isChecked()
        training_active = getattr(self.classicPredictor, "training", False)
        classic_ready = self.classicPredictor.trained and not getattr(self.classicPredictor, "training", False)
        classic_selected = self.recogMethod.currentIndex() == self.config.recog_methods_mapper["classic"]
        classifier_selected = self.recogMethod.currentIndex() == self.config.recog_methods_mapper["cnn_classifier"]
        has_name = bool(self.nameInput.text().strip())
        has_samples = self.sample_identity_count > 0

        self.switchButton.setDisabled(camera_busy)
        if classifier_selected:
            self.startTrainButton.setDisabled(not has_samples)
            self.startTrainButton.setText("训练/更新分类器")
        else:
            self.startTrainButton.setDisabled(training_active or not has_samples)
            if training_active:
                self.startTrainButton.setText("训练中...")
            elif classic_ready:
                self.startTrainButton.setText("重新训练本地样本")
            else:
                self.startTrainButton.setText("用本地样本训练")
        disable_input = camera_busy or (training_active and not classifier_selected)
        self.nameInput.setDisabled(disable_input)
        self.imgRegister.setDisabled(not camera_active or disable_input)
        self.deleteSampleButton.setDisabled((training_active and not classifier_selected) or not has_name)
        self.clearSamplesButton.setDisabled((training_active and not classifier_selected) or not has_samples)
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
        elif self.recogMethod.currentIndex() == self.config.recog_methods_mapper["cnn_classifier"]:
            self.putLog("提示：CNN分类识别需先训练/更新分类器")
        self.update_status_labels()
        self.refresh_recog_button_label()

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

        latest_pic = max(
            picList,
            key=lambda pic: os.path.getmtime(os.path.join(result_dir, pic)),
        )
        resPic = cv2.imread(os.path.join(result_dir, latest_pic))
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
        if self.recogMethod.currentIndex() == self.config.recog_methods_mapper["cnn_classifier"]:
            if self.sample_identity_count == 0:
                self.putLog("样本不足，无法训练CNN分类器")
                return
            self.putLog("开始训练/更新CNN分类器，请稍候...")
            service_url = os.getenv("FACE_SERVICE_BASE_URL", "http://127.0.0.1:18000").rstrip("/")
            try:
                response = requests.post(
                    f"{service_url}/train_classifier",
                    timeout=30,
                )
                if response.status_code == 200:
                    payload = response.json()
                    message = payload.get("message") or payload.get("status") or "CNN分类器训练完成"
                    self.putLog(f"{message}")
                else:
                    self.putLog(f"CNN分类器训练失败：状态码 {response.status_code}")
            except requests.RequestException as exc:
                self.putLog(f"CNN分类器训练请求失败：{exc}")
            self.update_status_labels()
            return
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
        name = self.nameInput.text().strip()
        if not name:
            self.putLog("请输入姓名")
            return
        if name != self.nameInput.text():
            self.nameInput.setText(name)
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
        if self.recogMethod.currentIndex() == self.config.recog_methods_mapper["cnn_classifier"]:
            self.start_train()
            return
        service_url = os.getenv("FACE_SERVICE_BASE_URL", "http://127.0.0.1:18000")
        self.putLog(f"当前深度识别服务地址: {service_url}")
        self.putLog("说明：默认地址通常指向本机转发端口，深度识别由远端服务处理")
