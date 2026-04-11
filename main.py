"""
课程设计主模块
:author: 李正楠
:date: 2024-4-14
"""
import sys

import cv2
from PyQt5.QtWidgets import QApplication

from service.ssh_tunnel import bootstrap_face_service_tunnel
from service.vision import VisionService, ClassicFaceRecognizer
from store.config import ConfigStore
from view.main import MainPage

if __name__ == "__main__":
    tunnel_bootstrap = bootstrap_face_service_tunnel()
    app = QApplication(sys.argv)
    store = ConfigStore()
    classicPreditor = ClassicFaceRecognizer()
    camera = VisionService(store)
    window = MainPage(camera.all_queues, classicPreditor, camera, store)
    if tunnel_bootstrap.manager is not None:
        app.aboutToQuit.connect(tunnel_bootstrap.manager.stop)
    if tunnel_bootstrap.message:
        window.putLog(tunnel_bootstrap.message)
    window.show()
    sys.exit(app.exec())
