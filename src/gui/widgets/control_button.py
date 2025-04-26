from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QSpacerItem, QSizePolicy

class ControlButtons(QWidget):
    camera_toggled = pyqtSignal(bool)
    detection_toggled = pyqtSignal(bool)
    logout_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setSpacing(15)

        # 摄像头控制按钮
        self.camera_btn = QPushButton("开启摄像头")
        self.camera_btn.setCheckable(True)
        self.camera_btn.setStyleSheet(
            "QPushButton { padding: 10px; font-size: 14px; }"
            "QPushButton:checked { background-color: #ff4444; }"
        )

        # 检测控制按钮
        self.detect_btn = QPushButton("开启检测")
        self.detect_btn.setCheckable(True)
        self.detect_btn.setEnabled(False)
        self.detect_btn.setStyleSheet(
            "QPushButton { padding: 10px; font-size: 14px; }"
            "QPushButton:checked { background-color: #44ff44; }"
        )

        # 退出登录按钮
        self.logout_btn = QPushButton("退出登录")
        self.logout_btn.setEnabled(False)
        self.logout_btn.setStyleSheet(
            "QPushButton { padding: 10px; font-size: 14px; background-color: #666; }"
        )

        # 信号连接
        self.camera_btn.clicked.connect(self._toggle_camera)
        self.detect_btn.clicked.connect(self._toggle_detection)
        self.logout_btn.clicked.connect(self.logout_clicked)

        # 布局
        layout.addWidget(self.camera_btn)
        layout.addWidget(self.detect_btn)
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        layout.addWidget(self.logout_btn)
        self.setLayout(layout)

    def _toggle_camera(self, checked):
        self.camera_toggled.emit(checked)
        self.detect_btn.setEnabled(checked)
        self.detect_btn.setChecked(False)
        self.detect_btn.setText("开启检测")

    def _toggle_detection(self, checked):
        self.detection_toggled.emit(checked)
        self.detect_btn.setText("关闭检测" if checked else "开启检测")

    def update_login_state(self, is_logged_in):
        self.logout_btn.setEnabled(is_logged_in)
        self.logout_btn.setStyleSheet(
            f"background-color: {'#ff4444' if is_logged_in else '#666'};"
            "padding: 10px; font-size: 14px;"
        )
