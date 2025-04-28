from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton,
                             QSpacerItem, QSizePolicy, QLabel, QComboBox)


class ControlButtons(QWidget):
    camera_toggled = pyqtSignal(bool)
    detection_toggled = pyqtSignal(bool)
    logout_clicked = pyqtSignal()
    method_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["OpenCV", "MTCNN"])

        self.camera_btn = QPushButton("开启摄像头")
        self.detect_btn = QPushButton("开启检测")
        self.logout_btn = QPushButton("退出登录")

        self._configure_buttons()
        self._setup_layout(layout)

    def _configure_buttons(self):
        self.camera_btn.setCheckable(True)
        self.detect_btn.setCheckable(True)
        self.detect_btn.setEnabled(False)
        self.logout_btn.setEnabled(False)

        button_style = "QPushButton { padding: 10px; font-size: 14px; }"
        self.camera_btn.setStyleSheet(
            f"{button_style} QPushButton:checked {{ background-color: #ff4444; }}")
        self.detect_btn.setStyleSheet(
            f"{button_style} QPushButton:checked {{ background-color: #44ff44; }}")
        self.logout_btn.setStyleSheet(f"{button_style} background-color: #666;")

    def _setup_layout(self, layout):
        method_layout = QVBoxLayout()
        method_layout.addWidget(QLabel("检测方式:"))
        method_layout.addWidget(self.method_combo)

        layout.addLayout(method_layout)
        layout.addWidget(self.camera_btn)
        layout.addWidget(self.detect_btn)
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        layout.addWidget(self.logout_btn)
        self.setLayout(layout)

    def _connect_signals(self):
        self.camera_btn.clicked.connect(self._toggle_camera)
        self.detect_btn.clicked.connect(self._toggle_detection)
        self.logout_btn.clicked.connect(self.logout_clicked)
        self.method_combo.currentTextChanged.connect(self.method_changed)

    def _toggle_camera(self, checked):
        self.camera_toggled.emit(checked)
        self.detect_btn.setEnabled(checked)
        self.detect_btn.setChecked(False)
        self.detect_btn.setText("开启检测")
        self.camera_btn.setText("关闭摄像头" if checked else "开启摄像头")

    def _toggle_detection(self, checked):
        self.detection_toggled.emit(checked)
        self.detect_btn.setText("关闭检测" if checked else "开启检测")

    def update_login_state(self, is_logged_in):
        self.logout_btn.setEnabled(is_logged_in)
        self.logout_btn.setStyleSheet(
            f"background-color: {'#ff4444' if is_logged_in else '#666'};"
            "padding: 10px; font-size: 14px;")

    def update_camera_button(self, is_camera_on):
        self.camera_btn.setChecked(is_camera_on)
        self.camera_btn.setText("关闭摄像头" if is_camera_on else "开启摄像头")
