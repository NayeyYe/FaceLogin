from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QSpacerItem, QSizePolicy, QLabel, QComboBox


class ControlButtons(QWidget):
    camera_toggled = pyqtSignal(bool)
    detection_toggled = pyqtSignal(bool)
    logout_clicked = pyqtSignal()
    method_changed = pyqtSignal(str)  # 新增检测方法切换信号

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

        # 新增检测方法选择
        method_layout = QVBoxLayout()
        method_label = QLabel("检测方式:")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["OpenCV", "MTCNN"])
        self.method_combo.currentTextChanged.connect(self.method_changed)
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_combo)

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

        # 整合布局
        layout.addLayout(method_layout)
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

    def update_camera_button(self, is_camera_on):
        """更新摄像头按钮状态"""
        self.camera_btn.setText("关闭摄像头" if is_camera_on else "开启摄像头")
        self.camera_btn.setChecked(is_camera_on)
