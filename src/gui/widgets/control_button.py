from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton


class ControlButtons(QWidget):
    camera_toggled = pyqtSignal(bool)
    logout_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.camera_btn = QPushButton("开启摄像头")
        self.logout_btn = QPushButton("退出登录")
        self.logout_btn.setEnabled(False)

        self.camera_btn.clicked.connect(
            lambda: self.camera_toggled.emit(not self.camera_btn.isChecked())
        )
        self.logout_btn.clicked.connect(self.logout_clicked)

        layout.addWidget(self.camera_btn)
        layout.addWidget(self.logout_btn)
        self.setLayout(layout)

    def update_login_state(self, is_logged_in):
        self.logout_btn.setEnabled(is_logged_in)
        self.camera_btn.setEnabled(not is_logged_in)

    def update_camera_button(self, is_camera_on):
        """更新摄像头按钮文字"""
        self.camera_btn.setText("关闭摄像头" if is_camera_on else "开启摄像头")
        self.camera_btn.setChecked(is_camera_on)