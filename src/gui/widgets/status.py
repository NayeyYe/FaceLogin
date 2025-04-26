from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from PyQt5.QtGui import QColor, QPainter, QFont
from PyQt5.QtCore import Qt


class StatusIndicator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(20, 20)
        self._color = QColor(255, 0, 0)

    def set_color(self, color):
        self._color = color
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(self._color)
        painter.drawEllipse(0, 0, 20, 20)


class StatusBar(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setSpacing(15)

        # 状态指示灯区域
        status_grid = QHBoxLayout()

        # 摄像头状态
        cam_layout = QHBoxLayout()
        cam_label = QLabel("摄像头:")
        self.cam_indicator = StatusIndicator()
        cam_layout.addWidget(cam_label)
        cam_layout.addWidget(self.cam_indicator)

        # 检测状态
        det_layout = QHBoxLayout()
        det_label = QLabel("检测状态:")
        self.det_indicator = StatusIndicator()
        det_layout.addWidget(det_label)
        det_layout.addWidget(self.det_indicator)

        status_grid.addLayout(cam_layout)
        status_grid.addLayout(det_layout)
        layout.addLayout(status_grid)

        # 人脸计数
        self.face_count = QLabel("检测到人脸: 0")
        self.face_count.setFont(QFont("Arial", 12))
        layout.addWidget(self.face_count)

        # 系统消息
        self.message_box = QLabel("系统就绪")
        self.message_box.setWordWrap(True)
        self.message_box.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(self.message_box)

        self.setLayout(layout)

    def update_camera_status(self, is_on):
        color = QColor(0, 255, 0) if is_on else QColor(255, 0, 0)
        self.cam_indicator.set_color(color)

    def update_detection_status(self, is_detecting):
        color = QColor(0, 255, 0) if is_detecting else QColor(255, 0, 0)
        self.det_indicator.set_color(color)

    def update_face_count(self, count):
        self.face_count.setText(f"检测到人脸: {count}")

    def show_message(self, msg, is_error=False):
        color = "#ff4444" if is_error else "#44ff44"
        self.message_box.setText(f'<font color="{color}">▶ {msg}</font>')
