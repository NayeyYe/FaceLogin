from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtCore import Qt, QTimer


class StatusIndicator(QWidget):
    def __init__(self):
        super().__init__()
        self._color = QColor(255, 0, 0)
        self.setFixedSize(20, 20)

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

        # 在原有布局前添加FPS显示
        fps_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: --")
        fps_layout.addWidget(self.fps_label)
        layout.addLayout(fps_layout)

        # 状态指示灯
        status_layout = QHBoxLayout()
        self.status_indicator = StatusIndicator()
        self.status_label = QLabel("摄像头状态:")
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.status_indicator)
        layout.addLayout(status_layout)

        # 人脸计数
        self.face_count = QLabel("检测到人脸: 0")
        layout.addWidget(self.face_count)

        # 消息框
        self.message_box = QLabel()
        self.message_box.setAlignment(Qt.AlignTop)
        layout.addWidget(self.message_box)

        self.setLayout(layout)

    def update_camera_status(self, is_on):
        self.status_indicator.set_color(QColor(0, 255, 0) if is_on else QColor(255, 0, 0))

    def update_face_count(self, count):
        self.face_count.setText(f"检测到人脸: {count}")

    def show_message(self, msg, is_error=False):
        color = "red" if is_error else "black"
        self.message_box.setText(f'<font color="{color}">{msg}</font>')

    def update_fps(self, fps):
        """更新FPS显示"""
        self.fps_label.setText(f"FPS: {fps:.1f}" if fps else "FPS: --")
