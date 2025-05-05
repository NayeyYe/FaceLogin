import time
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel,
                             QHBoxLayout, QTextBrowser)
from PyQt5.QtGui import QColor, QPainter, QFont, QTextCursor
from PyQt5.QtCore import QTimer


class StatusIndicator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
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
        self.message_history = []
        self._init_ui()
        self._init_message_system()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)

        self._create_fps_label(layout)
        self._create_status_indicators(layout)
        self._create_method_label(layout)
        self._create_face_counter(layout)
        self._create_message_box(layout)

        self.setLayout(layout)

    def _init_message_system(self):
        self.msg_timer = QTimer(self)
        self.msg_timer.timeout.connect(self._refresh_messages)
        self.msg_timer.start(5000)

    def _create_fps_label(self, layout):
        fps_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setFont(QFont("Arial", 12))
        fps_layout.addWidget(self.fps_label)
        layout.insertLayout(0, fps_layout)

    def _create_status_indicators(self, layout):
        status_grid = QHBoxLayout()

        self.cam_indicator = StatusIndicator()
        cam_layout = self._create_indicator_layout("摄像头:", self.cam_indicator)

        self.det_indicator = StatusIndicator()
        det_layout = self._create_indicator_layout("检测状态:", self.det_indicator)

        status_grid.addLayout(cam_layout)
        status_grid.addLayout(det_layout)
        layout.addLayout(status_grid)

    def _create_indicator_layout(self, text, indicator):
        h_layout = QHBoxLayout()
        label = QLabel(text)
        h_layout.addWidget(label)
        h_layout.addWidget(indicator)
        return h_layout

    def _create_method_label(self, layout):
        self.method_label = QLabel("当前检测方式: --")
        self.method_label.setFont(QFont("Arial", 12))
        layout.insertWidget(2, self.method_label)

    def _create_face_counter(self, layout):
        self.face_count = QLabel("检测到人脸: 0")
        self.face_count.setFont(QFont("Arial", 12))
        layout.addWidget(self.face_count)

    def _create_message_box(self, layout):
        self.message_box = QTextBrowser()
        self.message_box.setMaximumHeight(80)
        self.message_box.setStyleSheet("""
            QTextBrowser {
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                font-size: 12px;
                padding: 5px;
            }
        """)
        layout.addWidget(self.message_box)

    def update_camera_status(self, is_on):
        color = QColor(0, 255, 0) if is_on else QColor(255, 0, 0)
        self.cam_indicator.set_color(color)

    def update_detection_status(self, is_detecting):
        color = QColor(0, 255, 0) if is_detecting else QColor(255, 0, 0)
        self.det_indicator.set_color(color)

    def update_face_count(self, count):
        self.face_count.setText(f"检测到人脸: {count}")

    def update_fps(self, fps):
        if fps > 0:
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self.fps_label.setStyleSheet("color: #4CAF50;")
        else:
            self.fps_label.setText("FPS: --")
            self.fps_label.setStyleSheet("color: #666;")

    def update_detection_method(self, method):
        color_map = {"OpenCV": "#4CAF50", "MTCNN": "#2196F3"}
        self.method_label.setText(f"当前检测方式: {method}")
        self.method_label.setStyleSheet(f"color: {color_map.get(method, '#666')};")

    def show_message(self, msg, is_error=False):
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        css_class = "error" if is_error else "success"
        new_msg = f'<span class="{css_class}">[{timestamp}] ▶ {msg}</span>'
        self.message_history.append(new_msg)
        self._update_message_display()

    def _refresh_messages(self):
        if len(self.message_history) > 10:
            self.message_history = self.message_history[-10:]
            self._update_message_display()

    def _update_message_display(self):
        self.message_box.clear()
        html = "<style>.error{color:#ff4444} .success{color:#44ff44}</style>"
        html += "<br>".join(self.message_history[-5:])
        self.message_box.setHtml(html)

        cursor = self.message_box.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.message_box.setTextCursor(cursor)
