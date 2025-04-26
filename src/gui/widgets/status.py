import time

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QTextBrowser
from PyQt5.QtGui import QColor, QPainter, QFont, QTextCursor
from PyQt5.QtCore import Qt, QTimer


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

        # 新增FPS显示布局
        fps_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setFont(QFont("Arial", 12))
        fps_layout.addWidget(self.fps_label)
        layout.insertLayout(0, fps_layout)  # 插入到最顶部

        # 新增检测方式显示
        self.method_label = QLabel("当前检测方式: --")
        self.method_label.setFont(QFont("Arial", 12))
        layout.insertWidget(2, self.method_label)  # 插入到人脸计数上方

        # 系统消息
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

        # 消息自动清理定时器
        self.msg_timer = QTimer(self)
        self.msg_timer.timeout.connect(self._refresh_messages)
        self.msg_timer.start(5000)  # 每5秒清理旧消息

        self.message_history = []
        layout.addWidget(self.message_box)
        self.setLayout(layout)

    def update_detection_method(self, method):
        """更新检测方式显示"""
        color_map = {
            "OpenCV": "#4CAF50",
            "MTCNN": "#2196F3",
            "Dlib": "#9C27B0"
        }
        self.method_label.setText(f"当前检测方式: {method}")
        self.method_label.setStyleSheet(f"color: {color_map.get(method, '#666')};")


    def _refresh_messages(self):
        """保留最近10条消息"""
        if len(self.message_history) > 10:
            self.message_history = self.message_history[-10:]
            self._update_message_display()

    def update_fps(self, fps):
        """更新FPS显示"""
        if fps > 0:
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self.fps_label.setStyleSheet("color: #4CAF50;")
        else:
            self.fps_label.setText("FPS: --")
            self.fps_label.setStyleSheet("color: #666;")

    def update_camera_status(self, is_on):
        color = QColor(0, 255, 0) if is_on else QColor(255, 0, 0)
        self.cam_indicator.set_color(color)

    def update_detection_status(self, is_detecting):
        color = QColor(0, 255, 0) if is_detecting else QColor(255, 0, 0)
        self.det_indicator.set_color(color)

    def update_face_count(self, count):
        self.face_count.setText(f"检测到人脸: {count}")

    def show_message(self, msg, is_error=False):
        """显示新消息并保留历史"""
        css_class = "error" if is_error else "success"
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        new_msg = f'<span class="{css_class}">[{timestamp}] ▶ {msg}</span>'

        self.message_history.append(new_msg)
        self._update_message_display()

    def _update_message_display(self):
        """更新消息显示"""
        self.message_box.clear()
        html = "<style>.error{color:#ff4444} .success{color:#44ff44}</style>"
        html += "<br>".join(self.message_history[-5:])  # 最多显示5条
        self.message_box.setHtml(html)

        # 自动滚动到底部
        cursor = self.message_box.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.message_box.setTextCursor(cursor)
