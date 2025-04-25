import time

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import QLabel, QWidget
import cv2

class CameraWidget(QLabel):
    frame_updated = pyqtSignal(QImage)
    camera_status_changed = pyqtSignal(bool)  # 新增状态信号
    fps_updated = pyqtSignal(float)  # 新增FPS信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self._is_camera_on = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_frame)
        self._cap = None
        self._show_gray_background()  # 初始灰色背景
        self.frame_times = []

    def _show_gray_background(self):
        """绘制灰色背景"""
        pixmap = QPixmap(self.size())
        pixmap.fill(QColor(100, 100, 100))
        self.setPixmap(pixmap)

    def resizeEvent(self, event):
        """窗口大小变化时重绘背景"""
        super().resizeEvent(event)
        if not self._is_camera_on:
            self._show_gray_background()

    def toggle_camera(self):
        if not self._is_camera_on:
            self._start_camera()
        else:
            self._stop_camera()

    def _start_camera(self):
        self._cap = cv2.VideoCapture(0)
        if self._cap.isOpened():
            self._is_camera_on = True
            self._timer.start(30)
            self.frame_updated.connect(self._display_frame)
            self.setStyleSheet("")  # 清除背景色
            self.camera_status_changed.emit(True)

    def _stop_camera(self):
        self._is_camera_on = False
        self._timer.stop()
        if self._cap:
            self._cap.release()
        self.clear()
        self._show_gray_background()
        self.camera_status_changed.emit(False)
        self.fps_updated.emit(0)

    def _update_frame(self):
        start_time = time.time()
        ret, frame = self._cap.read()
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = image.shape
            bytes_per_line = ch * w
            q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frame_updated.emit(q_img)

        # 计算FPS
        self.frame_times.append(start_time)
        self.frame_times = self.frame_times[-30:]  # 保留最近30帧
        if len(self.frame_times) >= 2:
            fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
            self.fps_updated.emit(fps)

    def _display_frame(self, q_img):
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation  # 添加平滑缩放
        )
        self.setPixmap(pixmap)
