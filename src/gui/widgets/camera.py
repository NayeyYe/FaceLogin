import time

import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter, QFont
from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout
import cv2

from src.detection.mtcnn import FaceRecognitionSystem


class CameraWidget(QLabel):
    # 新增信号
    detection_toggled = pyqtSignal(bool)
    faces_detected = pyqtSignal(int)
    camera_status_changed = pyqtSignal(bool)  # 新增状态信号
    fps_updated = pyqtSignal(float)  # 新增FPS信号


    def __init__(self, parent=None):
        super().__init__(parent)
        # 初始化布局
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        # 状态标签
        self.status_label = QLabel("摄像头未开启", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: white; font-size: 24px;")
        self.layout.addWidget(self.status_label)

        # 图像显示区域
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        # 初始化参数
        self._is_camera_on = False
        self._show_gray_background()
        self._is_detecting = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_frame)
        self._cap = None
        self.frame_times = []

        self.detection_method = "OpenCV"

        # 添加人脸识别系统初始化
        self.face_system = FaceRecognitionSystem()
        self.last_frame = None  # 用于保存最新帧

    def set_detection_method(self, method):
        """设置检测方法接口"""
        self.detection_method = method
        print(f"Switched to {method} detection")  # 临时调试输出

    def toggle_camera(self):
        if not self._is_camera_on:
            self._start_camera()
        else:
            self._stop_camera()

    def toggle_detection(self):
        self._is_detecting = not self._is_detecting
        self.detection_toggled.emit(self._is_detecting)

    def _start_camera(self):
        self._cap = cv2.VideoCapture(0)
        if self._cap.isOpened():
            self._is_camera_on = True
            self.status_label.hide()
            self._timer.start(30)
            self.setStyleSheet("background-color: black;")
            self.camera_status_changed.emit(True)  # 新增信号发射

    def _stop_camera(self):
        self._is_camera_on = False
        # 强制关闭检测状态并发送信号
        if self._is_detecting:
            self._is_detecting = False
            self.detection_toggled.emit(False)
        if self._cap:
            self._cap.release()
        self._timer.stop()
        self._show_gray_background()
        self.faces_detected.emit(0)
        self.camera_status_changed.emit(False)  # 新增信号发射
        self.fps_updated.emit(0)

    def _show_gray_background(self):
        self.status_label.show()
        self.image_label.clear()
        self.setStyleSheet("background-color: #646464;")
        self.status_label.setText("摄像头未开启")

    def _update_frame(self):
        start_time = time.time()
        ret, frame = self._cap.read()
        # 计算FPS
        self.frame_times.append(time.time())
        self.frame_times = self.frame_times[-30:]
        fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0]) if len(self.frame_times) > 1 else 0
        self.fps_updated.emit(fps)  # 发射信号

        if ret:
            # 人脸检测逻辑（示例）
            # 性能优化：当FPS低于15时自动跳帧
            if fps < 15 and len(self.frame_times) > 10:
                return
            if self._is_detecting:
                # TODO: 调用实际的人脸检测方法
                try:
                    faces = self._mock_detect_faces(frame)
                    self.faces_detected.emit(len(faces))
                    frame = self._draw_detections(frame, faces)
                except Exception as e:
                    print(f"检测异常: {str(e)}")

            # 转换图像格式
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 显示图像
            pixmap = QPixmap.fromImage(q_img).scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)



    def _mock_detect_faces(self, frame):
        """示例检测方法切换"""
        self.status_label.setText(f"当前检测方式: {self.detection_method}")

        if self.detection_method == "MTCNN":
            try:
                # 转换图像格式为PIL
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # 执行MTCNN检测
                boxes, _, _ = self.face_system.detect_and_extract(img_pil)

                # 转换坐标格式为(x,y,w,h)
                faces = []
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        faces.append((x1, y1, x2 - x1, y2 - y1))
                return faces
            except Exception as e:
                print(f"MTCNN检测异常: {str(e)}")
                return []
        elif self.detection_method == "Dlib":
            # TODO: 调用Dlib检测
            return np.array([])  # 示例返回
        else:  # OpenCV
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            return face_cascade.detectMultiScale(gray, 1.1, 4)

    def _draw_detections(self, frame, faces):
        # 绘制检测框
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame
