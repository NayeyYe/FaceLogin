import time
import sys
sys.path.append("../")
import cv2
from PIL import Image
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout
from detect.liveness import BlinkDetector
from detect.mtcnn import FaceRecognitionSystem


class CameraWidget(QLabel):
    detection_toggled = pyqtSignal(bool)
    faces_detected = pyqtSignal(int)
    camera_status_changed = pyqtSignal(bool)
    fps_updated = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._init_variables()
        self._init_systems()

    def _init_ui(self):
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        self.status_label = QLabel("摄像头未开启", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: white; font-size: 24px;")
        self.layout.addWidget(self.status_label)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

    def _init_variables(self):
        self._is_camera_on = False
        self._is_detecting = False
        self.detection_method = "OpenCV"
        self.frame_times = []
        self.last_frame = None
        self.liveness_status = False
        self.blink_counter = 0
        self.last_blink_time = time.time()
        self.current_face_feature = None
        self.current_prob = 0.0
        self.current_angles = []
        self.registration_enabled = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_frame)
        self._show_gray_background()

    def _init_systems(self):
        self.face_system = FaceRecognitionSystem()
        self.blink_detector = BlinkDetector()

    def set_detection_method(self, method):
        self.detection_method = method

    def toggle_camera(self):
        if not self._is_camera_on:
            self._start_camera()
        else:
            self._stop_camera()

    def toggle_detection(self):
        self._is_detecting = not self._is_detecting
        self.detection_toggled.emit(self._is_detecting)

    def _start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self._is_camera_on = True
            self.status_label.hide()
            self._timer.start(30)
            self.setStyleSheet("background-color: black;")
            self.camera_status_changed.emit(True)

    def _stop_camera(self):
        self._is_camera_on = False
        if self._is_detecting:
            self._is_detecting = False
            self.detection_toggled.emit(False)
        if self.cap:
            self.cap.release()
        self._timer.stop()
        self._show_gray_background()
        self.faces_detected.emit(0)
        self.camera_status_changed.emit(False)
        self.fps_updated.emit(0)
        self.blink_detector.reset()
        self.liveness_status = False
        self.blink_counter = 0

    def _show_gray_background(self):
        self.status_label.show()
        self.image_label.clear()
        self.setStyleSheet("background-color: #646464;")
        self.status_label.setText("摄像头未开启")

    def _update_frame(self):
        start_time = time.time()
        ret, frame = self.cap.read()

        self.frame_times.append(time.time())
        self.frame_times = self.frame_times[-30:]
        fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0]) if len(self.frame_times) > 1 else 0
        self.fps_updated.emit(fps)

        if ret and fps >= 15:
            self._process_frame(frame)

    def _process_frame(self, frame):
        if self._is_detecting:
            try:
                faces = self._mock_detect_faces(frame)
                self.faces_detected.emit(len(faces))
                self._update_liveness_detection(frame)
                frame = self._draw_detections(frame, faces)
            except Exception as e:
                print(f"检测异常: {str(e)}")

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        q_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

    def _update_liveness_detection(self, frame):
        result, _ = self.blink_detector.detect(frame)
        current_time = time.time()

        if current_time - self.last_blink_time > 15.0:
            self.blink_counter = 0
            self.last_blink_time = current_time

        if result["blink_detected"]:
            self.blink_counter += 1

        self.liveness_status = self.blink_counter >= 1

    def _mock_detect_faces(self, frame):
        self.status_label.setText(f"当前检测方式: {self.detection_method}")

        if self.detection_method == "MTCNN":
            return self._mtcnn_detection(frame)
        elif self.detection_method == "OpenCV":
            return self._opencv_detection(frame)
        return []

    def _mtcnn_detection(self, frame):
        try:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes, probs, landmarks, angles = self.face_system.detect_and_extract(frame)
            faces = []

            if boxes is not None:
                self.current_face_feature = self.face_system.get_embedding(img_pil, boxes) if len(boxes) == 1 else None
                self.current_prob = probs[0] if len(boxes) > 0 else 0.0
                self.current_angles = angles[0] if len(angles) == 1 else []

                for i in range(len(boxes)):
                    faces.append({'box': boxes[i], 'prob': probs[i], 'landmarks': landmarks[i], 'angles': angles[i]})
            return faces
        except Exception as e:
            print(f"MTCNN检测异常: {str(e)}")
            return []

    def _opencv_detection(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return face_cascade.detectMultiScale(gray, 1.1, 4)

    def _draw_detections(self, frame, faces):
        if self.detection_method == "MTCNN":
            for data in faces:
                x1, y1, x2, y2 = map(int, data['box'])
                color = (0, 255, 0) if self.liveness_status else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"Live: {self.liveness_status}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if 'prob' in data:
                    cv2.putText(frame, f"{data['prob']:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if 'angles' in data:  # 假设数据中包含angles
                    pitch, yaw, roll = data['angles']
                    angle_text = f"P:{pitch:.1f} Y:{yaw:.1f} R:{roll:.1f}"
                    cv2.putText(frame, angle_text, (x1, y2 + 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        elif self.detection_method == "OpenCV":
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return frame
