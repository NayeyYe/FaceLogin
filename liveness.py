# liveness.py
import cv2
import numpy as np
from mtcnn import FaceRecognitionSystem  # 从现有文件导入MTCNN
from time import time


class LivenessDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.face_system = FaceRecognitionSystem()
        self.eye_ar_thresh = 0.25  # 眼睛纵横比阈值
        self.eye_ar_frames = 3  # 连续帧数阈值
        self.blink_counter = 0  # 眨眼计数器
        self.blink_threshold = 2  # 成功眨眼次数

        # 状态跟踪
        self.face_angle_threshold = 20  # 头部姿态角度阈值
        self.last_movement = {'timestamp': time(), 'position': None}

    def _calculate_ear(self, eye_points):
        """计算眼睛纵横比(Eye Aspect Ratio)"""
        # MTCNN的5个关键点排列为：左右眼、鼻尖、左右嘴角
        # 使用简化的两点计算法（实际EAR通常需要6个点）
        width = np.linalg.norm(eye_points[0] - eye_points[1])
        height = np.linalg.norm(eye_points[0] - eye_points[1])
        return height / (width + 1e-6)  # 防止除以零

    def _check_head_movement(self, current_points):
        """检测头部自然微动"""
        if self.last_movement['position'] is None:
            self.last_movement['position'] = current_points
            return False

        # 计算关键点平均移动距离
        movement = np.mean(np.linalg.norm(
            current_points - self.last_movement['position'],
            axis=1
        ))

        # 更新时间戳和位置
        self.last_movement['position'] = current_points
        self.last_movement['timestamp'] = time()
        return movement > 2.0  # 移动距离阈值

    def analyze_frame(self, frame):
        """核心检测逻辑"""
        boxes, probs, landmarks = self.face_system.detect_and_extract(frame)

        if landmarks is None or len(landmarks) == 0:
            return False, frame

        # 取第一个检测到的人脸
        face_landmarks = landmarks[0]

        # 提取眼部关键点（MTCNN的索引0为左眼，1为右眼）
        left_eye = face_landmarks[0]
        right_eye = face_landmarks[1]

        # 计算眼睛纵横比
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # 眨眼检测逻辑
        if avg_ear < self.eye_ar_thresh:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.eye_ar_frames:
                self.blink_threshold -= 1
            self.blink_counter = 0

        # 头部微动检测
        head_moved = self._check_head_movement(face_landmarks)

        # 绘制检测信息
        frame = self.draw_detection_info(
            frame, avg_ear, head_moved, boxes[0]
        )

        # 综合判断条件
        liveness_detected = (
                self.blink_threshold <= 0 and
                head_moved and
                (time() - self.last_movement['timestamp']) < 5
        )
        return liveness_detected, frame

    def draw_detection_info(self, frame, ear, head_moved, box):
        """可视化检测信息"""
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 显示眼部信息
        cv2.putText(frame, f"EAR: {ear:.2f}", (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Blinks left: {self.blink_threshold}", (x1, y1 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Head moved: {head_moved}", (x1, y1 - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame

    def run(self):
        """主运行循环"""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # 水平翻转以获得镜像视图
            frame = cv2.flip(frame, 1)

            # 执行活体检测
            detected, output_frame = self.analyze_frame(frame)

            # 显示结果
            cv2.imshow('Liveness Detection', output_frame)

            # 退出条件
            if detected:
                print("Live face detected!")
                break
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = LivenessDetector()
    detector.run()
