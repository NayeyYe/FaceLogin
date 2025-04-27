# blink_detection.py
import os
from config import cfg
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist


class BlinkDetector:
    def __init__(self,
                 shape_predictor="shape_predictor_68_face_landmarks.dat",
                 ear_threshold=0.25,
                 consec_frames=3):
        """
        纯眨眼检测器
        参数:
            shape_predictor: 人脸关键点检测模型路径
            ear_threshold: 眼睛纵横比阈值
            consec_frames: 连续帧数阈值
        """
        # 初始化dlib组件
        shape_predictor = os.path.join(cfg.root_dir, shape_predictor)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor)

        # 眼睛关键点索引 (dlib 68点模型)
        self.LEFT_EYE_START, self.LEFT_EYE_END = 36, 42
        self.RIGHT_EYE_START, self.RIGHT_EYE_END = 42, 48

        # 检测参数
        self.EAR_THRESH = ear_threshold
        self.CONSEC_FRAMES = consec_frames

        # 状态计数器
        self.eye_counter = 0
        self.total_blinks = 0
        self.ear_history = []

    def _eye_aspect_ratio(self, eye):
        """计算眼睛纵横比(EAR)"""
        # 计算垂直距离
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # 计算水平距离
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def detect(self, frame):
        """
        执行单帧检测
        返回:
            result: 包含检测结果的字典
            visualization: 可视化后的帧
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        result = {
            "blink_detected": False,
            "ear_value": 0.0,
            "total_blinks": self.total_blinks
        }

        for rect in rects:
            # 获取面部关键点
            shape = self.predictor(gray, rect)
            shape = np.array([(p.x, p.y) for p in shape.parts()])

            # 提取双眼坐标
            left_eye = shape[self.LEFT_EYE_START:self.LEFT_EYE_END]
            right_eye = shape[self.RIGHT_EYE_START:self.RIGHT_EYE_END]

            # 计算EAR
            left_ear = self._eye_aspect_ratio(left_eye)
            right_ear = self._eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            self.ear_history.append(ear)

            # 保持最近5次EAR值的平滑处理
            if len(self.ear_history) > 5:
                self.ear_history.pop(0)
            smooth_ear = np.mean(self.ear_history)

            # 眨眼检测逻辑
            if smooth_ear < self.EAR_THRESH:
                self.eye_counter += 1
            else:
                if self.eye_counter >= self.CONSEC_FRAMES:
                    self.total_blinks += 1
                    result["blink_detected"] = True
                self.eye_counter = 0

            # 更新返回结果
            result.update({
                "ear_value": smooth_ear,
                "total_blinks": self.total_blinks
            })

            # 可视化标注
            frame = self._draw_annotations(frame, left_eye, right_eye, smooth_ear)

        return result, frame

    def _draw_annotations(self, frame, left_eye, right_eye, ear):
        """绘制检测结果标注"""
        # 绘制眼部区域
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

        # 显示EAR值和眨眼次数
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Blinks: {self.total_blinks}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 绘制实时阈值线
        h, w = frame.shape[:2]
        cv2.line(frame, (0, int(h * 0.8)), (w, int(h * 0.8)), (255, 0, 0), 2)
        current_pos = int(w * 0.9 * (ear / 0.4))  # 假设EAR最大0.4
        cv2.line(frame, (current_pos, h - 50), (current_pos, h), (0, 255, 0), 10)

        return frame

    def reset(self):
        """重置检测状态"""
        self.eye_counter = 0
        self.total_blinks = 0
        self.ear_history = []


def live_blink_test():
    detector = BlinkDetector()
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result, vis_frame = detector.detect(frame)

            # 添加状态提示
            status_text = "Blink Detected!" if result["blink_detected"] else "Normal"
            cv2.putText(vis_frame, status_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Blink Detection", vis_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    live_blink_test()
