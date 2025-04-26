# liveness.py
import cv2
import time
import numpy as np
from PIL import Image
from mtcnn import FaceRecognitionSystem


class LivenessDetector:
    def __init__(self, movement_threshold=3, required_movements=3):
        """活体检测器
        Args:
            movement_threshold (int): 关键点移动判定阈值（像素）
            required_movements (int): 需要检测到的最小有效移动次数
        """
        self.face_system = FaceRecognitionSystem()
        self.movement_threshold = movement_threshold
        self.required_movements = required_movements
        self.reset()

    def reset(self):
        """重置检测状态"""
        self.prev_landmarks = None
        self.movement_count = 0

    def _calculate_movement(self, current_landmarks):
        """计算关键点移动量"""
        if self.prev_landmarks is None:
            return 0
        return np.mean(np.abs(current_landmarks - self.prev_landmarks))

    def detect_frame(self, frame):
        """单帧检测
        Returns:
            bool: 是否检测到活体特征
            np.ndarray: 检测结果可视化图像
        """
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, probs, landmarks = self.face_system.detect_and_extract(img)

        is_live = False
        if boxes is not None and len(boxes) > 0:
            current = landmarks[0]  # 仅处理第一个检测到的人脸
            movement = self._calculate_movement(current)

            if movement > self.movement_threshold:
                self.movement_count += 1

            self.prev_landmarks = current
            is_live = self.movement_count >= self.required_movements

            # 绘制检测信息
            x1, y1, x2, y2 = map(int, boxes[0])
            color = (0, 255, 0) if is_live else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"Live: {self.movement_count}/{self.required_movements}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return is_live, frame


def run():
    """执行3秒活体检测
    Returns:
        bool: 是否通过活体检测
    """
    detector = LivenessDetector()
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    try:
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if not ret:
                break

            is_live, _ = detector.detect_frame(frame)
            if is_live:
                return True
    finally:
        cap.release()

    return detector.movement_count >= detector.required_movements


def test():
    """实时活体检测演示"""
    detector = LivenessDetector()
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            is_live, vis_frame = detector.detect_frame(frame)
            cv2.imshow('Live Detection', vis_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 测试实时检测
    # test()

    # 测试3秒检测
    result = run()
    print(f"Live detection result: {result}")
