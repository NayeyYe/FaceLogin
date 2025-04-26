import cv2
from PIL import Image
from config import detcfg
from facenet_pytorch import MTCNN, InceptionResnetV1

class OfficialMTCNN:
    def __init__(self):
        # 官方MTCNN初始化（启用CUDA加速）
        self.mtcnn = MTCNN(
            keep_all=True,
            thresholds=[0.6, 0.7, 0.7],  # 与原参数保持一致
            post_process=False,
            device=detcfg.device,
            select_largest=False  # 适合学生项目的设置
        )

        self.device = detcfg.device
        # 特征编码器
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def detect_faces(self, img):
        """一站式检测与识别接口"""
        # 官方MTCNN检测
        boxes, probs, landmarks = self.mtcnn.detect(img, landmarks=True)

        # 新增人脸数量统计
        face_count = len(boxes) if boxes is not None else 0
        print(f"检测到 {face_count} 张人脸")  # 控制台输出
        return boxes, probs, landmarks

    def draw_detections(self, img_path, boxes, probs, landmarks):
        """可视化增强版（兼容OpenCV）"""
        img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        if boxes is not None:
            # 在图像左上角添加人数统计
            face_count = len(boxes)
            cv2.putText(img, f"Persons: {face_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                x1, y1, x2, y2 = map(int, box)
                # 绘制边界框
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # 置信度显示
                cv2.putText(img, f"{prob:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # 关键点绘制
                if landmarks is not None:
                    for point in landmarks[i]:
                        x, y = map(int, point)
                        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
        else:
            # 未检测到人脸时显示提示
            cv2.putText(img, "No faces detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return img

    def adaptive_scale(self, frame_size):
        """动态调整检测参数（优化性能）"""
        w, h = frame_size
        if w * h >= 1920 * 1080:
            self.mtcnn.margin = 40
            self.mtcnn.factor = 0.7
        else:
            self.mtcnn.margin = 20
            self.mtcnn.factor = 0.6


# 使用示例
if __name__ == "__main__":
    # 初始化
    mtcnn = OfficialMTCNN()

    # 一站式调用（检测+特征提取）
    img_path = detcfg.test_img
    img = Image.open(img_path)
    boxes, probs, landmarks = mtcnn.detect_faces(img)

    # 可视化（兼容原接口）
    vis_img = mtcnn.draw_detections(img_path, boxes, probs, landmarks)
    cv2.imshow('img', vis_img)
    cv2.waitKey(0)
