# core/recognition/face_system.py
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from config import detcfg


class FaceRecognitionSystem:
    def __init__(self):
        # 初始化MTCNN检测器
        self.mtcnn = MTCNN(
            keep_all=True,
            thresholds=[0.6, 0.7, 0.7],
            post_process=False,
            device=detcfg.device,
            select_largest=False
        )

        # 初始化FaceNet模型
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(detcfg.device)
        self._init_preprocess()

        # 注册用户特征库 {user_id: embedding}
        self.registered_features = {}

    def _init_preprocess(self):
        """初始化标准预处理流程"""
        self.preprocess = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def detect_and_extract(self, img):
        """
        一站式检测与特征提取
        返回: (boxes, probs, landmarks, embeddings)
        """
        # 转换输入格式
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # 执行检测
        boxes, probs, landmarks = self.mtcnn.detect(img, landmarks=True)

        return boxes, probs, landmarks

    def get_embedding(self, img, boxes):
        # 提取特征
        embeddings = []
        img = Image.open(img).convert('RGB')
        if boxes is not None:
            face_tensors = []
            for box in boxes:
                face = img.crop(box)
                try:
                    processed = self.preprocess(face).to(detcfg.device)
                    face_tensors.append(processed)
                except:
                    continue

            if face_tensors:
                batch = torch.stack(face_tensors)
                with torch.no_grad():
                    embeddings = self.resnet(batch).cpu().numpy()
        return embeddings

    def draw_detections(self, img, boxes, probs, landmarks, similarity_matrix=None):
        """
        可视化检测结果
        :param similarity_matrix: 人脸相似度矩阵 (n,n)
        """
        if isinstance(img, (str, Image.Image)):
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # 绘制基础信息
        cv2.putText(img, f"Persons: {len(boxes)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 绘制每个人脸信息
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            x1, y1, x2, y2 = map(int, box)

            # 绘制边界框和置信度
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{prob:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 绘制关键点
            for (x, y) in landmarks[i]:
                cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

            # 绘制相似度信息
            if similarity_matrix is not None:
                max_sim = np.max(similarity_matrix[i])
                cv2.putText(img, f"Sim: {max_sim:.2f}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return img

    def calculate_similarity(self, embeddings, metric='cosine'):
        """计算特征相似度矩阵"""
        if metric == 'cosine':
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return np.dot(embeddings, embeddings.T) / (norm * norm.T)
        elif metric == 'euclidean':
            dist = np.linalg.norm(embeddings[:, None] - embeddings, axis=2)
            return 1 / (1 + dist)
        else:
            raise ValueError("Unsupported metric")

    def register_user(self, user_id, embedding):
        """注册用户特征"""
        self.registered_features[user_id] = embedding

    def verify_user(self, embedding, threshold=0.7):
        """验证用户身份"""
        similarities = []
        for uid, reg_emb in self.registered_features.items():
            sim = np.dot(embedding, reg_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(reg_emb))
            similarities.append((uid, sim))

        max_uid, max_sim = max(similarities, key=lambda x: x[1])
        return max_uid if max_sim > threshold else None


# 使用示例
if __name__ == "__main__":
    system = FaceRecognitionSystem()

    # 测试图片路径
    img_path = detcfg.test_img

    # 执行检测与特征提取
    boxes, probs, landmarks = system.detect_and_extract(img_path)
    embeddings = system.get_embedding(img_path, boxes)
    print(embeddings.shape)
    # 计算相似度矩阵
    sim_matrix = system.calculate_similarity(embeddings) if len(embeddings) > 0 else None
    print(sim_matrix)
    # 可视化结果
    vis_img = system.draw_detections(cv2.imread(img_path), boxes, probs, landmarks, sim_matrix)

    # 显示结果
    cv2.imshow("Detection Result", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 注册与验证示例
    if len(embeddings) > 0:
        system.register_user("test_user", embeddings[0])
        print("验证结果:", system.verify_user(embeddings[0]))
