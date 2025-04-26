import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torch.nn.functional import embedding
from torchvision import transforms
from config import detcfg
import cv2

from ..utils.logger import enable_logging
from mtcnn import OfficialMTCNN


class FaceNetModel:
    def __init__(self, pretrained='vggface2'):
        """
        初始化FaceNet模型
        :param pretrained: 预训练模型类型 ('vggface2' 或 'casia-webface')
        """
        self.device = detcfg.device
        self.resnet = InceptionResnetV1(pretrained=pretrained, classify=False).eval().to(self.device)
        self._init_preprocess()

    def _init_preprocess(self):
        """初始化标准预处理流程"""
        self.preprocess = transforms.Compose([
            transforms.Resize((160, 160)),  # FaceNet标准输入尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def preprocess_image(self, img):
        """
        图像预处理方法（支持多种输入格式）
        :param img: 输入图像 (支持文件路径/PIL.Image/numpy.ndarray)
        :return: 预处理后的图像张量 [1,3,160,160]
        """
        if isinstance(img, str):
            # 处理文件路径输入
            img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            # 处理numpy数组输入 (OpenCV格式)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # 应用预处理流程
        return self.preprocess(img).unsqueeze(0).to(self.device)

    def extract_multi_faces(self, image, face_boxes):
        """
        处理单张图像中的多个人脸
        :param image: 原始图像 (numpy数组或PIL.Image)
        :param face_boxes: 人脸框坐标 [[x1,y1,x2,y2], ...]
        :return: 特征矩阵 (n,512), 对应的人脸框
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        face_tensors = []
        valid_boxes = []

        for box in face_boxes:
            # 提取人脸区域
            face = image.crop(box)

            try:
                # 预处理并验证尺寸
                processed = self.preprocess(face)
                if processed.shape == (3, 160, 160):
                    face_tensors.append(processed)
                    valid_boxes.append(box)
            except:
                continue

        if not face_tensors:
            return np.array([]), []

        # 批量处理
        batch = torch.stack(face_tensors).to(self.device)
        embeddings = self.get_embedding(batch)
        return embeddings, valid_boxes

    def get_embedding(self, img_tensor):
        """
        提取人脸特征向量
        :param img_tensor: 预处理后的图像张量 [batch,3,160,160]
        :return: 512维特征向量 (numpy数组)
        """
        with torch.no_grad(), torch.amp.autocast('cuda'):
            embedding = self.resnet(img_tensor)
        return embedding.cpu().numpy().astype(np.float32)

    def batch_process(self, img_list):
        """
        批量处理接口（优化显存使用）
        :param img_list: 图像路径/PIL图像列表
        :return: 特征向量矩阵 [n,512]
        """
        batch_tensors = [self.preprocess_image(img) for img in img_list]
        batch = torch.cat(batch_tensors, dim=0)
        return self.get_embedding(batch)

    @torch.no_grad()
    def similarity(self, emb1, emb2, metric='cosine'):
        """
        相似度计算接口
        :param emb1: 特征向量1
        :param emb2: 特征向量2
        :param metric: 相似度度量方式 ('cosine'/'euclidean')
        """
        if metric == 'cosine':
            return torch.cosine_similarity(
                torch.tensor(emb1),
                torch.tensor(emb2))
        elif metric == 'euclidean':
            return torch.pairwise_distance(
                torch.tensor(emb1),
                torch.tensor(emb2))
        else:
            raise ValueError("不支持的相似度度量方式")

# 使用示例
if __name__ == "__main__":
    logger = enable_logging()
    # 初始化模型
    facenet = FaceNetModel(pretrained='vggface2')
    img_path = detcfg.test_img

    # 测试多人脸场景
    test_img = cv2.imread(detcfg.test_img)
    img = Image.open(img_path)

    # 假设通过MTCNN检测到n个人脸
    mtcnn = OfficialMTCNN()
    boxes, probs, landmarks = mtcnn.detect_faces(img)
    face_boxes = boxes

    # 提取多人脸特征
    embeddings, valid_boxes = facenet.extract_multi_faces(test_img, face_boxes)

    print(f"检测到有效人脸数: {len(valid_boxes)}")
    print(f"特征矩阵形状: {embeddings.shape}")  # 应输出 (n,512)
    print(f"特征样例:\n{embeddings[0][:5]}")

    sims = []
    for i in range(len(face_boxes)):
        embedding = embeddings[i:i+1, :]
        sim = facenet.similarity(embedding, embedding)
        sims.append(sim.item())
    print(sims)
