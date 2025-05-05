# core/recognition/face_system.py
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from config import detcfg
from infer_camera import infer_image


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
        # if isinstance(img, str):
        #     img = Image.open(img).convert('RGB')
        # elif isinstance(img, np.ndarray):
        #     img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # img = cv2.imread(img)
        # 执行检测
        boxes_c, landmarks = infer_image(img)
        boxes, probs = boxes_c[:, :4], boxes_c[:, 4]
        # 新增姿态角度计算
        angles = []
        if landmarks is not None:
            for lm in landmarks:
                angles.append(self.estimate_head_pose(lm))

        return boxes, probs, landmarks, angles

    # 在FaceRecognitionSystem类中新增以下方法
    def estimate_head_pose(self, landmarks):
        """
        基于5点特征估计头部姿态角度（单位：度）
        :param landmarks: MTCNN返回的5个特征点坐标
        :return: (pitch, yaw, roll) 俯仰角/偏航角/翻滚角
        """
        # 特征点索引定义（根据MTCNN输出顺序）
        LEFT_EYE = 0
        RIGHT_EYE = 1
        NOSE = 2
        MOUTH_LEFT = 3
        MOUTH_RIGHT = 4

        # 转换为numpy数组方便计算
        points = np.array(landmarks, dtype=np.float32)

        # 计算两眼连线的旋转角度（Roll）
        dX = points[RIGHT_EYE][0] - points[LEFT_EYE][0]
        dY = points[RIGHT_EYE][1] - points[LEFT_EYE][1]
        roll = np.degrees(np.arctan2(dY, dX))
        roll = float(roll)
        # 计算鼻尖相对眼线的高度（Pitch）
        eye_center = (points[LEFT_EYE] + points[RIGHT_EYE]) / 2
        nose_vector = points[NOSE] - eye_center
        pitch = np.degrees(np.arctan2(nose_vector[1], np.linalg.norm(nose_vector[:1])))
        pitch = float(pitch)
        # 计算嘴部对称性（Yaw）
        mouth_center = (points[MOUTH_LEFT] + points[MOUTH_RIGHT]) / 2
        yaw_vector = mouth_center - eye_center
        yaw = np.degrees(np.arctan2(yaw_vector[0], np.linalg.norm(yaw_vector)))
        yaw = float(yaw)

        return pitch, yaw, roll

    def get_embedding(self, img, boxes):
        if not isinstance(img, Image.Image):
            # 如果是文件路径或numpy数组，转换为PIL Image
            if isinstance(img, np.ndarray):
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                img = Image.open(img).convert('RGB')

        # 提取特征
        embeddings = []
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

    def calculate_similarity(self, embeddings1, embeddings2, metric='cosine'):
        """计算特征相似度矩阵"""
        if metric == 'cosine':
            norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True) # shape: (n,1)
            norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)  # shape: (m,1)
            return np.dot(embeddings1, embeddings2.T) / (norm1 * norm2.T)
        elif metric == 'euclidean':
            dist = np.linalg.norm(embeddings1[:, None] - embeddings2, axis=2)
            return 1 / (1 + dist)
        else:
            raise ValueError("Unsupported metric")



