import cv2
import numpy as np
from config import detcfg, dbcfg
from detect.liveness import BlinkDetector
from detect.mtcnn import FaceRecognitionSystem
from utils.encryption import BcryptHasher, AESEncryptor


def facerecognition():
    system = FaceRecognitionSystem()

    # 测试图片路径
    img_path = detcfg.test_img

    # 执行检测与特征提取
    boxes, probs, landmarks, angles = system.detect_and_extract(img_path)
    embeddings = system.get_embedding(img_path, boxes)
    # print(embeddings.shape)
    # 计算相似度矩阵
    sim_matrix = system.calculate_similarity(embeddings, embeddings) if len(embeddings) > 0 else None
    # print(sim_matrix)
    # 可视化结果
    vis_img = system.draw_detections(cv2.imread(img_path), boxes, probs, landmarks, sim_matrix)

    # 显示结果
    cv2.imshow("Detection Result", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def blinkdetector(self):
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

def bcrypthasher():
    # 密码哈希测试
    password = dbcfg.password
    hashed = BcryptHasher.generate(password)
    print(f"哈希结果示例: {hashed}")
    print(f"验证成功: {BcryptHasher.verify(password, hashed)}")
    print(f"验证失败: {BcryptHasher.verify('wrongpass', hashed)}")

def aesencryptor():
    # 特征加密测试
    encryptor = AESEncryptor()
    test_feature = np.random.randn(512).astype(np.float32)
    encrypted = encryptor.encrypt_feature(test_feature)
    decrypted = encryptor.decrypt_feature(encrypted)

    print(f"加密前: {test_feature[:5]}")
    print(f"解密后: {decrypted[:5]}")
    print(f"数据一致性: {np.allclose(test_feature, decrypted)}")

if __name__ == '__main__':
    aesencryptor()