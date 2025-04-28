# core/utils/encryption.py
import bcrypt
import numpy as np
from cryptography.fernet import Fernet, InvalidToken
from config import logscfg
from logger import setup_logger
logger = setup_logger(logscfg)
class BcryptHasher:
    """密码哈希与验证模块"""

    @staticmethod
    def generate(password: str) -> str:
        """
        生成安全密码哈希
        :param password: 明文密码
        :return: bcrypt哈希字符串
        """
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    @staticmethod
    def verify(password: str, hashed: str) -> bool:
        """
        验证密码与哈希是否匹配
        :param password: 待验证密码
        :param hashed: 存储的哈希值
        :return: 验证结果
        """
        try:
            return bcrypt.checkpw(
                password.encode('utf-8'),
                hashed.encode('utf-8')
            )
        except Exception as e:
            logger.error(f"密码验证失败: {str(e)}")
            return False


class AESEncryptor:
    """人脸特征加密模块"""

    def __init__(self):
        self.cipher = Fernet(rootcfg.AES_KEY)

    def encrypt_feature(self, feature: np.ndarray) -> bytes:
        """
        加密人脸特征向量
        :param feature: 512维特征向量
        :return: 加密后的字节流
        """
        try:
            return self.cipher.encrypt(
                feature.astype(np.float32).tobytes()
            )
        except Exception as e:
            logger.error(f"特征加密失败: {str(e)}")
            raise

    def decrypt_feature(self, ciphertext: bytes) -> np.ndarray:
        """
        解密人脸特征
        :param ciphertext: 加密数据
        :return: 原始特征向量
        """
        try:
            decrypted = self.cipher.decrypt(ciphertext)
            return np.frombuffer(decrypted, dtype=np.float32)
        except InvalidToken:
            logger.error("无效的解密令牌")
            raise
        except Exception as e:
            logger.error(f"特征解密失败: {str(e)}")
            raise


# 密钥管理工具类
class KeyManager:
    """密钥生命周期管理"""

    @staticmethod
    def generate_key() -> bytes:
        """生成新的AES密钥"""
        return Fernet.generate_key()

    @staticmethod
    def validate_key(key: bytes) -> bool:
        """验证密钥有效性"""
        try:
            Fernet(key)
            return True
        except ValueError:
            return False

