import sys

sys.path.append("../")
import pymysql
import numpy as np
from config import dbcfg
from utils.encryption import BcryptHasher, AESEncryptor


class DBOperator:
    """数据库操作核心类"""

    def __init__(self):
        self.aes = AESEncryptor()
        self.conn = None

    def __enter__(self):
        self.connect()
        return self

    def connect(self):
        """建立数据库连接"""
        try:
            self.conn = pymysql.connect(
                host=dbcfg.host,
                port=dbcfg.port,
                user=dbcfg.super_admin,
                password=dbcfg.password,
                database=dbcfg.database,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
        except pymysql.Error as e:
            raise ConnectionError(f"数据库连接失败: {str(e)}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """关闭数据库连接"""
        if self.conn and self.conn.open:
            self.conn.close()

    def register_user(self, name: str, password: str, feature: np.ndarray) -> bool:
        """用户注册"""
        try:
            with self.conn.cursor() as cursor:
                # 检查用户是否存在
                if self.get_user(name):
                    raise ValueError("用户已存在")

                # 密码哈希和特征加密
                hashed_pwd = BcryptHasher.generate(password)
                encrypted_feature = self.aes.encrypt_feature(feature)

                sql = """INSERT INTO users(name, password_hash, face_feature) VALUES (%s, %s, %s)"""
                cursor.execute(sql, (name, hashed_pwd, encrypted_feature))
                self.conn.commit()
                return True
        except pymysql.Error as e:
            self.conn.rollback()
            raise RuntimeError(f"注册失败: {str(e)}")

    def verify_user(self, name: str, password: str) -> dict:
        """用户验证"""
        try:
            with self.conn.cursor() as cursor:
                sql = "SELECT * FROM users WHERE name = %s"
                cursor.execute(sql, (name,))
                user = cursor.fetchone()

                if not user:
                    raise ValueError("用户不存在")

                if not BcryptHasher.verify(password, user['password_hash']):
                    raise ValueError("密码错误")

                # 解密特征向量
                user['face_feature'] = self.aes.decrypt_feature(user['face_feature'])
                return user
        except pymysql.Error as e:
            raise RuntimeError(f"查询失败: {str(e)}")

    def get_user(self, name: str) -> dict:
        """获取用户信息"""
        try:
            with self.conn.cursor() as cursor:
                sql = "SELECT * FROM users WHERE name = %s"
                cursor.execute(sql, (name,))
                return cursor.fetchone()
        except pymysql.Error as e:
            raise RuntimeError(f"查询失败: {str(e)}")
