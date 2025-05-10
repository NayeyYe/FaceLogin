import os
import torch
from cryptography.fernet import Fernet

class BaseConfig:
    def __init__(self):
        # 硬件配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = min(8, os.cpu_count()-4)

        # 文件目录
        self.root_dir = os.path.dirname(__file__)
        # gui目录
        self.gui_dir = os.path.join(self.root_dir, 'gui')

class DBConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.host = 'localhost'
        self.port = 3306
        self.super_admin = 'root'
        self.password = '13Password,'
        self.database = 'facelogin'
        self.AES_KEY = b'om-A9wg5K0-sZxmjBbHIL8o_iUZvIZd3g5Te9z2PpnA='
        self.users_table = 'users'
        self.admin_table = 'admins'

        # csv文件
        self.database_dir = os.path.join(self.root_dir, 'db')
        self.csv = os.path.join(self.database_dir, 'info.csv')

class DetectionConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        # 图片保存目录
        self.img_dir = os.path.join(self.root_dir, 'img')
        self.test_img = os.path.join(self.img_dir, 'test.jpg')
        # 数据集
        self.dataset_dir = os.path.join(self.root_dir, 'dataset')
        self.train_dir = os.path.join(self.dataset_dir, 'WIDER_train')
        # 预训练权重
        self.model_dir = os.path.join(self.root_dir, 'models')
        self.pnet_weight = os.path.join(self.model_dir, 'PNet.pth')
        self.rnet_weight = os.path.join(self.model_dir, 'RNet.pth')
        self.onet_weight = os.path.join(self.model_dir, 'ONet.pth')

class LogsConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.logs = os.path.join(self.root_dir, '.log')

cfg = BaseConfig()
dbcfg = DBConfig()
detcfg = DetectionConfig()
logscfg = LogsConfig()

if __name__ == '__main__':
    print(Fernet.generate_key())