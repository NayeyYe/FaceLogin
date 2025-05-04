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
        # 数据保存目录
        self.database_dir = os.path.join(self.root_dir, 'db')
        self.csv_dir = os.path.join(self.database_dir, 'info.csv')
        # gui目录
        self.gui_dir = os.path.join(self.root_dir, 'gui')
        # logs目录
        self.logs_dir = os.path.join(self.root_dir, 'logs')

class DBConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.host = 'localhost'
        self.port = 3306
        self.user = 'root'
        self.password = '13Password,'
        self.database = 'facelogin'
        self.AES_KEY = Fernet.generate_key()

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
        self.logs_dir = os.path.join(self.root_dir, 'logs')

cfg = BaseConfig()
dbcfg = DBConfig()
detcfg = DetectionConfig()
logscfg = LogsConfig()

if __name__ == '__main__':
    print(cfg.device)