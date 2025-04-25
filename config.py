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
        # 核心代码目录
        self.core_dir = os.path.join(self.root_dir, 'core')
        # 数据保存目录
        self.database_dir = os.path.join(self.root_dir, 'db')
        self.csv_dir = os.path.join(self.database_dir, 'info')
        # gui目录
        self.gui_dir = os.path.join(self.root_dir, 'gui')
        self.assets_dir = os.path.join(self.root_dir, 'assets')
        self.icons_dir = os.path.join(self.assets_dir, 'icons')
        # logs目录
        self.logs_dir = os.path.join(self.root_dir, 'logs')
        # 测试目录
        self.tests_dir = os.path.join(self.root_dir, 'tests')
        self.data_dir = os.path.join(self.tests_dir, 'data')
        self.faces_dir = os.path.join(self.data_dir, 'faces')
        self.features_dir = os.path.join(self.data_dir, 'features')

class RootConfig(BaseConfig):
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
        # 预训练权重
        self.weight_dir = os.path.join(self.root_dir, 'weights')
        self.pnet_weight = os.path.join(self.weight_dir, 'pnet.pt')
        self.onet_weight = os.path.join(self.weight_dir, 'onet.pt')
        self.rnet_weight = os.path.join(self.weight_dir, 'rnet.pt')


class GUIConfig(BaseConfig):
    def __init__(self):
        super().__init__()


class LogsConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.logs_dir = os.path.join(self.root_dir, 'logs')

class TestConfig(BaseConfig):
    def __init__(self):
        super().__init__()

cfg = BaseConfig()
rootcfg = RootConfig()
detcfg = DetectionConfig()
logscfg = LogsConfig()

if __name__ == '__main__':
    print(cfg.device)