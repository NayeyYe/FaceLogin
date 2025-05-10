import torch.nn as nn
import torch
import numpy as np
from utils.logger import logger

class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(3, 3))
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3))
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.prelu3 = nn.PReLU()
        self.conv4_1 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(1, 1))
        self.conv4_2 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=(1, 1))
        self.conv4_3 = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        # 分类是否人脸的卷积输出层
        class_out = self.conv4_1(x)
        class_out = torch.squeeze(class_out, dim=2)
        class_out = torch.squeeze(class_out, dim=2)
        # 人脸box的回归卷积输出层
        bbox_out = self.conv4_2(x)
        bbox_out = torch.squeeze(bbox_out, dim=2)
        bbox_out = torch.squeeze(bbox_out, dim=2)
        # 5个关键点的回归卷积输出层
        landmark_out = self.conv4_3(x)
        landmark_out = torch.squeeze(landmark_out, dim=2)
        landmark_out = torch.squeeze(landmark_out, dim=2)
        return class_out, bbox_out, landmark_out

class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=28, kernel_size=(3, 3))
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(in_channels=28, out_channels=48, kernel_size=(3, 3))
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(2, 2))
        self.prelu3 = nn.PReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=576, out_features=128)
        self.class_fc = nn.Linear(in_features=128, out_features=2)
        self.bbox_fc = nn.Linear(in_features=128, out_features=4)
        self.landmark_fc = nn.Linear(in_features=128, out_features=10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.pool2(x)
        x = self.prelu3(self.conv3(x))
        x = self.flatten(x)
        x = self.fc(x)
        # 分类是否人脸的卷积输出层
        class_out = self.class_fc(x)
        # 人脸box的回归卷积输出层
        bbox_out = self.bbox_fc(x)
        # 5个关键点的回归卷积输出层
        landmark_out = self.landmark_fc(x)
        return class_out, bbox_out, landmark_out

class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))
        self.prelu3 = nn.PReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2))
        self.prelu4 = nn.PReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=1152, out_features=256)
        self.class_fc = nn.Linear(in_features=256, out_features=2)
        self.bbox_fc = nn.Linear(in_features=256, out_features=4)
        self.landmark_fc = nn.Linear(in_features=256, out_features=10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.pool2(x)
        x = self.prelu3(self.conv3(x))
        x = self.pool3(x)
        x = self.prelu4(self.conv4(x))
        x = self.flatten(x)
        x = self.fc(x)
        # 分类是否人脸的卷积输出层
        class_out = self.class_fc(x)
        # 人脸box的回归卷积输出层
        bbox_out = self.bbox_fc(x)
        # 5个关键点的回归卷积输出层
        landmark_out = self.landmark_fc(x)
        return class_out, bbox_out, landmark_out




class ClassLoss(nn.Module):
    def __init__(self):
        super(ClassLoss, self).__init__()
        self.entropy_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        self.keep_ratio = 0.7

    def forward(self, class_out, label):
        # 保留neg 0 和pos 1 的数据，忽略掉part -1, landmark -2
        label = torch.where(label < 0, -100, label)
        # 求neg 0 和pos 1 的数据70%数据
        valid_label = torch.where(label >= 0, 1, 0)
        num_valid = torch.sum(valid_label)
        keep_num = int((num_valid * self.keep_ratio).cpu().numpy())
        label = torch.squeeze(label)
        # 计算交叉熵损失
        loss = self.entropy_loss(input=class_out, target=label)
        # 取有效数据的70%计算损失
        loss, _ = torch.topk(torch.squeeze(loss), k=keep_num)
        return torch.mean(loss)


class BBoxLoss(nn.Module):
    def __init__(self):
        super(BBoxLoss, self).__init__()
        self.square_loss = nn.MSELoss(reduction='none')
        self.keep_ratio = 1.0

    def forward(self, bbox_out, bbox_target, label):
        # 保留pos 1 和part -1 的数据
        valid_label = torch.where(torch.abs(label) == 1, 1, 0)
        valid_label = torch.squeeze(valid_label)
        # 获取有效值的总数
        keep_num = int(torch.sum(valid_label).cpu().numpy() * self.keep_ratio)
        loss = self.square_loss(input=bbox_out, target=bbox_target)
        loss = torch.sum(loss, dim=1)
        loss = loss.cuda() * valid_label
        # 取有效数据计算损失
        loss, _ = torch.topk(loss, k=keep_num, dim=0)
        return torch.mean(loss)


class LandmarkLoss(nn.Module):
    def __init__(self):
        super(LandmarkLoss, self).__init__()
        self.square_loss = nn.MSELoss(reduction='none')
        self.keep_ratio = 1.0

    def forward(self, landmark_out, landmark_target, label):
        # 只保留landmark数据 -2
        valid_label = torch.where(label == -2, 1, 0)
        valid_label = torch.squeeze(valid_label)
        # 获取有效值的总数
        keep_num = int(torch.sum(valid_label).cpu().numpy() * self.keep_ratio)
        loss = self.square_loss(input=landmark_out, target=landmark_target)
        loss = torch.sum(loss, dim=1)
        loss = loss.cuda() * valid_label
        # 取有效数据计算损失
        loss, _ = torch.topk(loss, k=keep_num, dim=0)
        return torch.mean(loss)


# 求训练时的准确率
def accuracy(class_out, label):
    # 查找neg 0 和pos 1所在的位置
    class_out = class_out.detach().cpu().numpy()
    label = label.cpu().numpy()
    label = np.squeeze(label)
    zeros = np.zeros(label.shape)
    cond = np.greater_equal(label, zeros)
    picked = np.where(cond)
    valid_label = label[picked]
    valid_class_out = class_out[picked]
    # 求neg 0 和pos 1的准确率
    acc = np.sum(np.argmax(valid_class_out, axis=1) == valid_label, dtype='float')
    acc = acc / valid_label.shape[0]
    return acc