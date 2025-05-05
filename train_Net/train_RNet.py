import os
import sys
from datetime import datetime
sys.path.append("../")
import numpy as np
from utils.utils import plot_metrics
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torchsummary import summary
from torch.utils.data import DataLoader
from config import detcfg
from model import ClassLoss, BBoxLoss, LandmarkLoss, accuracy, RNet
from utils.data import CustomDataset

# 设置损失值的比例
radio_cls_loss = 1.0
radio_bbox_loss = 0.5
radio_landmark_loss = 0.5

# 训练参数值
data_path = os.path.join(detcfg.dataset_dir, '24', 'all_data')
batch_size = 384
learning_rate = 1e-3
epoch_num = 22
model_path = detcfg.model_dir

# 获取R模型
device = detcfg.device
model = RNet()
model.to(device)
summary(model, (3, 24, 24))

# 获取数据
train_dataset = CustomDataset(data_path)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 设置优化方法
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=1e-4)

# 获取学习率衰减函数
scheduler = MultiStepLR(optimizer, milestones=[6, 14, 20], gamma=0.1)

# 获取损失函数
class_loss = ClassLoss()
bbox_loss = BBoxLoss()
landmark_loss = LandmarkLoss()
log_dict = {
    'epoch': [],
    'cls_loss': [],
    'box_loss': [],
    'landmark_loss': [],
    'total_loss': [],
    'acc': []
}
# 开始训练
for epoch in range(epoch_num):
    epoch_cls, epoch_box = [], []
    epoch_landmark, epoch_total = [], []
    epoch_acc = []
    for batch_id, (img, label, bbox, landmark) in enumerate(train_loader):
        img = img.to(device)
        label = label.to(device).long()
        bbox = bbox.to(device)
        landmark = landmark.to(device)
        class_out, bbox_out, landmark_out = model(img)
        cls_loss = class_loss(class_out, label)
        box_loss = bbox_loss(bbox_out, bbox, label)
        landmarks_loss = landmark_loss(landmark_out, landmark, label)
        total_loss = radio_cls_loss * cls_loss + radio_bbox_loss * box_loss + radio_landmark_loss * landmarks_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if batch_id % 100 == 0:
            acc = accuracy(class_out, label)
            print('[%s] Train epoch %d, batch %d, total_loss: %f, cls_loss: %f, box_loss: %f, landmarks_loss: %f, '
                  'accuracy：%f' % (
                  datetime.now(), epoch, batch_id, total_loss, cls_loss, box_loss, landmarks_loss, acc))
        # 收集batch指标
        epoch_cls.append(cls_loss.item())
        epoch_box.append(box_loss.item())
        epoch_landmark.append(landmarks_loss.item())
        epoch_total.append(total_loss.item())
        epoch_acc.append(acc.item())
    scheduler.step()

    # 计算epoch平均值
    log_dict['epoch'].append(epoch)
    log_dict['cls_loss'].append(np.mean(epoch_cls))
    log_dict['box_loss'].append(np.mean(epoch_box))
    log_dict['landmark_loss'].append(np.mean(epoch_landmark))
    log_dict['total_loss'].append(np.mean(epoch_total))
    log_dict['acc'].append(np.mean(epoch_acc))

    # 保存模型
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.jit.save(torch.jit.script(model), detcfg.rnet_weight)
plot_metrics(log_dict, detcfg.img_dir)
