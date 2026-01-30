import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import warnings
import json
from datetime import datetime
import pandas as pd
from collections import defaultdict

# 忽略警告
warnings.filterwarnings("ignore")
#  1. 数据集定义
class COCOSubsetDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root = root_dir
        self.img_dir = os.path.join(root_dir, f'{split}2017', 'images')
        self.ann_file = os.path.join(root_dir, f'{split}2017', f'instances_{split}2017.json')
        # 加载COCO标注
        self.coco = COCO(self.ann_file)
        self.image_ids = self.coco.getImgIds()
        # 过滤有效图片
        self.valid_image_ids = []
        for img_id in self.image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) > 0:
                self.valid_image_ids.append(img_id)
        self.image_ids = self.valid_image_ids
        print(f"{split}集有效样本数: {len(self.image_ids)}")
    def __len__(self):
        return len(self.image_ids)
    def __getitem__(self, idx):
        # 加载图片
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img = cv2.imread(os.path.join(self.img_dir, img_info['file_name']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]
        # 加载标注
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes, labels = [], []
        for ann in anns:
            x1, y1, w, h = ann['bbox']
            if w > 0 and h > 0:
                x2, y2 = x1 + w, y1 + h
                # 边界裁剪
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(ann['category_id'])
        # 转换为tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            'iscrowd': torch.zeros_like(labels)
        }
        # 图片归一化
        img = torch.as_tensor(img.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        return img, target
#  2. 改进模块：CBAM注意力
class CBAMBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 通道注意力
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 16),
            nn.ReLU(),
            nn.Linear(channels // 16, channels)
        )
        # 空间注意力
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # 通道注意力
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        max_pool = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)
        channel_att = self.sigmoid(self.fc(avg_pool) + self.fc(max_pool)).view(x.size(0), -1, 1, 1)
        x = x * channel_att
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        spatial_att = self.sigmoid(self.spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        return x
# 3. 模型构建
def build_model(num_classes=91, use_attention=False):
    """构建Faster R-CNN，可选添加CBAM注意力"""
    # 加载预训练模型
    try:
        from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    except:
        model = fasterrcnn_resnet50_fpn(pretrained=True)
    # 替换分类头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # 添加注意力机制
    if use_attention:
        # 为FPN输出添加CBAM
        for name, module in model.backbone.named_modules():
            if 'fpn' in name and 'conv' in name and module.out_channels == 256:
                setattr(model.backbone, name, nn.Sequential(module, CBAMBlock(256)))
    return model
# 4. 训练/验证函数
def train_model(model, train_loader, val_loader, config, exp_name):
    device = config['device']
    model.to(device)
    # 优化器和调度器
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)
    writer = SummaryWriter(f'runs/{exp_name}')
    # 损失记录
    loss_history = {
        'train': [],
        'val': [],
        'train_components': defaultdict(list),
        'val_components': defaultdict(list)
    }
    best_val_loss = float('inf')
    for epoch in range(config['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Train Epoch {epoch + 1}')
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # 前向传播
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            # 反向传播
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            # 记录损失
            train_loss += losses.item()
            pbar.set_postfix({'loss': f'{losses.item():.4f}'})
            # 记录损失组件
            for k, v in loss_dict.items():
                loss_history['train_components'][k].append(v.item())
        # 验证阶段
        model.train()  # 保持train模式计算损失
        val_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Val Epoch {epoch + 1}')
            for images, targets in pbar:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                # 记录损失组件
                for k, v in loss_dict.items():
                    loss_history['val_components'][k].append(v.item())
        # 计算平均损失
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        loss_history['train'].append(avg_train_loss)
        loss_history['val'].append(avg_val_loss)
        # 学习率调整
        scheduler.step(avg_val_loss)
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'{exp_name}_best.pth')
        # 打印结果
        print(
            f'Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Best Val Loss={best_val_loss:.4f}')
        # TensorBoard记录
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
    writer.close()
    return loss_history, best_val_loss
# 5. 可视化函数
def plot_results(all_results):
    # 1. 损失曲线对比
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red']
    markers = ['o', 's']
    for i, (exp_name, loss_hist, best_loss) in enumerate(all_results):
        epochs = range(1, len(loss_hist['train']) + 1)
        plt.plot(epochs, loss_hist['train'], color=colors[i], marker=markers[i],
                 label=f'{exp_name} (Train)')
        plt.plot(epochs, loss_hist['val'], color=colors[i], marker=markers[i], linestyle='--',
                 label=f'{exp_name} (Val)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('loss_curves.png')
    plt.show()
    # 2. 最佳损失对比
    plt.figure(figsize=(8, 5))
    exp_names = [r[0] for r in all_results]
    best_losses = [r[2] for r in all_results]
    plt.bar(exp_names, best_losses, color=['blue', 'red'], alpha=0.7)
    plt.xlabel('Experiment')
    plt.ylabel('Best Validation Loss')
    plt.title('Best Validation Loss Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    # 添加数值标签
    for i, v in enumerate(best_losses):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    plt.savefig('best_loss_comparison.png')
    plt.show()
    # 3. 生成对比表格
    data = {
        'Experiment': [r[0] for r in all_results],
        'Best Val Loss': [f'{r[2]:.4f}' for r in all_results],
        'Final Train Loss': [f'{r[1]["train"][-1]:.4f}' for r in all_results],
        'Final Val Loss': [f'{r[1]["val"][-1]:.4f}' for r in all_results]
    }
    df = pd.DataFrame(data)
    print("\n实验结果对比:")
    print(df)
    df.to_csv('experiment_results.csv', index=False)
# 6. 主程序
if __name__ == "__main__":
    # 基础配置
    CONFIG = {
        'data_root': r"D:\.learning\git\COCO2017_panopic_subset",
        'num_classes': 91,
        'batch_size': 1,
        'epochs': 5,  # 减少训练轮数加速实验
        'lr': 0.005,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    print("Faster R-CNN 消融实验 ")
    print(f"使用设备: {CONFIG['device']}")
    # 加载数据集
    print("\n加载数据集...")
    train_dataset = COCOSubsetDataset(CONFIG['data_root'], 'train')
    val_dataset = COCOSubsetDataset(CONFIG['data_root'], 'val')
    # 数据加载器
    def collate_fn(batch):
        return tuple(zip(*batch))
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=False, collate_fn=collate_fn, num_workers=0)
    # 运行消融实验
    all_results = []
    # 实验1: 基准模型
    print("实验1: 基准模型 (Baseline)")
    model = build_model(CONFIG['num_classes'], use_attention=False)
    loss_hist, best_loss = train_model(model, train_loader, val_loader, CONFIG, 'Baseline')
    all_results.append(('Baseline', loss_hist, best_loss))
    # 实验2: 添加CBAM注意力的模型
    print("实验2: CBAM注意力模型")
    model = build_model(CONFIG['num_classes'], use_attention=True)
    loss_hist, best_loss = train_model(model, train_loader, val_loader, CONFIG, 'CBAM_Attention')
    all_results.append(('CBAM_Attention', loss_hist, best_loss))
    # 可视化结果
    print("实验完成！生成可视化结果...")
    plot_results(all_results)
    # 计算改进效果
    baseline_loss = all_results[0][2]
    improved_loss = all_results[1][2]
    improvement = (baseline_loss - improved_loss) / baseline_loss * 100
    print(f"\n改进效果分析:")
    print(f"基准模型最佳验证损失: {baseline_loss:.4f}")
    print(f"CBAM模型最佳验证损失: {improved_loss:.4f}")
    print(f"损失降低幅度: {improvement:.2f}%")
    if improvement > 0:
        print("✓ CBAM注意力机制有效提升了模型性能！")
    else:
        print("→ CBAM注意力机制未带来明显提升 (可能需要更多训练轮数)")
