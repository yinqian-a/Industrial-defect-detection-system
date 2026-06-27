# main.py
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

print("=" * 50)
print(f"Python版本: {__import__('sys').version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print("=" * 50)


# 1. 自定义数据集类
class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.image_paths = []

        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.PNG', '*.JPG']:
            self.image_paths.extend(self.root.glob(f'**/{ext}'))

        print(f"找到 {len(self.image_paths)} 张图片在 {root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, str(img_path)


# 2. 数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("正在加载数据集...")
train_dataset = ImageFolderDataset(
    root="./datasets/MVTecAD/bottle/train/good",
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

print(f"训练集大小: {len(train_dataset)} 张图片")
print("数据集加载完成！")

# 3. 加载预训练模型（使用 timm）
import timm

print("加载预训练模型...")
backbone = timm.create_model('wide_resnet50_2', pretrained=True, features_only=True)
backbone.eval()
print("模型加载完成！")


# 4. 特征提取函数
def extract_features(backbone, dataloader, layers=['layer2', 'layer3']):
    """提取图像特征"""
    features = {layer: [] for layer in layers}

    with torch.no_grad():
        for images, _ in dataloader:
            # 提取特征
            outputs = backbone(images)

            # 只取需要的层
            for layer in layers:
                if layer == 'layer2':
                    feat = outputs[2]  # layer2
                elif layer == 'layer3':
                    feat = outputs[3]  # layer3
                else:
                    continue

                # 调整特征形状：BCHW -> (B*H*W, C)
                b, c, h, w = feat.shape
                feat = feat.permute(0, 2, 3, 1).reshape(-1, c)
                features[layer].append(feat)

    # 合并所有批次
    for layer in layers:
        features[layer] = torch.cat(features[layer], dim=0)

    return features


print("开始提取特征...")
train_features = extract_features(backbone, train_loader)
print(f"特征提取完成！layer2: {train_features['layer2'].shape}, layer3: {train_features['layer3'].shape}")


# 5. 核心集采样（快速随机采样）
def coreset_sampling(features, ratio=0.1):
    """从特征集中快速随机采样核心集"""
    n_samples = int(len(features) * ratio)
    if n_samples >= len(features):
        return features

    # 随机采样
    indices = torch.randperm(len(features))[:n_samples]
    return features[indices]


print("进行核心集采样...")
for layer in train_features:
    train_features[layer] = coreset_sampling(train_features[layer], ratio=0.1)
    print(f"{layer} 采样后: {train_features[layer].shape}")

# 6. 构建特征库（用于推理）
# 方案1：分别存储不同层的特征
feature_bank = {
    'layer2': train_features['layer2'],
    'layer3': train_features['layer3']
}
print(f"特征库构建完成！layer2: {feature_bank['layer2'].shape}, layer3: {feature_bank['layer3'].shape}")

# 7. 推理测试
print("开始测试推理...")
test_dataset = ImageFolderDataset(
    root="./datasets/MVTecAD/bottle/test",
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# 分别计算两个层的异常分数
from sklearn.neighbors import NearestNeighbors

# 为每个层训练最近邻模型
nn_models = {}
for layer in ['layer2', 'layer3']:
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nn.fit(feature_bank[layer].numpy())
    nn_models[layer] = nn

anomaly_scores = []
image_paths = []

with torch.no_grad():
    for images, paths in test_loader:
        # 提取测试图像特征
        outputs = backbone(images)

        # 提取 layer2 特征
        feat_layer2 = outputs[2].permute(0, 2, 3, 1).reshape(-1, outputs[2].shape[1])

        # 提取 layer3 特征
        feat_layer3 = outputs[3].permute(0, 2, 3, 1).reshape(-1, outputs[3].shape[1])

        # 分别计算距离
        dist2, _ = nn_models['layer2'].kneighbors(feat_layer2.numpy())
        dist3, _ = nn_models['layer3'].kneighbors(feat_layer3.numpy())

        # 综合异常分数（取平均或最大值）
        anomaly_score = (np.mean(dist2) + np.mean(dist3)) / 2  # 或者用 np.maximum

        anomaly_scores.append(anomaly_score)
        image_paths.append(paths[0])

# 8. 显示结果
print("\n测试结果（前10张）:")
for i in range(min(10, len(anomaly_scores))):
    path = Path(image_paths[i])
    is_defective = "缺陷" if "good" not in str(path).lower() else "良品"
    print(f"图片 {i + 1}: {path.name}")
    print(f"  类型: {is_defective}")
    print(f"  异常分数: {anomaly_scores[i]:.4f}")
    print()

print("=" * 50)
print("项目运行完成！")