# main-2.py
# 简化版：使用少量图片快速演示异常检测
# 适合快速验证、演示、调试

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

print("=" * 50)
print("🚀 PatchCore 异常检测演示版（4张图片）")
print(f"Python版本: {__import__('sys').version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print("=" * 50)


# ============================================================
# 1. 自定义数据集类
# ============================================================
class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.image_paths = []

        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.PNG', '*.JPG']:
            self.image_paths.extend(self.root.glob(f'**/{ext}'))

        print(f"  找到 {len(self.image_paths)} 张图片")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, str(img_path)


# ============================================================
# 2. 数据预处理
# ============================================================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ============================================================
# 3. 核心集采样
# ============================================================
def coreset_sampling(features, ratio=0.1):
    """从特征集中快速随机采样核心集"""
    n_samples = int(len(features) * ratio)
    if n_samples >= len(features):
        return features
    indices = torch.randperm(len(features))[:max(1, n_samples)]
    return features[indices]


# ============================================================
# 4. 提取特征
# ============================================================
def extract_features(backbone, dataloader, layers=['layer2', 'layer3']):
    """提取图像特征"""
    features = {layer: [] for layer in layers}

    with torch.no_grad():
        for images, _ in dataloader:
            outputs = backbone(images)

            for layer in layers:
                if layer == 'layer2':
                    feat = outputs[2]
                elif layer == 'layer3':
                    feat = outputs[3]
                else:
                    continue

                b, c, h, w = feat.shape
                feat = feat.permute(0, 2, 3, 1).reshape(-1, c)
                features[layer].append(feat)

    for layer in layers:
        features[layer] = torch.cat(features[layer], dim=0)

    return features


# ============================================================
# 5. 创建样本数据（4张图片）
# ============================================================
print("\n📁 创建样本数据...")

# 创建临时目录
sample_dir = Path("./sample_data")
sample_dir.mkdir(exist_ok=True)

# 从完整数据集中挑选图片
import shutil

# 训练集：使用原数据集中的良品图片（至少需要一些）
train_good_dir = Path("./datasets/MVTecAD/bottle/train/good")
if not train_good_dir.exists():
    print("❌ 未找到完整数据集，请先确保 datasets 目录存在")
    print("   路径: ./datasets/MVTecAD/bottle/train/good")
    exit(1)

# 创建良品训练目录
sample_train_dir = sample_dir / "train" / "good"
sample_train_dir.mkdir(parents=True, exist_ok=True)

# 复制前10张良品图片作为训练集
train_images = sorted(train_good_dir.glob("*.png"))[:10]
for img in train_images:
    shutil.copy(img, sample_train_dir / img.name)
print(f"  训练集: {len(train_images)} 张良品图片")

# 创建测试目录（包含良品和缺陷各2张）
sample_test_dir = sample_dir / "test"
sample_test_dir.mkdir(exist_ok=True)

# 从原测试集中复制图片
test_good_dir = Path("./datasets/MVTecAD/bottle/test/good")
test_defect_dir = Path("./datasets/MVTecAD/bottle/test/broken_large")

if test_good_dir.exists() and test_defect_dir.exists():
    # 复制2张良品
    good_images = sorted(test_good_dir.glob("*.png"))[:2]
    for img in good_images:
        shutil.copy(img, sample_test_dir / f"good_{img.name}")

    # 复制2张缺陷
    defect_images = sorted(test_defect_dir.glob("*.png"))[:2]
    for img in defect_images:
        shutil.copy(img, sample_test_dir / f"defect_{img.name}")

    print(f"  测试集: {len(good_images)} 张良品 + {len(defect_images)} 张缺陷")
else:
    # 如果原数据集不存在，生成4张随机图作为演示
    print("⚠️  原数据集不完整，生成4张随机图作为演示...")
    import numpy as np

    for i in range(4):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        label = "good" if i < 2 else "defect"
        Image.fromarray(img).save(sample_test_dir / f"{label}_{i}.png")
    print("  已生成4张随机图片")

# ============================================================
# 6. 加载训练数据
# ============================================================
print("\n📊 加载训练数据...")
train_dataset = ImageFolderDataset(
    root=sample_train_dir,
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

print(f"  训练集大小: {len(train_dataset)} 张图片")

# ============================================================
# 7. 加载预训练模型
# ============================================================
print("\n🧠 加载预训练模型...")
import timm

backbone = timm.create_model('wide_resnet50_2', pretrained=True, features_only=True)
backbone.eval()
print("  模型加载完成！")

# ============================================================
# 8. 提取训练特征
# ============================================================
print("\n🔍 提取训练特征...")
train_features = extract_features(backbone, train_loader)

print(f"  layer2: {train_features['layer2'].shape}")
print(f"  layer3: {train_features['layer3'].shape}")

# ============================================================
# 9. 核心集采样
# ============================================================
print("\n✂️  核心集采样...")
for layer in train_features:
    original_len = len(train_features[layer])
    train_features[layer] = coreset_sampling(train_features[layer], ratio=0.5)
    print(f"  {layer}: {original_len} → {len(train_features[layer])}")

# ============================================================
# 10. 构建特征库
# ============================================================
print("\n📚 构建特征库...")
feature_bank = {
    'layer2': train_features['layer2'],
    'layer3': train_features['layer3']
}

# ============================================================
# 11. 加载测试数据（4张图片）
# ============================================================
print("\n🧪 加载测试数据...")
test_dataset = ImageFolderDataset(
    root=sample_test_dir,
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
print(f"  测试集大小: {len(test_dataset)} 张图片")

# ============================================================
# 12. 推理
# ============================================================
print("\n🔮 开始推理...")

# 训练最近邻模型
nn_models = {}
for layer in ['layer2', 'layer3']:
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nn.fit(feature_bank[layer].numpy())
    nn_models[layer] = nn

anomaly_scores = []
image_names = []

with torch.no_grad():
    for images, paths in test_loader:
        outputs = backbone(images)

        # layer2
        feat_layer2 = outputs[2].permute(0, 2, 3, 1).reshape(-1, outputs[2].shape[1])
        dist2, _ = nn_models['layer2'].kneighbors(feat_layer2.numpy())

        # layer3
        feat_layer3 = outputs[3].permute(0, 2, 3, 1).reshape(-1, outputs[3].shape[1])
        dist3, _ = nn_models['layer3'].kneighbors(feat_layer3.numpy())

        # 综合分数
        anomaly_score = (np.mean(dist2) + np.mean(dist3)) / 2
        anomaly_scores.append(anomaly_score)
        image_names.append(Path(paths[0]).name)

# ============================================================
# 13. 显示结果
# ============================================================
print("\n" + "=" * 50)
print("📋 测试结果（全部4张）:")
print("=" * 50)

for i, (name, score) in enumerate(zip(image_names, anomaly_scores)):
    # 根据文件名判断真实标签
    is_good = "good" in name.lower()
    label = "✅ 良品" if is_good else "❌ 缺陷"
    print(f"  {i + 1}. {name}")
    print(f"     标签: {label}")
    print(f"     异常分数: {score:.4f}")
    print()

# 统计对比
good_scores = [score for name, score in zip(image_names, anomaly_scores) if "good" in name.lower()]
defect_scores = [score for name, score in zip(image_names, anomaly_scores) if "good" not in name.lower()]

if good_scores and defect_scores:
    print("📊 统计对比:")
    print(f"  良品平均异常分数: {np.mean(good_scores):.4f}")
    print(f"  缺陷平均异常分数: {np.mean(defect_scores):.4f}")
    if np.mean(defect_scores) > np.mean(good_scores):
        print("  ✅ 缺陷图片分数普遍高于良品，模型有效！")
    else:
        print("  ⚠️  良品与缺陷分数区分度不明显，可能需要调整参数")

print("\n" + "=" * 50)
print("✅ 项目运行完成！")