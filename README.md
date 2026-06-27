
# 🧠 PatchCore 工业缺陷检测系统

基于 **PatchCore** 算法的无监督工业缺陷检测系统。仅使用良品图像训练，即可检测出产品表面的各类异常（划痕、破损、污染等）。

## 📋 项目简介

本项目实现了 PatchCore 论文的核心算法，包括：
- 使用预训练的 WideResNet50 提取图像特征
- 核心集采样（Coreset Sampling）压缩特征库
- 基于最近邻搜索的异常分数计算
- 支持自定义数据集训练和推理

## 🚀 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.0+

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行

**完整版（使用全部数据集）：**
```bash
python main-1.py
```

**演示版（4张图片快速演示）：**
```bash
python main-2.py
```

## 📁 项目结构

```
PyPatchCore/
├── main-1.py              # 完整版：全量数据集训练和测试
├── main-2.py              # 演示版：4张图片快速演示
├── requirements.txt       # 项目依赖
├── README.md              # 项目说明
├── datasets/              # 数据集目录（需自行下载）
│   └── MVTecAD/
│       └── bottle/
│           ├── train/
│           │   └── good/      # 良品训练图片
│           └── test/
│               ├── good/      # 良品测试图片
│               └── broken_large/  # 缺陷测试图片
└── sample_data/           # 示例数据（4张图片）
    ├── train/
    │   └── good/              # 训练用良品图片
    └── test/
        ├── good_xxx.png       # 测试用良品图片
        └── defect_xxx.png     # 测试用缺陷图片
```

## 📊 数据集

本项目使用 **MVTec AD** 数据集进行验证。

**下载方式：**
1. 官网下载（需填写表单）：https://www.mvtec.com/company/research/datasets/mvtec-ad
2. 国内镜像：https://beta.hyper.ai/cn/datasets/9066

下载后解压到 `./datasets/MVTecAD/` 目录。

## 🎯 核心原理

### 1. 特征提取
使用在 ImageNet 上预训练的 WideResNet50 作为骨干网络，提取图像的 `layer2` 和 `layer3` 特征图，并将特征图展平为 Patch 特征向量。

### 2. 核心集采样
对提取到的海量 Patch 特征进行随机采样（默认 10%），在保证检测精度的同时大幅降低内存占用和推理时间。

### 3. 异常检测
对测试图片的每个 Patch 特征，在特征库中寻找最近邻，计算欧氏距离。取所有 Patch 距离的平均值作为该图片的异常分数。

## 📈 输出示例

```bash
📋 测试结果（全部4张）:
  1. good_000.png
     标签: ✅ 良品
     异常分数: 18.2345
  2. good_001.png
     标签: ✅ 良品
     异常分数: 19.8765
  3. defect_000.png
     标签: ❌ 缺陷
     异常分数: 28.4321
  4. defect_001.png
     标签: ❌ 缺陷
     异常分数: 31.5678

📊 统计对比:
  良品平均异常分数: 19.0555
  缺陷平均异常分数: 29.9999
  ✅ 缺陷图片分数普遍高于良品，模型有效！
```

## 🛠️ 技术栈

| 库 | 用途 |
|---|---|
| PyTorch | 深度学习框架 |
| timm | 预训练模型库 |
| scikit-learn | 最近邻搜索 |
| Pillow | 图像处理 |
| NumPy | 数值计算 |

## 🔧 后续优化方向

- [ ] 使用 KCenterGreedy 替代随机采样，提升检测精度
- [ ] 添加异常热力图可视化，定位缺陷位置
- [ ] 支持 ONNX/TensorRT 部署，满足实时推理需求
- [ ] 封装为 FastAPI 服务，提供 HTTP 接口

## 📝 License

本项目仅供学习和研究使用。

## 👤 作者

- GitHub: [yinqian-a](https://github.com/yinqian-a)

## 🙏 参考

- [PatchCore 论文](https://arxiv.org/abs/2106.08265)
- [Anomalib 官方库](https://github.com/openvinotoolkit/anomalib)
- [MVTec AD 数据集](https://www.mvtec.com/company/research/datasets/mvtec-ad)
