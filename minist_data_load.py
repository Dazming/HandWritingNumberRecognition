"""
MNIST 手写数字数据集加载模块

本模块用于加载MNIST手写数字数据集，这是深度学习中最经典的数据集之一。
MNIST数据集包含70000张手写数字图片（0-9），其中60000张用于训练，10000张用于测试。
"""

# ==================== 导入必要的库 ====================

import torch                          # PyTorch核心库，用于构建和训练神经网络
import torchvision.datasets as dataset # torchvision提供常用数据集，这里用dataset模块
import torchvision.transforms as transforms # 图像预处理工具
import torch.utils.data as utils      # 数据加载工具，提供DataLoader功能

# ==================== 加载MNIST训练数据集 ====================

# dataset.MNIST: PyTorch内置的MNIST数据集类
# 参数说明：
#   root='./data/': 数据集下载/存放的目录路径，'./'表示当前目录
#   train=True: True表示加载训练集（60000张），False表示测试集（10000张）
#   transform=transforms.ToTensor(): 将PIL图片转换为PyTorch张量（Tensor）
#     - 同时会自动将像素值从0-255归一化到0-1范围
#   download=True: 如果本地没有数据集，自动从网络下载
train_dataset = dataset.MNIST(
    root='./data/',              # 数据存放文件夹（会自动创建）
    train=True,                  # True=训练集（60000张图片），False=测试集（10000张）
    transform=transforms.ToTensor(),  # 图片格式转换：PIL Image → PyTorch Tensor
    download=True                # 本地没有时自动下载MNIST数据集
)

# ==================== 加载MNIST测试数据集 ====================

# 测试集与训练集格式相同，但train=False
# 测试集用于评估模型性能，是模型未见过的数据
test_dataset = dataset.MNIST(
    root='./data/',               # 与训练集共用同一个数据目录
    train=False,                  # False=测试集（10000张图片）
    transform=transforms.ToTensor(),
    download=True
)

# ==================== 创建训练数据加载器 ====================

# DataLoader是PyTorch的数据迭代器，它负责：
#   1. 分批次（batch）加载数据
#   2. 打乱数据顺序（shuffle）以提高训练效果
#   3. 支持多进程加载加速
train_loader = utils.DataLoader(
    dataset=train_dataset,        # 要加载的数据集（训练集）
    batch_size=64,                # 每批加载64张图片，训练时一批一批地喂给模型
    shuffle=True                  # True=随机打乱数据顺序，防止模型记忆数据顺序
)

# ==================== 创建测试数据加载器 ====================

# 测试时通常不需要打乱数据顺序，因为我们需要稳定的评估结果
test_loader = utils.DataLoader(
    dataset=test_dataset,          # 要加载的数据集（测试集）
    batch_size=64,                 # 每批64张图片
    shuffle=False                  # False=不打乱，保持数据顺序一致，便于分析错误样本
)

# ==================== 使用说明 ====================
#
# 训练时这样使用：
#   for images, labels in train_loader:
#       # images: [64, 1, 28, 28] 的张量，64张28x28的单通道图片
#       # labels: [64] 的张量，对应每张图片的数字标签(0-9)
#       # 在这里进行模型前向传播、计算损失、反向传播等操作
#
# 测试时这样使用：
#   for images, labels in test_loader:
#       # 同样获取一批图片和标签
#       # 在这里进行模型评估，计算准确率等指标
