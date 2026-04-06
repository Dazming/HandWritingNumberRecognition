"""
MNIST 手写数字识别 - 训练脚本

本脚本用于训练 CNN 模型识别手写数字 0-9。
训练完成后会自动保存模型到 mnist_model.pkl 文件。

【什么是 MNIST 数据集？】
- 共 70000 张手写数字图片（0-9）
- 训练集：60000 张
- 测试集：10000 张
- 图片尺寸：28×28 像素，灰度图
"""

# ==================== 第一部分：导入必要的库 ====================
# 相当于 Python 的"工具箱"，提前准备好要用的函数和类

import torch  # PyTorch 核心库，提供张量和神经网络功能

# torchvision.datasets: 包含常用数据集（包括 MNIST）
import torchvision.datasets as dataset

# torchvision.transforms: 图像预处理工具
# 可以对图片进行缩放、裁剪、归一化等操作
import torchvision.transforms as transforms

# torch.utils.data: 数据加载工具
# DataLoader 可以分批次、随机打乱、多进程加载数据
import torch.utils.data as utils

# 从 model.py 文件导入 Model 类（自己定义的神经网络）
from model import Model


# ==================== 第二部分：加载 MNIST 数据集 ====================

# ---------- 加载训练集 ----------
# dataset.MNIST: PyTorch 内置的 MNIST 数据集类
train_dataset = dataset.MNIST(
    root='./data/',              # 数据存放的文件夹路径
                                # './' 表示当前目录，会在当前目录下创建 data 文件夹
                                # 如果目录不存在会自动创建

    train=True,                  # True = 加载训练集（60000 张图片）
                                # False = 加载测试集（10000 张图片）

    transform=transforms.ToTensor(),  # 数据转换：将 PIL 图片转换为 PyTorch 张量
                                # transforms.ToTensor() 会：
                                # 1. 将图片从 PIL 格式转换为张量 (28, 28) → (1, 28, 28)
                                # 2. 将像素值从 0-255 归一化到 0-1 范围

    download=True                # 如果本地没有数据集，自动从网络下载
                                # 首次运行时会下载约 11MB 的数据
)

# ---------- 加载测试集 ----------
# 测试集用于评估模型性能，是模型在训练时没有见过的数据
test_dataset = dataset.MNIST(
    root='./data/',              # 与训练集共用同一个文件夹
    train=False,                  # False = 加载测试集（10000 张图片）
    transform=transforms.ToTensor(),
    download=True
)


# ==================== 第三部分：创建数据加载器 ====================

"""
为什么要用 DataLoader？
- 数据集可能很大，无法一次性全部加载到内存
- 需要分批次（batch）训练，每批处理64张图片
- 打乱顺序可以让模型学到更泛化的特征
- 支持多进程加速数据加载

【数据形状说明】
- train_loader 中的每个 batch：
    images: torch.Size([64, 1, 28, 28])   # 64张图片，1通道，28×28像素
    labels: torch.Size([64])              # 64个标签，对应 0-9 的数字
"""

# ---------- 创建训练数据加载器 ----------
train_loader = utils.DataLoader(
    dataset=train_dataset,        # 要加载的数据集（训练集）
    batch_size=64,                # 每批加载 64 张图片
                                # 60000 / 64 ≈ 938 批

    shuffle=True                  # True = 随机打乱数据顺序
                                # 作用：防止模型记忆数据的顺序规律，提高泛化能力
)

# ---------- 创建测试数据加载器 ----------
test_loader = utils.DataLoader(
    dataset=test_dataset,          # 要加载的数据集（测试集）
    batch_size=64,                 # 每批 64 张图片
    shuffle=False                  # False = 不打乱数据顺序
                                # 原因：测试时需要稳定可复现的结果
                                #       也便于分析哪些数据预测错误
)


# ==================== 第四部分：初始化模型 ====================

# 创建模型实例
cnn = Model()

# 将模型移动到 GPU 上
# .cuda() 会将模型的所有参数和缓冲区复制到 GPU 显存中
# 如果没有 GPU，会报错；可以用 .cpu() 强制使用 CPU
cnn = cnn.cuda()


# ==================== 第五部分：定义损失函数和优化器 ====================

# ---------- 损失函数 ----------
# torch.nn.CrossEntropyLoss: 交叉熵损失函数
# 这是分类问题最常用的损失函数
# 它会自动做 Softmax，所以网络输出不需要额外做 Softmax
loss_func = torch.nn.CrossEntropyLoss()

# ---------- 优化器 ----------
# torch.optim.Adam: Adam 优化器
# Adam 是最常用的优化器之一，自动调整学习率，收敛速度快
optimizer = torch.optim.Adam(
    cnn.parameters(),            # 告诉优化器要优化哪些参数（模型的所有参数）
    lr=0.01                       # learning rate，学习率，控制参数更新的步长
                                # 学习率太大训练不稳定，太小收敛太慢
)


# ==================== 第六部分：训练循环 ====================

"""
训练的核心流程（每个 batch 执行一次）：
1. 前向传播：把图片输入模型，得到预测结果
2. 计算损失：比较预测结果和真实标签，计算误差
3. 反向传播：计算每个参数对损失的梯度
4. 参数更新：用梯度更新模型参数

【专业术语解释】
- Epoch（轮）：看完整 个训练数据集一次
- Batch（批次）：一次处理的多张图片
- Batch Size：每批几张图片
- Iteration（迭代）：一个 batch 就是一次迭代
"""

for epoch in range(10):           # 外层循环：训练 10 轮（10 个 epoch）
                                # 一个 epoch = 看完整 个训练数据集一次
                                # 60000 张图片，batch_size=64，所以每轮有 938 个 batch

    # ---------- 训练阶段 ----------
    for index, (images, labels) in enumerate(train_loader):
        # enumerate() 会返回：(batch编号, (图片, 标签))
        # index: 当前是第几个 batch，从 0 开始
        # images: 当前 batch 的图片，形状 [64, 1, 28, 28]
        # labels: 当前 batch 的标签，形状 [64]

        # 将数据移动到 GPU 上
        images = images.cuda()     # 图片张量移到 GPU
        labels = labels.cuda()   # 标签张量移到 GPU

        # 1. 前向传播：把图片输入模型，得到预测结果
        outputs = cnn(images)    # outputs 形状: [64, 10]
                                # 每张图片输出 10 个值，分别代表 0-9 的得分

        # 2. 计算损失：比较预测结果和真实标签
        loss = loss_func(outputs, labels)
                                # loss 是一个标量，表示当前 batch 的预测误差

        # 3. 反向传播：计算梯度（准备好优化器需要的信息）
        optimizer.zero_grad()    # 清零之前的梯度（否则会累加）
        loss.backward()          # 自动计算每个参数对 loss 的梯度

        # 4. 参数更新：用梯度更新模型参数
        optimizer.step()         # 根据梯度更新参数，降低 loss

        # 打印训练进度
        # index + 1 是因为 index 从 0 开始，需要加 1
        print(f"epoch: {epoch+1}  batch:{index + 1}/{len(train_loader)}   loss:{loss.item():.4f}")

    # ---------- 测试阶段 ----------
    # 每轮训练结束后，用测试集评估模型性能
    loss_test = 0                 # 累计测试损失
    right_count = 0               # 累计正确预测的数量

    for index, (images, labels) in enumerate(test_loader):
        # 同样将数据移到 GPU
        images = images.cuda()
        labels = labels.cuda()

        # 用模型进行预测
        outputs = cnn(images)

        # 计算测试损失（累计）
        loss_test += loss_func(outputs, labels).item()

        # 获取预测结果
        # outputs.max(1) 返回每行（每张图片）最大值的索引
        # 索引就是预测的数字（0-9）
        _, pred = outputs.max(1)  # pred 形状: [64]

        # 统计正确预测的数量
        # (pred == labels) 生成布尔张量
        # .sum().item() 统计 True 的数量
        right_count += (pred == labels).sum().item()

    # 计算并打印准确率
    # right_count / len(test_dataset) 得到正确率（0-1 之间）
    # * 100 转换为百分比
    acc = right_count / len(test_dataset) * 100
    print(f"epoch: {epoch+1}  test acc:{acc:.2f}%  loss:{loss_test:.4f}")
    print("-" * 50)  # 分隔线，便于查看


# ==================== 第七部分：保存模型 ====================

# 训练完成后，将模型保存到文件
# 保存后，下次可以直接加载模型进行预测，无需重新训练
torch.save(cnn, 'mnist_model.pkl')
print("模型已保存到 mnist_model.pkl")