"""
MNIST 手写数字识别模型

这个文件定义了卷积神经网络(CNN)模型，用于识别手写数字0-9。
"""

import torch  # PyTorch 核心库，提供了张量(tensor)和神经网络功能


class Model(torch.nn.Module):
    """
    CNN 模型类
    继承自 torch.nn.Module，是所有神经网络模型的基类

    网络结构：
        输入图片 (1×28×28)
            ↓
        卷积层 → 批标准化 → ReLU激活 → 最大池化
            ↓
        展平成一维向量
            ↓
        全连接层 → 输出10个数字的预测概率
    """

    def __init__(self):
        """
        初始化模型结构

        super(Model, self).__init__() 是固定写法：
        - 调用父类 torch.nn.Module 的初始化函数
        - 让 Model 类具备神经网络的基本功能
        """
        super(Model, self).__init__()

        # ==================== 卷积层部分 ====================
        # torch.nn.Sequential: 顺序容器，将多个层组合在一起
        # 数据会按顺序通过这里定义的所有层
        self.conv = torch.nn.Sequential(
            # ---------- 第1层：卷积层 ----------
            # torch.nn.Conv2d: 2D卷积层，用于提取图片特征
            # 参数说明：
            #   in_channels=1:  输入图片是灰度图，只有1个通道
            #   out_channels=32: 输出32个特征图(通道)，即提取出32种特征
            #   kernel_size=5:  卷积核大小为5×5
            #   padding=2:      在图片边缘补0，防止边缘信息丢失
            #                   补2圈后，28×28的图片变成32×32
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),

            # ---------- 第2层：批标准化 ----------
            # torch.nn.BatchNorm2d: 对每批数据进行标准化，加速训练
            # 参数32表示我们有32个特征通道
            # 作用：让数据分布在0附近，便于网络学习
            torch.nn.BatchNorm2d(32),

            # ---------- 第3层：激活函数 ----------
            # torch.nn.ReLU: 线性整流函数，公式：max(0, x)
            # 作用：给网络引入非线性，让网络能学习复杂模式
            torch.nn.ReLU(),

            # ---------- 第4层：最大池化 ----------
            # torch.nn.MaxPool2d: 最大池化，降低图片尺寸
            # 参数2表示2×2的窗口，步长也为2
            # 作用：减少计算量，提取主要特征
            # 32×32 → 16×16
            torch.nn.MaxPool2d(2),
        )

        # ==================== 全连接层部分 ====================
        # torch.nn.Linear: 全连接层，所有神经元都相互连接
        # 参数说明：
        #   in_features=14*14*32:  输入特征数 = 14×14×32 = 6272
        #                         这是因为经过池化后，32×32变成了14×14
        #   out_features=10:       输出10个值，对应数字0-9的预测概率
        self.fc = torch.nn.Linear(in_features=14*14*32, out_features=10)

    def forward(self, x):
        """
        前向传播函数，定义数据如何通过模型

        参数 x: 输入图片，形状为 [batch_size, 1, 28, 28]
              - batch_size: 每批图片的数量
              - 1: 通道数(灰度图)
              - 28×28: 图片尺寸

        返回: 形状为 [batch_size, 10] 的张量
             每行10个值，分别是对应数字0-9的预测得分( logits )
        """
        # ---------- 第1步：通过卷积层 ----------
        # self.conv(x) 会让数据依次通过卷积→标准化→ReLU→池化
        # 输入: [batch_size, 1, 28, 28]
        # 输出: [batch_size, 32, 14, 14]
        out = self.conv(x)

        # ---------- 第2步：展平 ----------
        # .view() 相当于 numpy 的 reshape，用于改变张量形状
        # out.size()[0] 是 batch_size，保持批次大小不变
        # -1 表示自动计算该维度的大小
        # 输入: [batch_size, 32, 14, 14]
        # 输出: [batch_size, 6272]
        out = out.view(out.size()[0], -1)

        # ---------- 第3步：通过全连接层 ----------
        # self.fc 是最后的分类层
        # 输入: [batch_size, 6272] (展平后的特征)
        # 输出: [batch_size, 10] (10个数字的预测得分)
        out = self.fc(out)

        # 返回预测结果
        return out