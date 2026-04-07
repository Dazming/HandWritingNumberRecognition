# MNIST 手写数字识别

基于 PyTorch 的 MNIST 手写数字识别项目。

## 项目结构

```
HandWritingNumber/
├── model.py            # CNN 模型定义
├── minist_train.py     # 训练脚本
├── mnist_predict.py    # 预测/展示脚本
├── nn_predict_demo.py  # 神经网络预测过程演示
├── mnist_model.pkl     # 训练好的模型权重
├── data/               # MNIST 数据集目录
└── README.md
```

## 环境要求

- Python 3.8+
- PyTorch (CUDA 支持)
- OpenCV (用于预测结果可视化)


## 模型架构

- **卷积层**: Conv2d(1, 32, kernel_size=5, padding=2) → BatchNorm → ReLU → MaxPool
- **全连接层**: Linear(14×14×32, 10)
- **输出**: 10 个类别 (0-9)

## 训练

python minist_train.py

- 训练集: 60,000 张图片
- 测试集: 10,000 张图片
- Batch Size: 64
- Epochs: 10
- 优化器: Adam (lr=0.01)
- 损失函数: CrossEntropyLoss

## 预测

python mnist_predict.py

- 按任意键查看下一张图片
- 按 ESC 退出
- 绿色边框 = 正确，红色边框 = 错误

## 神经网络预测过程演示

python nn_predict_demo.py

可视化展示神经网络预测的内部工作过程：
- **左列**: 原始输入图片 (28×28 灰度图)
- **中列**: 卷积层特征图 (32 张 28×28) 和池化层特征图 (32 张 14×14)
- **右列**: 10 个数字的 Softmax 置信度分布
- **底部**: 张量形状变化流程

按任意键查看下一张图片，按 ESC 退出。