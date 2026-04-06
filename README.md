# MNIST 手写数字识别

基于 PyTorch 的 MNIST 手写数字识别项目。

## 项目结构

```
HandWritingNumber/
├── model.py            # CNN 模型定义
├── minist_train.py     # 训练脚本
├── mnist_predict.py    # 预测/展示脚本
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