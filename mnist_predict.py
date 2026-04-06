"""
MNIST 手写数字识别 - 预测脚本

本脚本用于加载训练好的模型，对测试集图片进行预测并可视化展示。
按任意键查看下一张图片，按 ESC 键退出。

【展示界面说明】
- 绿色边框 = 预测正确
- 红色边框 = 预测错误
- 顶部显示：真实标签、预测标签、是否正确
- 底部显示：当前准确率
"""

# ==================== 导入必要的库 ====================

import torch  # PyTorch 核心库

# torchvision.datasets: MNIST 数据集
import torchvision.datasets as dataset

# torchvision.transforms: 图像预处理
import torchvision.transforms as transforms

# torch.utils.data: 数据加载工具
import torch.utils.data as utils

# cv2: OpenCV 库，用于图像显示和GUI
# 注意：OpenCV 导入时使用别名 cv2
import cv2

# numpy: 数值计算库，用于处理图像数组
import numpy as np


# ==================== 加载测试数据集 ====================

"""
注意：这里只加载测试集，不加载训练集
因为我们要测试模型"没见过"的数据，评估真实性能
"""

test_dataset = dataset.MNIST(
    root="./data/",               # 数据存放文件夹（与训练脚本共用）
    train=False,                  # False = 加载测试集（10000 张图片）
    transform=transforms.ToTensor(),  # 转换为张量，归一化到 0-1
    download=True                # 如果没有则下载
)

# 创建测试数据加载器
# batch_size=1: 每次只加载一张图片，方便逐张查看和展示
test_loader = utils.DataLoader(
    dataset=test_dataset,
    batch_size=1,                # 每次返回 1 张图片
    shuffle=False                # 不打乱，顺序查看
)


# ==================== 加载训练好的模型 ====================

# torch.load(): 从文件加载保存的模型
# 文件名 'mnist_model.pkl' 是训练脚本中保存的模型文件
cnn = torch.load("mnist_model.pkl")

# 将模型移到 GPU 上（如果有 GPU 的话）
cnn = cnn.cuda()

# 设置为评估模式
# 这两个模式的区别：
# - 训练模式 (train mode): 会使用 Dropout、BatchNorm 的训练行为
# - 评估模式 (eval mode): 关闭 Dropout，使用固定的 BatchNorm 统计量
cnn.eval()


# ==================== 初始化统计变量 ====================

total_loss = 0       # 累计损失值
right_count = 0      # 累计正确预测的数量
total_count = 0      # 累计已预测的图片总数


# ==================== 创建展示窗口 ====================

"""
创建一个全黑的画布用于显示图片和文字
画布尺寸：400×400 像素，3通道（彩色）
"""

# np.zeros(): 创建一个全零数组，即全黑背景
# (400, 400, 3): 高400，宽400，3个颜色通道（RGB）
# dtype=np.uint8: 数据类型，无符号8位整数（0-255）
canvas = np.zeros((400, 400, 3), dtype=np.uint8)

# cv2.namedWindow(): 创建一个窗口
# 'MNIST Prediction': 窗口标题
# cv2.WINDOW_NORMAL: 窗口大小可调（可以拖动改变大小）
cv2.namedWindow('MNIST Prediction', cv2.WINDOW_NORMAL)

# cv2.resizeWindow(): 设置窗口初始大小
cv2.resizeWindow('MNIST Prediction', 400, 400)


# ==================== 遍历测试集进行预测 ====================

# enumerate() 返回 (索引, (图片, 标签)) 的元组
for i, (images, labels) in enumerate(test_loader):

    # ---------- 第1步：数据预处理 ----------
    images = images.cuda()         # 将图片张量移到 GPU
    labels = labels.cuda()       # 将标签张量移到 GPU

    # ---------- 第2步：模型预测 ----------
    outputs = cnn(images)        # 前向传播，获取预测结果
                                # outputs 形状: [1, 10]，1张图片，10个分类得分

    # ---------- 第3步：计算损失 ----------
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(outputs, labels)
    total_loss += loss.item()

    # ---------- 第4步：获取预测类别 ----------
    # outputs.max(1) 返回每行最大值的 (值, 索引)
    # 索引就是预测的数字（0-9）
    _, pred = outputs.max(1)      # pred 形状: [1]

    # ---------- 第5步：判断对错 ----------
    # pred == labels 返回布尔值（True/False）
    # .item() 将单元素张量转换为 Python 标量
    is_correct = (pred == labels).item()
    right_count += is_correct
    total_count += 1

    # ---------- 第6步：准备显示图片 ----------
    # images[0]: 因为 batch_size=1，所以取第一个元素
    # .cpu(): 将张量从 GPU 移回 CPU（OpenCV 只支持 CPU）
    # .numpy(): 将张量转换为 numpy 数组
    # .squeeze(): 移除所有大小为 1 的维度 (1,1,28,28) → (28,28)
    img = images[0].cpu().numpy().squeeze()  # 形状: (28, 28)

    # 将归一化的像素值（0-1）转回 0-255 范围
    # 并转换为 uint8 类型（OpenCV 需要的格式）
    img = (img * 255).astype(np.uint8)

    # 将灰度图转换为彩色图（方便画彩色边框）
    # (28, 28) → (28, 28, 3)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 放大图片：28×28 → 280×280（更清晰）
    # cv2.resize() 用于调整图片大小
    # interpolation=cv2.INTER_NEAREST: 使用最近邻插值（适合像素图）
    img_large = cv2.resize(img_color, (280, 280), interpolation=cv2.INTER_NEAREST)

    # ---------- 第7步：画边框 ----------
    # 边框颜色：绿色=正确，红色=错误
    border_color = (0, 255, 0) if is_correct else (0, 0, 255)  # BGR 格式
    border_thickness = 4  # 边框厚度（像素）

    # cv2.rectangle(): 画矩形
    # 参数：(图片, 左上角坐标, 右下角坐标, 颜色, 边框厚度)
    cv2.rectangle(img_large, (0, 0), (279, 279), border_color, border_thickness)

    # ---------- 第8步：创建画布并放置图片 ----------
    # 全黑背景
    canvas.fill(0)

    # 将放大后的图片放到画布中央
    # 计算偏移量使图片居中
    y_offset = 20   # 顶部留 20 像素显示文字
    x_offset = (400 - 280) // 2  # 水平居中
    canvas[y_offset:y_offset+280, x_offset:x_offset+280] = img_large

    # ---------- 第9步：绘制文字信息 ----------

    # 获取真实标签和预测标签
    true_label = labels[0].item()  # 真实数字
    pred_label = pred[0].item()   # 预测数字
    status = "Correct" if is_correct else "Wrong"  # 是否正确

    # 选择字体
    font = cv2.FONT_HERSHEY_SIMPLEX  # 简洁的字体

    # 选择文字颜色（与边框一致）
    text_color = (0, 255, 0) if is_correct else (0, 0, 255)  # BGR 格式

    # 顶部信息：真实值、预测值、是否正确
    info = f"True: {true_label}  Pred: {pred_label}  [{status}]"
    # cv2.putText(): 在图片上绘制文字
    # 参数：(图片, 文字内容, 坐标, 字体, 字号, 颜色, 粗细)
    cv2.putText(canvas, info, (20, 15), font, 0.6, text_color, 2)

    # 底部统计：当前准确率
    acc = right_count / total_count * 100
    stats = f"Acc: {acc:.2f}%  ({right_count}/{total_count})"
    cv2.putText(canvas, stats, (20, 385), font, 0.5, (200, 200, 200), 1)

    # ---------- 第10步：显示窗口 ----------
    cv2.imshow('MNIST Prediction', canvas)

    # ---------- 第11步：等待按键 ----------
    # cv2.waitKey(): 等待按键事件
    # 参数：等待时间（毫秒），0 表示无限等待
    # 返回值：按键的 ASCII 码
    # & 0xFF 是为了兼容 32 位系统
    key = cv2.waitKey(0) & 0xFF

    if key == 27:  # ESC 键的 ASCII 码是 27
        break  # 退出循环


# ==================== 退出程序 ====================

# 关闭所有 OpenCV 创建的窗口
cv2.destroyAllWindows()

# 打印最终结果
print("\n=== 最终结果 ===")
print(f"总测试图片数: {total_count}")
print(f"正确预测数: {right_count}")
print(f"最终准确率: {right_count / total_count * 100:.2f}%")
print(f"平均损失: {total_loss / total_count:.4f}")