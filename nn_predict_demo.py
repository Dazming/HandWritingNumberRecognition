"""
MNIST 神经网络预测过程演示

本脚本可视化展示神经网络预测一张 MNIST 图片时的内部工作过程。
按任意键查看下一张图片，按 ESC 键退出。

【四个可视化阶段】
- 阶段1: 卷积层激活 (32 张特征图)
- 阶段2: 池化后特征图 (32 张)
- 阶段3: 张量形状变化 (打印到终端)
- 阶段4: 预测置信度柱状图
"""

import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as utils
import torch.nn.functional as F
import cv2
import numpy as np



# ==================== 加载数据集和模型 ====================

test_dataset = dataset.MNIST(
    root="./data/",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

test_loader = utils.DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=True
)

cnn = torch.load("mnist_model.pkl")
cnn = cnn.cuda()
cnn.eval()


# ==================== 创建 Hook 捕获中间激活 ====================

conv_output = None
pool_output = None


def conv_hook(module, input, output):
    global conv_output
    conv_output = output


def pool_hook(module, input, output):
    global pool_output
    pool_output = output


conv_layer = cnn.conv[0]
pool_layer = cnn.conv[3]

conv_handle = conv_layer.register_forward_hook(conv_hook)
pool_handle = pool_layer.register_forward_hook(pool_hook)


# ==================== 创建窗口 ====================

cv2.namedWindow('NN Prediction Demo', cv2.WINDOW_NORMAL)
cv2.resizeWindow('NN Prediction Demo', 1100, 750)


# ==================== 辅助函数 ====================


def create_grid(images, rows, cols, img_size):
    """将多张图片拼接成网格"""
    grid = np.zeros((rows * img_size, cols * img_size), dtype=np.uint8)
    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break
        r = idx // cols
        c = idx % cols
        grid[r*img_size:(r+1)*img_size, c*img_size:(c+1)*img_size] = img
    return grid


def draw_confidence_bar(canvas, probs, pred, start_x, start_y, width, height):
    """在指定区域绘制置信度柱状图"""
    bar_height = height // 11
    max_bar_width = width - 80
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(10):
        bar_w = int(probs[i] * max_bar_width)
        y = start_y + i * bar_height + 5

        color = (0, 255, 0) if i == pred else (180, 180, 180)
        cv2.rectangle(canvas, (start_x, y), (start_x + bar_w, y + bar_height - 3), color, -1)

        label = f"{i}: {probs[i]*100:.1f}%"
        cv2.putText(canvas, label, (start_x + max_bar_width + 5, y + bar_height - 5),
                    font, 0.4, (255, 255, 255), 1)

    cv2.putText(canvas, "Confidence", (start_x, start_y - 5), font, 0.5, (255, 255, 255), 1)


# ==================== 主循环 ====================

print("=" * 60)
print("神经网络预测过程演示")
print("按任意键查看下一张图片, ESC 退出")
print("=" * 60)

for i, (images, labels) in enumerate(test_loader):
    images = images.cuda()
    labels_val = labels.cuda()

    outputs = cnn(images)
    probs = F.softmax(outputs, dim=1)[0].cpu().detach().numpy()
    _, pred = outputs.max(1)

    is_correct = (pred == labels_val).item()
    true_label = labels[0].item()
    pred_label = pred[0].item()

    print(f"\n--- 图片 {i+1} ---")
    print(f"输入形状: {images.shape}")
    print(f"卷积层输出形状: {conv_output.shape}")
    print(f"池化层输出形状: {pool_output.shape}")
    print(f"展平后形状: ({pool_output.shape[0]}, {pool_output.shape[1]*pool_output.shape[2]*pool_output.shape[3]})")
    print(f"全连接层输出形状: {outputs.shape}")
    print(f"真实标签: {true_label}, 预测: {pred_label}, {'正确' if is_correct else '错误'}")
    print(f"置信度分布: {probs}")

    conv_output_np = conv_output[0].cpu().detach().numpy()
    pool_output_np = pool_output[0].cpu().detach().numpy()

    orig_img = images[0].cpu().numpy().squeeze()
    orig_img = (orig_img * 255).astype(np.uint8)
    orig_img_large = cv2.resize(orig_img, (200, 200), interpolation=cv2.INTER_NEAREST)

    # 动态获取特征图尺寸
    conv_h, conv_w = conv_output_np.shape[1], conv_output_np.shape[2]
    pool_h, pool_w = pool_output_np.shape[1], pool_output_np.shape[2]

    conv_grid = create_grid([(conv_output_np[j] * 255 / conv_output_np.max()).astype(np.uint8)
                             for j in range(32)], 4, 8, conv_h)
    conv_grid_large = cv2.resize(conv_grid, (320, 160), interpolation=cv2.INTER_NEAREST)

    pool_grid = create_grid([(pool_output_np[j] * 255 / pool_output_np.max()).astype(np.uint8)
                            for j in range(32)], 4, 8, pool_h)
    pool_grid_large = cv2.resize(pool_grid, (320, 160), interpolation=cv2.INTER_NEAREST)

    # 窗口尺寸: 1100 x 750
    canvas = np.zeros((750, 1100, 3), dtype=np.uint8)
    canvas[:, :] = (25, 25, 35)

    font = cv2.FONT_HERSHEY_SIMPLEX
    status_color = (0, 255, 0) if is_correct else (0, 0, 255)

    # ========== 左列: 原始图片 (居中靠上) ==========
    # 图片 200x200，放在左侧中间
    canvas[50:250, 30:230] = cv2.cvtColor(orig_img_large, cv2.COLOR_GRAY2BGR)
    cv2.putText(canvas, "Input", (80, 270), font, 0.6, (200, 200, 200), 1)
    cv2.putText(canvas, "28x28 grayscale", (60, 290), font, 0.4, (150, 150, 150), 1)

    # ========== 中列: 卷积层 + 池化层特征图 (上下排列) ==========
    conv_display = cv2.resize(conv_grid_large, (280, 130), interpolation=cv2.INTER_NEAREST)
    canvas[50:180, 280:560] = cv2.cvtColor(conv_display, cv2.COLOR_GRAY2BGR)
    cv2.putText(canvas, "Conv: 32 maps (28x28)", (285, 195), font, 0.45, (200, 200, 200), 1)

    pool_display = cv2.resize(pool_grid_large, (280, 130), interpolation=cv2.INTER_NEAREST)
    canvas[210:340, 280:560] = cv2.cvtColor(pool_display, cv2.COLOR_GRAY2BGR)
    cv2.putText(canvas, "Pool: 32 maps (14x14)", (285, 355), font, 0.45, (200, 200, 200), 1)

    # ========== 右列: 置信度柱状图 ==========
    draw_confidence_bar(canvas, probs, pred_label, 580, 60, 500, 350)

    # ========== 底部: 张量形状流程 ==========
    cv2.rectangle(canvas, (20, 580), (1080, 640), (40, 40, 55), -1)
    cv2.putText(canvas, "Tensor Flow:", (30, 610), font, 0.45, (180, 180, 180), 1)
    shape_txt = f"{images.shape[2]}x{images.shape[3]} --> Conv({conv_h}) --> Pool({pool_h}) --> Flatten(6272) --> FC(10)"
    cv2.putText(canvas, shape_txt, (140, 610), font, 0.45, (220, 220, 220), 1)

    # ========== 底部状态栏 ==========
    cv2.rectangle(canvas, (20, 660), (1080, 730), (35, 35, 50), -1)
    info = f"True: {true_label}    Pred: {pred_label}    {'Correct' if is_correct else 'Wrong'}"
    cv2.putText(canvas, info, (40, 700), font, 0.7, status_color, 2)
    cv2.putText(canvas, "Any key: next    ESC: quit", (750, 700), font, 0.4, (150, 150, 150), 1)

    # 底部状态栏
    cv2.rectangle(canvas, (10, 755), (1190, 795), (40, 40, 50), -1)
    info = f"True: {true_label}  |  Pred: {pred_label}  |  {'Correct' if is_correct else 'Wrong'}"
    cv2.putText(canvas, info, (30, 782), font, 0.6, status_color, 2)

    shape_flow = f"Shapes: {images.shape}  -->  {conv_output.shape}  -->  {pool_output.shape}  -->  ({pool_output.shape[0]}, {pool_output.shape[1]*pool_output.shape[2]*pool_output.shape[3]})  -->  {outputs.shape}"
    cv2.putText(canvas, shape_flow, (400, 782), font, 0.4, (180, 180, 180), 1)

    cv2.putText(canvas, "Any key: next image  |  ESC: quit", (950, 782), font, 0.4, (120, 120, 120), 1)

    cv2.imshow('NN Prediction Demo', canvas)
    key = cv2.waitKey(0) & 0xFF
    if key == 27:
        break

conv_handle.remove()
pool_handle.remove()
cv2.destroyAllWindows()
print("\n演示结束")