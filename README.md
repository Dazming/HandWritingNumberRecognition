<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; line-height: 1.6; }
        .lang-toggle { text-align: right; margin-bottom: 20px; }
        .lang-btn { padding: 8px 16px; cursor: pointer; background: #333; color: #fff; border: none; border-radius: 4px; }
        .lang-btn:hover { background: #555; }
        h1 { border-bottom: 2px solid #333; padding-bottom: 10px; }
        h2 { border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-top: 30px; }
        code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
        pre { background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }
        .image-container { display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }
        .image-container img { max-width: 400px; border: 1px solid #ddd; border-radius: 5px; }
        .en { display: none; }
        body:lang(en) .zh { display: none; }
        body:lang(en) .en { display: block; }
        .info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }
        .info-box { background: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #333; }
    </style>
</head>
<body>

<div class="lang-toggle">
    <button class="lang-btn" onclick="document.body.lang = document.body.lang === 'en' ? 'zh' : 'en'">中文/EN</button>
</div>

<h1 class="zh">MNIST 手写数字识别</h1>
<h1 class="en">MNIST Handwritten Digit Recognition</h1>

<p class="zh">基于 PyTorch 的 MNIST 手写数字识别项目。</p>
<p class="en">PyTorch-based MNIST handwritten digit recognition project.</p>

<h2 class="zh">项目结构</h2>
<h2 class="en">Project Structure</h2>
<pre>
HandWritingNumber/
├── model.py            <span class="zh"># CNN 模型定义</span><span class="en"># CNN model definition</span>
├── minist_train.py     <span class="zh"># 训练脚本</span><span class="en"># Training script</span>
├── mnist_predict.py    <span class="zh"># 预测/展示脚本</span><span class="en"># Prediction/display script</span>
├── nn_predict_demo.py  <span class="zh"># 神经网络预测过程演示</span><span class="en"># NN prediction demo</span>
├── mnist_model.pkl     <span class="zh"># 训练好的模型权重</span><span class="en"># Trained model weights</span>
├── data/               <span class="zh"># MNIST 数据集目录</span><span class="en"># MNIST dataset directory</span>
└── README.md
</pre>

<h2 class="zh">演示截图</h2>
<h2 class="en">Demo Screenshots</h2>
<div class="image-container">
    <div>
        <p class="zh" style="text-align:center">mnist_predict.py - 预测展示</p>
        <p class="en" style="text-align:center">mnist_predict.py - Prediction Display</p>
        <img src="E1E53A9A1BFAD6F7C67A835CBED62D65.png" alt="MNIST Prediction">
    </div>
    <div>
        <p class="zh" style="text-align:center">nn_predict_demo.py - 过程演示</p>
        <p class="en" style="text-align:center">nn_predict_demo.py - Process Demo</p>
        <img src="4F7E633A561BAA169080F44BFB509DE6.png" alt="NN Demo">
    </div>
</div>

<h2 class="zh">环境要求</h2>
<h2 class="en">Requirements</h2>
<ul class="zh">
    <li>Python 3.8+</li>
    <li>PyTorch (CUDA 支持)</li>
    <li>OpenCV (用于预测结果可视化)</li>
</ul>
<ul class="en">
    <li>Python 3.8+</li>
    <li>PyTorch (CUDA support)</li>
    <li>OpenCV (for prediction visualization)</li>
</ul>

<h2 class="zh">模型架构</h2>
<h2 class="en">Model Architecture</h2>
<ul class="zh">
    <li><strong>卷积层</strong>: Conv2d(1, 32, kernel_size=5, padding=2) → BatchNorm → ReLU → MaxPool</li>
    <li><strong>全连接层</strong>: Linear(14×14×32, 10)</li>
    <li><strong>输出</strong>: 10 个类别 (0-9)</li>
</ul>
<ul class="en">
    <li><strong>Conv Layer</strong>: Conv2d(1, 32, kernel_size=5, padding=2) → BatchNorm → ReLU → MaxPool</li>
    <li><strong>FC Layer</strong>: Linear(14×14×32, 10)</li>
    <li><strong>Output</strong>: 10 classes (0-9)</li>
</ul>

<h2 class="zh">训练</h2>
<h2 class="en">Training</h2>
<pre><code>python minist_train.py</code></pre>
<ul class="zh">
    <li>训练集: 60,000 张图片</li>
    <li>测试集: 10,000 张图片</li>
    <li>Batch Size: 64</li>
    <li>Epochs: 10</li>
    <li>优化器: Adam (lr=0.01)</li>
    <li>损失函数: CrossEntropyLoss</li>
</ul>
<ul class="en">
    <li>Training set: 60,000 images</li>
    <li>Test set: 10,000 images</li>
    <li>Batch Size: 64</li>
    <li>Epochs: 10</li>
    <li>Optimizer: Adam (lr=0.01)</li>
    <li>Loss: CrossEntropyLoss</li>
</ul>

<h2 class="zh">预测</h2>
<h2 class="en">Prediction</h2>
<pre><code>python mnist_predict.py</code></pre>
<ul class="zh">
    <li>按任意键查看下一张图片</li>
    <li>按 ESC 退出</li>
    <li>绿色边框 = 正确，红色边框 = 错误</li>
</ul>
<ul class="en">
    <li>Press any key to view next image</li>
    <li>Press ESC to exit</li>
    <li>Green border = correct, Red border = wrong</li>
</ul>

<h2 class="zh">神经网络预测过程演示</h2>
<h2 class="en">Neural Network Prediction Demo</h2>
<pre><code>python nn_predict_demo.py</code></pre>
<p class="zh">可视化展示神经网络预测的内部工作过程：</p>
<p class="en">Visualizes the internal working process of neural network prediction:</p>
<ul class="zh">
    <li><strong>左列</strong>: 原始输入图片 (28×28 灰度图)</li>
    <li><strong>中列</strong>: 卷积层特征图 (32 张 28×28) 和池化层特征图 (32 张 14×14)</li>
    <li><strong>右列</strong>: 10 个数字的 Softmax 置信度分布</li>
    <li><strong>底部</strong>: 张量形状变化流程</li>
</ul>
<ul class="en">
    <li><strong>Left</strong>: Original input image (28×28 grayscale)</li>
    <li><strong>Middle</strong>: Conv feature maps (32 × 28×28) and Pool feature maps (32 × 14×14)</li>
    <li><strong>Right</strong>: Softmax confidence distribution for 10 digits</li>
    <li><strong>Bottom</strong>: Tensor shape transformation flow</li>
</ul>
<p class="zh">按任意键查看下一张图片，按 ESC 退出。</p>
<p class="en">Press any key for next image, ESC to quit.</p>

</body>
</html>