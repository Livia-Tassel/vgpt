# Mini-GPT: 基于 PyTorch 实现的字符级 GPT 语言模型
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

本项目从零开始，使用 PyTorch 实现了一个仅解码的 GPT模型。其主要目的在于教学与研究，清晰地展示了 Transformer 架构的核心组件，包括多头自注意力机制、位置编码、以及自回归式的文本生成流程。

---

## 项目简介
Mini-GPT 是一个轻量级的语言模型，它在莎士比亚作品的文本数据集上进行训练，以学习生成类似风格的文本。与大型语言模型不同，本项目采用字符级的分词策略，使其结构更简单，更易于理解和训练。整个项目被设计为高度模块化，方便开发者进行实验、修改与扩展。

## 功能特性
* **完整的 Transformer 架构**：实现了标准的 Transformer Block，包含多头自注意力机制、前馈神经网络、层归一化和残差连接。
* **模块化代码**：将数据处理、模型定义、训练与生成逻辑分离到独立的文件中，提升了代码的可读性与可维护性。
* **弹性的配置管理**：所有超参数（如模型维度、网络层数、学习率等）均集中在 `config.py` 中，方便快速调整与实验。
* **硬件加速**：自动检测并支持 Apple Silicon (MPS) 与 NVIDIA (CUDA) 的 GPU 加速，大幅缩短训练时间。
* **端到端的流程**：提供从数据下载、预处理、模型训练到文本生成的完整脚本。

## 架构概览
模型的核心架构遵循标准的 GPT 设计：
1.  **嵌入层**：将输入的字符索引转换为高维度的词嵌入向量，并加上位置嵌入向量以提供序列顺序信息。
2.  **Transformer Blocks**：由 N 个 Transformer Block 堆叠而成。每个 Block 包含：
    * **带有掩码的多头自注意力层**，用于捕捉上下文信息。
    * **前馈神经网络**，用于进行非线性变换。
3.  **输出层**：线性层，将 Transformer 的输出转换为对应词汇表大小的 Logits，用于预测下一个字符。

## 环境要求
* Python 3.9+
* Git
* Conda

## 快速入门
对于熟悉 Python 环境的开发者，可以依照以下步骤快速启动项目：
```bash
# 1. 克隆项目
git clone [https://github.com/Livia-Tassel/vgpt.git](https://github.com/Livia-Tassel/vgpt.git)
cd vgpt

# 2. 创建并启用 Conda 环境
conda create --name vgpt python=3.10
conda activate vgpt

# 3. 安装依赖
pip install -r requirements.txt

# 4. 开始训练
python train.py

# 5. 生成文本
python generate.py
```

## 详细步骤
### 1. 克隆项目
```bash
git clone [https://github.com/Livia-Tassel/vgpt.git](https://github.com/Livia-Tassel/vgpt.git)
cd vgpt
```

### 2. 环境设置
我强烈建议使用 Conda 来创建一个干净且独立的 Python 环境。
#### 使用 Conda
```bash
# 创建一个名为 vgpt 的新环境
conda create --name vgpt python=3.10

# 启用该环境
conda activate vgpt

# 安装所有必要的函数库
pip install -r requirements.txt
```
启用成功后，您的终端提示符前会显示 `(vgpt)`。

#### 使用 venv
如果您未安装 Conda，也可以使用 Python 内置的 `venv`。
```bash
# 创建虚拟环境
python3 -m venv venv

# 启用虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 模型训练
在启用虚拟环境后，执行训练脚本。程序会自动下载数据集并开始训练。
```bash
python train.py
```
训练过程中，您会看到损失值在终端上定期输出。训练完成后，模型权重会被保存在 `saved_models/gpt_model.pth`。

### 4. 文本生成
使用训练好的模型来生成新的文本。
```bash
python generate.py
```
脚本会自动加载已保存的模型，并在屏幕上打印出一段模仿莎士比亚风格的文本。

## 配置管理
本项目的所有超参数都集中在 `config.py` 文件中，您可以轻松修改以进行实验。
* `BATCH_SIZE`, `BLOCK_SIZE`：控制训练时的批量大小与上下文长度。
* `MAX_ITERS`, `LEARNING_RATE`：设定训练的总迭代次数与学习率。
* `N_EMBD`, `N_HEAD`, `N_LAYER`：定义模型的维度、注意力头数和 Transformer 的层数，直接影响模型的容量与性能。

## 代码结构
```
vgpt/
├── data/                  # 存放训练数据
├── saved_models/          # 存放训练好的模型权重
├── config.py              # 所有超参数与配置
├── data_loader.py         # 数据下载、预处理与批量生成
├── model.py               # GPT 模型架构定义
├── train.py               # 执行模型训练的主程序
├── generate.py            # 使用已训练模型生成文本
├── requirements.txt       # 项目依赖的 Python 包
└── README.md
```

## 贡献指南
我们欢迎任何形式的贡献！如果您有任何建议或发现了 bug，请提交一个 Issue。如果您想贡献代码，请遵循以下流程：
1.  Fork 本项目。
2.  创建一个新的分支 (`git checkout -b feature/your-feature-name`)。
3.  提交您的变更 (`git commit -m 'Add some feature'`)。
4.  将您的分支推送到远程 (`git push origin feature/your-feature-name`)。
5.  创建一个 Pull Request。

## 授权条款
本项目采用 [MIT License](https://opensource.org/licenses/MIT) 授权。

## 联系信息
项目维护者：[Livia] - [3459465562@qq.com]
项目链接：[https://github.com/Livia-Tassel/vgpt](https://github.com/Livia-Tassel/vgpt)