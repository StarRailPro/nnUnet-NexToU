# nnU-Net NexToU

本项目是一个基于 **nnU-Net V2** 框架的高性能医学图像分割解决方案，集成了自定义的 **NexToU** 网络架构与 **BTI Loss** 优化策略。

## 1. 框架背景：nnU-Net
**nnU-Net**（no-new-Net）是由 **Helmholtz Imaging Applied Computer Vision Lab (HI-ACVL)** 和 **德国癌症研究中心 (DKFZ)** 开发的一款开箱即用的自动化分割框架。

* **核心理念**：nnU-Net 并不是提出一种新的网络结构，而是通过全自动化的方式，针对任何给定的数据集优化整个分割流水线（包括预处理、配置、训练和后处理）。
* **版本说明**：本项目基于 **nnU-Net V2**（版本 2.2），该版本较 V1 进行了彻底重构，支持更多样的数据格式（如 .png, .tif）和更灵活的扩展性。

## 2. 核心架构：NexToU
**NexToU** 是在 nnU-Net 框架基础上引入的改进型网络。它旨在结合 CNN 的局部提取能力与 Transformer 的长程建模优势：

* **混合模块**：集成了 `NexToU_Encoder_Decoder` 模块，通过多尺度特征融合提升分割精度。
* **BTI Loss (Boundary-Textured-Intensity)**：引入了专门的复合损失函数（如 `CompoundBTILoss`），针对边界、纹理和强度信息进行深度优化。
* **性能优异**：在 Synapse (BTCV) 腹部器官分割和 ICA 脑血管分割等挑战性任务中表现出色。

## 3. 安装说明
确保环境满足 `Python >= 3.9` 和 `PyTorch >= 2.0.0`。
在项目根目录下执行：
```bash
pip install -e .
```

## 4. 使用指南

### 数据规划与预处理
按照 nnU-Net V2 标准整理数据后，执行指纹提取与规划：
```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

### 模型训练
通过指定特定的 `Trainer` 来调用 NexToU 架构进行训练：
* **通用 NexToU 训练**：
    ```bash
    nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainer_NexToU
    ```
* **特定任务优化 (BTI Loss)**：
    * Synapse 任务：`-tr nnUNetTrainer_NexToU_BTI_Synapse`
    * RAVIR 任务：`-tr nnUNetTrainer_NexToU_BTI_RAVIR`

### 推理预测
使用训练好的模型进行自动推理：
```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIGURATION
```
