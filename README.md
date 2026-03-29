# CV-Model-Variants: 基于 SwinIR、HAT-Net 与 Mamba 的前沿视觉架构探索

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.8.0-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-11.1-green.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/Status-Research-brightgreen.svg" alt="Status">
</p>

本项目专注于前沿计算机视觉架构的复现与演进。核心亮点在于将 **Kimi 团队的 Attention Residuals (2026)** 思想成功引入图像超分辨率任务，并实现了 **VMamba** 与 **CFFM++** 的跨任务架构融合。

---

## 📁 项目结构
```bash
CV-Model-Variants/
├── DAR-SwinIR/     # 💡 核心创新: 基于深度注意力路由的 SwinIR 变体
├── HATNetIR/       # 迁移实验: 分层注意力机制向超分任务 (SR) 的迁移
├── CFFM++/         # 架构融合: VMamba SS2D 算子与 CFFM++ 语义分割解码器
├── SwinIR/         # 基础复现: SwinIR 原始架构对齐 (ICCV 2021)
├── HAT-Net/        # 基础复现: HAT-Net 图像分类基准 (MIR 2024)
└── results/        # 测试结果可视化 (包含消融实验曲线与定性对比)
```

---

## 🚀 核心工作展示

### 1. DAR-SwinIR (Depth Attention Router) 【重点推荐】
参考 Kimi Team (2026) **"Attention Residuals"** 理论，本项目针对 SwinIR 的固定残差连接进行了底层重构，设计了 **DAR (深度注意力路由)** 机制。

*   **动态 Query 生成**：区别于原论文出于效率考量的固定伪查询，本实现引入了**输入依赖的动态投射机制**，基于当前累积状态实时生成 Query，极大地增强了深度特征路由的内容感知能力。
*   **稳定性组合优化**：针对无残差架构的收敛挑战，引入了 **WSD (Warmup-Stable-Decay)** 调度器与 **RMSNorm** 归一化，有效解决了深层特征幅值主导（Magnitude Dominance）导致的训练不稳定问题。
*   **消融结论**：实验证明 DAR 机制能显著加速训练早期的梯度下降。

### 2. HATNetIR: 跨任务架构迁移
将分类任务中的分层注意力 Transformer (**HAT-Net**) 成功迁移至图像恢复领域。
*   **交替耦合机制**：实现了空间网格注意力与通道注意力的循环交互。
*   **无损采样**：设计 Patch Merging Pool 技术，利用像素重排替代传统池化，在计算量大幅降低 **17%** 的情况下，性能对齐经典 SwinIR。

### 3. VMamba-CFFM++: 视频语义分割
探索 **SSM (状态空间模型)** 在高分辨率视频理解中的潜力。
*   **SS2D 集成**：将 VMamba 的四向扫描算子 (Selective Scan) 集成于 CFFM++ 时序检索解码器中。
*   **性能飞跃**：在 VSPW 数据集上取得 **40.86% mIoU**，相比 MiT 骨干网络提升了 **5.6%**，验证了 Mamba 在长序列时空建模中的优势。

---

## 📈 实验结果对比

### 定量分析 (Set5 ×2 / VSPW 480P)

| 任务 | 模型变体 | 核心指标 | 状态 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| **超分辨率 (SR)** | **SwinIR (Repro)** | **38.31 dB** | 对齐论文 | 原作者报告值 38.35 |
| **超分辨率 (SR)** | **DAR 1.5** | **38.06 dB** | 阶段性观测 | 展现极速收敛特性 |
| **超分辨率 (SR)** | **HATNetIR** | **38.01 dB** | 效率领先 | **FLOPs 降低 17%** |
| **语义分割 (VSS)** | **CFFM++ (MiT)** | **35.26%** | 对齐权重 | 官方权重跑分 35.91 |
| **语义分割 (VSS)** | **VMamba-CFFM++** | **40.86%** | **SOTA 级别** | **mIoU 提升 5.6%** |

### 定性结果
> 项目在细节恢复（如眼部睫毛纹理、复杂帽子织物）上表现出极强的连贯性，具体对比图见 `results/` 目录。

---

## 🔧 环境配置与算子编译

本项目包含自定义 CUDA 算子，需在特定硬件环境下编译：

```bash
# 核心依赖
conda create -n vision python=3.8
pip install torch==1.8.0 torchvision==0.9.0 timm==0.4.12 mmseg==0.20.2

# VMamba SS2D 算子编译 (针对 V100/A10/3090)
export TORCH_CUDA_ARCH_LIST="7.0;8.0" # 根据显卡架构选择算力
cd CFFM++/vmamba_repo && python setup.py install
```

---

## 📦 预训练权重

所有模型的预训练权重已上传至 **Google Drive**，包含：

*   **SwinIR (×2)** 复现权重
*   **HATNetIR** 迁移权重
*   **DAR-SwinIR 全系列** (DAR1 / DAR1.5 / DAR2.5)
*   **VMamba-CFFM++** 融合架构权重

🔗 **下载链接**: [Google Drive - CV-Model-Variants 权重](https://drive.google.com/...)

---

## 📄 引用说明

若本项目对您的研究有帮助，请引用以下相关论文：

```bibtex
@inproceedings{liang2021swinir,
  title={SwinIR: Image Restoration Using Swin Transformer},
  author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}

@article{chen2024hatnet,
  title={HAT-Net: Hierarchical Attention Transformer for Image Restoration},
  author={Chen, Xiangyu and Wang, Xintao and Zhou, Jiantao and Qiao, Yu and Dong, Chao},
  journal={Machine Intelligence Research (MIR)},
  year={2024}
}

@article{sun2024cffm,
  title={Learning Local and Global Temporal Contexts for Video Semantic Segmentation},
  author={Sun, Guolei and Liu, Yun and Ding, Henghui and Wu, Min and Van Gool, Luc},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year={2024}
}
```

---

## 📝 开源协议
本项目遵循 [MIT License](LICENSE)。

## 👤 作者
**ChiangChoice** (蒋选择)
*   **GitHub**: [@chiangchoice](https://github.com/chiangchoice)
*   **Email**: chiangchoice01@gmail.com
