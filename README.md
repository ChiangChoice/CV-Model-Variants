# CV-Model-Variants

[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8.0-red.svg)](https://pytorch.org/)

基于 SwinIR、HAT-Net、CFFM++ 等前沿视觉模型的复现与改进项目集合。包含图像超分辨率、语义分割等任务的模型变体与消融实验。

**声明**：本仓库仅用于学术研究。如涉及任何侵权问题，请联系删除。

---

## 📁 项目结构

```
CV-Model-Variants/
├── SwinIR/                      # SwinIR 原始复现 (ICCV 2021)
├── HAT-Net/                     # HAT-Net 复现 (MIR 2024)
├── HATNetIR/                    # HATNetIR: 分层注意力超分迁移
├── DAR-SwinIR/                  # DAR 变体: 深度注意力路由
├── CFFM++/                      # CFFM++ + VMamba 架构融合
└── swinir_series_test_results/  # 测试结果可视化
```

---

## 🚀 模型概览

### 1. SwinIR (ICCV 2021)

图像超分辨率经典模型，基于 Swin Transformer 构建。

| 项目 | 配置 |
|------|------|
| 任务 | Classical SR (×2) |
| 训练集 | DIV2K |
| 测试集 | Set5 |
| 框架 | KAIR |

**复现结果**：
- PSNR: **38.31 dB** (论文 38.35 dB)
- 迭代次数: 525k

---

### 2. HAT-Net (MIR 2024)

分层注意力 Transformer 的复现与任务迁移实验。

| 项目 | 配置 |
|------|------|
| 任务 | ImageNet-1K 分类 → ×2 超分 |
| 复现模型 | HAT-Net-Tiny |
| 迁移模型 | HATNetIR |

**复现结果**：
- ImageNet-1K Top-1: **79.30%** (论文 79.80%)

**HATNetIR 创新点**：
- 空间网格注意力 + 通道注意力交替耦合机制
- Patch Merging Pool 无损下采样 (PixelUnshuffle)

**HATNetIR 结果**：
- 100k 迭代 PSNR: **38.08 dB**
- 计算量: 22.40G FLOPs (比 SwinIR 降低 **17%**)

---

### 3. DAR-SwinIR (深度注意力路由)

参考 Kimi 团队 AttnRes (2026)，为 SwinIR 设计深度注意力路由机制。

**技术特点**：
- 输入依赖的 Query 生成（基于当前累积状态投射）
- WSD 调度器 (Warmup-Stable-Decay)
- RMSNorm 替代 LayerNorm

**消融实验**：

| 变体 | 残差策略 | PSNR (Set5) |
|------|---------|-------------|
| DAR1 | 保留所有残差 | 38.18 dB |
| DAR1.5 | 移除块级残差 | 38.06 dB |
| DAR2.5 | 移除所有残差 | 37.99 dB |

**结论**：深度注意力路由能有效加速早期收敛，但完全依赖路由取代残差会导致精度折损。

---

### 4. CFFM++ + VMamba (TPAMI 2024)

将 VMamba 四向扫描机制 (SS2D) 引入 CFFM++ 解码器架构。

**技术理解**：
- CFFM 借鉴 Focal Transformer 的聚焦机制，实现跨帧时序粗细粒度检索
- CFFM++ 进一步通过 K-Means 聚类提取全局时序原型

**复现结果** (VSPW 480P)：

| 模型 | 骨干网络 | mIoU |
|------|---------|------|
| CFFM | MiT-B0 | 34.89% |
| CFFM++ | MiT-B0 | 35.26% |
| VMamba-CFFM | VMamba-Tiny | 40.19% |
| VMamba-CFFM++ | VMamba-Tiny | 40.86% |

**观察**：VMamba 在 ImageNet-1K 预训练下受短序列局限，性能不及同规模 MiT，初步验证了 MambaOut (CVPR 2025) 的观点。

---

## 🔧 环境配置

```bash
# 环境要求
Python 3.8
PyTorch 1.8.0
CUDA 11.1

# 安装依赖
pip install torch==1.8.0 torchvision==0.9.0
pip install timm==0.4.12 numpy==1.19 pillow==8.3.2
pip install opencv-python matplotlib
```

---

## 📊 实验结果可视化

`swinir_series_test_results/` 目录包含：
- GT (Ground Truth) 参考图像
- SwinIR_Repro 复现结果
- HATNetIR 迁移结果
- HATNetIR2 完整训练曲线结果

---

## 📦 预训练权重

所有模型的预训练权重已上传至 Google Drive，包含：
- SwinIR (×2) 复现权重
- HATNetIR 迁移权重
- DAR-SwinIR 系列权重 (DAR1 / DAR1.5 / DAR2.5)
- VMamba-CFFM++ 权重

🔗 [Google Drive 下载链接](https://drive.google.com/drive/folders/1wrAMu4Z54rtBZB_y3CAMbtPaDpXDiJjR?usp=drive_link)

---

## 📝 引用

若本仓库对您的研究有帮助，请引用相关论文：

```bibtex
@inproceedings{liang2021swinir,
  title={SwinIR: Image Restoration Using Swin Transformer},
  author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={ICCV},
  year={2021}
}

@article{chen2024hatnet,
  title={HAT-Net: Hierarchical Attention Transformer for Image Restoration},
  author={Chen, Xiangyu and Wang, Xintao and Zhou, Jiantao and Qiao, Yu and Dong, Chao},
  journal={Machine Intelligence Research},
  year={2024}
}

@article{sun2024cffm,
  title={Learning Local and Global Temporal Contexts for Video Semantic Segmentation},
  author={Sun, Guolei and Liu, Yun and Ding, Henghui and Wu, Min and Van Gool, Luc},
  journal={IEEE TPAMI},
  year={2024}
}
```

---

## 👤 Author

- GitHub: [@ChiangChoice](https://github.com/ChiangChoice)
- Email: chiangchoice01@gmail.com

---

## ⭐ Star History

如果这个项目对你有帮助，欢迎 Star ⭐
