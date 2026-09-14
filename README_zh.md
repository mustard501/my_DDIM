# DDIM 复现

[English](README.md)

基于 Ho et al. (NeurIPS 2020) 的 **Denoising Diffusion Probabilistic Models** 与 Song et al. (ICLR 2021) 的 **Denoising Diffusion Implicit Models** 简化复现，在 CIFAR-10 上训练与采样

代码风格参考 [nanoGPT](https://github.com/karpathy/nanoGPT)：单文件、可读、便于学习与扩展。

**论文：**

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)

## 功能

- 前向扩散、L_simple 训练、1000 步 DDPM 采样
- DDIM 采样：默认 100 步子序列，η 可调（η=0 确定性，η=1 DDPM 级方差）
- U-Net 噪声预测器（正弦时间嵌入 + 空间自注意力）
- EMA 权重采样
- TensorBoard 监控（loss、速度、采样图、FID）
- FID 评测（相对 CIFAR-10 **训练集**，与论文协议一致）

## 项目结构

```
my_DDIM/
├── train.py              # 训练 / 采样 / FID 评测（单文件）
├── requirements.txt
├── docs/
│   ├── DDPM-note.html    # 论文精读笔记
│   └── 2010.02502v4.pdf  # DDIM 论文原文
├── data/                 # CIFAR-10（自动下载）
└── runs/                 # checkpoint、采样图、tensorboard 日志
```

## 环境安装

需要 Python 3.11+，建议使用 NVIDIA GPU（驱动支持 CUDA 12.0+）。

```bash
pip install -r requirements.txt
```

## 使用示例

### 训练

```bash
python train.py --out runs/ddpm_cifar
```

CIFAR-10 自动下载到 `data/`，checkpoint 和采样图保存在 `runs/ddpm_cifar/`。

### 断点续训

```bash
python train.py --out runs/ddpm_cifar --resume runs/ddpm_cifar/ckpt.pt
```

### 采样

`--sample` 默认走 DDIM（100 步子序列）：

```bash
python train.py --sample --ckpt runs/ddpm_cifar/ckpt.pt --out runs/ddpm_cifar   # eta=0，确定性
python train.py --sample --ckpt runs/ddpm_cifar/ckpt.pt --eta 1.0               # DDPM 级别的噪声
```

生成 `samples_final.png` 和 `progression.png`（由粗到细的渐进去噪过程，配 `--progress_every 10` 可查看中间去噪阶段）。η=0 时同 seed 重复采样结果逐像素一致。

> FID 评测（`--eval_fid`）仍使用完整 1000 步 DDPM 采样链，保持基准口径不变。

### TensorBoard 监控

```bash
tensorboard --logdir runs/ddpm_cifar/tb
```

记录项：`train/loss`、`train/ms_per_step`、`samples/grid`、`eval/fid`。

### FID 评测

对已有 checkpoint 评测：

```bash
python train.py --eval_fid --ckpt runs/ddpm_cifar/ckpt.pt --n_fid 10000
```

训练期间定期评测（默认每 2 万步）：

```bash
python train.py --out runs/ddpm_cifar --fid_every 20000
```

对标论文时使用 5 万张图（更慢、更准确）：

```bash
python train.py --eval_fid --ckpt runs/ddpm_cifar/ckpt.pt --n_fid 50000
```

> FID 评测较慢：每张图需完整 1000 步采样。开发阶段可用 `--n_fid 5000` 快速看趋势。

## 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_steps` | 200000 | 训练步数 |
| `--batch_size` | 128 | batch size |
| `--dim` | 64 | U-Net 基础通道（约 3.6M 参数） |
| `--T` | 1000 | 扩散步数 |
| `--eta` | 0.0 | DDIM 噪声系数：0 = 确定性，1 = DDPM 方差 |
| `--sample_every` | 5000 | 每 N 步保存采样图 |
| `--fid_every` | 20000 | 每 N 步评 FID（`0` 关闭） |
| `--n_fid` | 10000 | FID 生成样本数（论文 50000） |

## 与论文的差异

已实现 DDPM 核心配方（ε-预测 + L_simple + EMA），但 U-Net 为 **缩小版**，便于快速实验：

| | 本仓库 | 论文 CIFAR-10 |
|--|--------|---------------|
| 参数量 | ~3.6M（`dim=64`） | ~35.7M（`dim=128`） |
| 每级 ResBlock | 1 个 | 2 个 |
| 训练步数 | 200k（默认） | 800k |
| 目标 FID | — | 3.17 |

详细论文解读见 `docs/DDPM-note.html`。

## 引用

```bibtex
@inproceedings{ho2020ddpm,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  booktitle={NeurIPS},
  year={2020}
}

@inproceedings{song2021ddim,
  title={Denoising Diffusion Implicit Models},
  author={Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
  booktitle={ICLR},
  year={2021}
}
```
