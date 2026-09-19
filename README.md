# DDIM from Scratch

[中文文档](README_zh.md)

A minimal, single-file reimplementation of **Denoising Diffusion Probabilistic Models** (Ho et al., NeurIPS 2020) and **Denoising Diffusion Implicit Models** (Song et al., ICLR 2021) on CIFAR-10. Inspired by the [nanoGPT](https://github.com/karpathy/nanoGPT) style: one script, readable code, easy to learn and extend.

**Papers:**

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (arXiv:2006.11239)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (arXiv:2010.02502)

## Features

- Forward diffusion, L_simple training, and full 1000-step DDPM sampling
- DDIM sampling: 100-step sub-sequence by default, adjustable η (η=0 deterministic, η=1 DDPM-level stochasticity)
- U-Net noise predictor with sinusoidal timestep embedding and spatial self-attention
- EMA weights for sampling
- TensorBoard logging (loss, speed, sample grids, FID)
- FID evaluation against the CIFAR-10 **training set** (same protocol as the paper)

## Project Layout

```
my_DDIM/
├── train.py              # training, sampling, FID eval (all-in-one)
├── requirements.txt
├── docs/
│   ├── DDPM-note.html    # paper reading notes (Chinese)
│   └── 2010.02502v4.pdf  # DDIM paper
├── data/                 # CIFAR-10 (auto-downloaded)
└── runs/                 # checkpoints, samples, tensorboard logs
```

## Setup

Requires Python 3.11+, NVIDIA GPU recommended (CUDA 12.0+ driver).

```bash
pip install -r requirements.txt
```

## Quick Start

### Train

```bash
python train.py --out runs/ddpm_cifar
```

CIFAR-10 is downloaded to `data/` automatically. Checkpoints and sample grids are saved under `runs/ddpm_cifar/`.

### Resume

```bash
python train.py --out runs/ddpm_cifar --resume runs/ddpm_cifar/ckpt.pt
```

### Sample

`--sample` uses DDIM by default (100-step sub-sequence):

```bash
python train.py --sample --ckpt runs/ddpm_cifar/ckpt.pt --out runs/ddpm_cifar   # eta=0, deterministic
python train.py --sample --ckpt runs/ddpm_cifar/ckpt.pt --eta 1.0               # DDPM-level noise
```

Outputs `samples_final.png` and `progression.png` (coarse-to-fine denoising; add `--progress_every 10` to see intermediate stages). With `--eta 0` and the same seed, repeated runs produce identical images.

> FID defaults to DDIM (`--fid_sampler ddim`, 100-step, η=0). Use `--fid_sampler ddpm` for the full 1000-step DDPM chain.

### TensorBoard

```bash
tensorboard --logdir runs/ddpm_cifar/tb
```

Logs: `train/loss`, `train/ms_per_step`, `samples/grid`, `eval/fid`.

### FID Evaluation

Evaluate an existing checkpoint (DDIM by default):

```bash
python train.py --eval_fid --ckpt runs/ddpm_cifar/ckpt.pt --n_fid 10000
python train.py --eval_fid --ckpt runs/ddpm_cifar/ckpt.pt --fid_sampler ddpm
```

Run FID periodically during training (every 20k steps by default):

```bash
python train.py --out runs/ddpm_cifar --fid_every 20000
python train.py --out runs/ddpm_cifar --fid_every 20000 --fid_sampler ddpm
```

For paper-comparable numbers, use 50k samples (slower):

```bash
python train.py --eval_fid --ckpt runs/ddpm_cifar/ckpt.pt --n_fid 50000 --fid_sampler ddpm
```

> Default DDIM FID uses a 100-step chain. `--fid_sampler ddpm` is the paper protocol but expensive (1000 steps per image). Use `--n_fid 5000` for quick checks during development.

## Common Options

| Flag | Default | Description |
|------|---------|-------------|
| `--max_steps` | 200000 | Training steps |
| `--batch_size` | 128 | Batch size |
| `--dim` | 64 | U-Net base channels (~3.6M params) |
| `--T` | 1000 | Diffusion timesteps |
| `--eta` | 0.0 | DDIM noise level: 0 = deterministic, 1 = DDPM variance |
| `--sample_every` | 5000 | Save sample grid every N steps |
| `--fid_sampler` | `ddim` | FID sampler: `ddim` (100-step) or `ddpm` (1000-step) |
| `--fid_every` | 20000 | FID every N steps (`0` = disable) |
| `--n_fid` | 10000 | Images for FID (paper: 50000) |

## Dataset

| Item | This repo | Paper |
|------|-----------|-------|
| Dataset | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) | CIFAR-10 |
| Task | Unconditional generation | Unconditional generation |
| Training data | 50,000 train images | 50,000 train images |
| Resolution | 32×32 RGB | 32×32 RGB |
| Preprocessing | Scaled to **[-1, 1]** (`Normalize(0.5)`) | Scaled to [-1, 1] |
| Augmentation | Random horizontal flip | Random horizontal flip |
| FID reference | CIFAR-10 **train set** (50k) | CIFAR-10 **train set** (50k) |

## Results

FID: generate **50,000** samples and compare against the CIFAR-10 training set (Inception-v3, 2048-d features; same protocol as the paper). Sampling uses EMA weights. Hardware: a single RTX 4090D.

| Metric | This repo | Paper |
|--------|-----------|-------|
| **FID ↓** (DDPM, 1000 steps) | **15.94** (58.4 min) | **3.17** (Ho et al.) |
| **FID ↓** (DDIM, 100 steps, η=0) | **18.46** (7.6 min) | **4.16** (Song et al.) |
| Inception Score | not evaluated | 9.46 |
| NLL (bits/dim) | not evaluated | ≤ 3.75 |

The gap is mainly from U-Net size and training steps (see below). These numbers show that the full **train → DDPM / DDIM sample → FID** pipeline works. 100-step DDIM is about 2.5 FID worse than 1000-step DDPM, at roughly 1/8 the wall time.

## vs. Original Paper

The core algorithm matches the papers (ε-prediction, L_simple, linear β schedule, EMA sampling; DDIM skips along an arbitrary sub-sequence).

| Item | This repo | Paper (CIFAR-10) |
|------|-----------|------------------|
| U-Net params | ~3.6M (`dim=64`) | ~35.7M (`dim=128`) |
| ResBlocks / level | 1 | 2 |
| Self-attention | 16×16 (+ 4×4 bottleneck) | 16×16 |
| Training steps | 200k (default) | 800k |
| Batch size | 128 | 128 |
| Optimizer / lr | Adam, 2×10⁻⁴ | Adam, 2×10⁻⁴ |
| EMA decay | 0.9999 | 0.9999 |
| Dropout | 0.1 | 0.1 |
| Diffusion steps T | 1000 | 1000 |
| DDPM variance σ² | β_t (default) | β_t or β̃_t (similar quality) |
| DDIM sub-sequence | 100 steps, uniform, η=0 | 10–1000 steps reported, η tunable |
| L₀ discrete decoder | not implemented | yes |
| Full variational bound L | not implemented (L_simple only) | ablated in the paper |

See `docs/DDPM-note.html` for a detailed paper walkthrough.

## Reference

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
