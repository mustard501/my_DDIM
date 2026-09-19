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

## vs. Original Paper

This repo implements the core DDPM recipe (ε-prediction + L_simple + EMA) but uses a **smaller U-Net** for faster iteration:

| | This repo | Paper (CIFAR-10) |
|--|-----------|------------------|
| Params | ~3.6M (`dim=64`) | ~35.7M (`dim=128`) |
| ResBlocks / level | 1 | 2 |
| Training steps | 200k (default) | 800k |
| Target FID | — | 3.17 |

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
