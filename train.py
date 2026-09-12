import argparse
import copy
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as TF
from torchvision.utils import save_image
from tqdm import tqdm

#======= diffusion =========

class GaussianDiffusion:
    """Forward/reverse process math with a linear beta schedule."""

    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, sigma_type="beta"):
        self.T = T
        self.sigma_type = sigma_type
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1 - betas
        acp = torch.cumprod(alphas, dim=0)
        acp_prev = F.pad(acp[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.sqrt_alphas = alphas.sqrt()
        self.sqrt_acp = acp.sqrt()
        self.sqrt_1m_acp = (1.0 - acp).sqrt()

        posterior_var = betas * (1.0 - acp_prev) / (1.0 - acp)
        self.posterior_std = posterior_var.sqrt()   


    def to(self, device):
        for k, v in vars(self).items():
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))
        return self

    def forward_sample(self, x0, t, noise):
        """Closed-form forward: x_t = sqrt(ab_t)*x_0 + sqrt(1-ab_t)*eps."""
        return (extract(self.sqrt_acp, t, x0.shape) * x0 + extract(self.sqrt_1m_acp, t, x0.shape) * noise)

    def predict_x0(self, model, xt, t):
        """Estimate x0_hat from x_t"""
        tv = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)
        eps = model(xt, tv)
        x0_hat = (xt - extract(self.sqrt_1m_acp, tv, xt.shape) * eps) \
            / extract(self.sqrt_acp, tv, xt.shape)
        return x0_hat

    def loss(self, model, x0):
        """L_simple: predict the noise eps from x_t."""
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.forward_sample(x0, t, noise)
        return F.mse_loss(model(xt, t), noise)

    @torch.no_grad()
    def p_sample(self, model, x, t):
        tv = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        eps = model(x, tv)
        coef = extract(self.betas, tv, x.shape) / extract(self.sqrt_1m_acp, tv, x.shape)
        mean = (x - coef * eps) / extract(self.sqrt_alphas, tv, x.shape)
        if t == 0:
            return mean
        if self.sigma_type == "beta_tilde":
            std = self.posterior_std[t]
        else:
            std = self.betas[t].sqrt()
        return mean + std * torch.randn_like(x)

    @torch.no_grad()
    def p_sample_loop(self, model, shape, device, progress_every=None):
        x = torch.randn(shape, device=device)
        snaps = []
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
            if progress_every and (t % progress_every == 0 or t == self.T - 1):
                snaps.append(self.predict_x0(model, x, t))
        return x, snaps
        



#======= U-Net =============

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_dim, dropout, num_groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb_proj = nn.Linear(temb_dim, out_ch)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)

class SinusoidalTimeEmbed(nn.Module):
    """Transformer sinusoidal embedding of the scalar timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class Attention(nn.Module):
    """Multi-head spatial self-attention."""

    def __init__(self, ch, num_heads=4, num_groups=8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(num_groups, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x)).view(B, 3, self.num_heads, C//self.num_heads, H*W)
        q, k, v = qkv.permute(1, 0, 2, 4, 3).unbind(0)
        att_wei = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        y = (att_wei.softmax(dim=-1) @ v).permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(y)

class Unet(nn.Module):
    """eps_theta(x_t, t). 4 resolution levels (32/16/8/4), one ResBlock per level,
    self-attention at 16x16.
    """

    def __init__(self, in_ch=3, dim=64, mults=(1, 2, 2, 2), attn_res=16, dropout=0.1):
        super().__init__()
        d1, d2, d3, d4 = (dim * m for m in mults)
        tdim = dim*4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbed(dim), nn.Linear(dim, tdim), nn.SiLU(), nn.Linear(tdim, tdim)
        )

        self.init_conv = nn.Conv2d(in_ch, d1, 3, padding=1)

        # down path
        self.res1 = ResBlock(d1, d1, tdim, dropout)
        self.attn1 = Attention(d1) if attn_res == 32 else nn.Identity()
        self.down1 = nn.Conv2d(d1, d2, 3, stride=2, padding=1)
        self.res2 = ResBlock(d2, d2, tdim, dropout)
        self.attn2 = Attention(d2) if attn_res == 16 else nn.Identity()
        self.down2 = nn.Conv2d(d2, d3, 3, stride=2, padding=1)
        self.res3 = ResBlock(d3, d3, tdim, dropout)
        self.down3 = nn.Conv2d(d3, d4, 3, stride=2, padding=1)

        # bottleneck
        self.mid1 = ResBlock(d4, d4, tdim, dropout)
        self.mid_attn = Attention(d4)
        self.mid2 = ResBlock(d4, d4, tdim, dropout)

        # up path
        self.up3 = nn.Conv2d(d4, d3, 3, padding=1)
        self.res4 = ResBlock(d3 * 2, d3, tdim, dropout)
        self.up2 = nn.Conv2d(d3, d2, 3, padding=1)
        self.res5 = ResBlock(d2 * 2, d2, tdim, dropout)
        self.attn5 = Attention(d2) if attn_res == 16 else nn.Identity()
        self.up1 = nn.Conv2d(d2, d1, 3, padding=1)
        self.res6 = ResBlock(d1 * 2, d1, tdim, dropout)

        # output
        self.out_norm = nn.GroupNorm(8, d1)
        self.out_conv = nn.Conv2d(d1, in_ch, 3, padding=1)


    def forward(self, x, t):
        temb = self.time_mlp(t)
        h = self.init_conv(x)
        s1 = self.attn1(self.res1(h, temb))
        h = self.down1(s1)
        s2 = self.attn2(self.res2(h, temb))
        h = self.down2(s2)
        s3 = self.res3(h, temb)
        h = self.down3(s3)
        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)
        h = self.up3(F.interpolate(h, scale_factor=2, mode="nearest"))
        h = self.res4(torch.cat([h, s3], dim=1), temb)
        h = self.up2(F.interpolate(h, scale_factor=2, mode="nearest"))
        h = self.attn5(self.res5(torch.cat([h, s2], dim=1), temb))
        h = self.up1(F.interpolate(h, scale_factor=2, mode="nearest"))
        h = self.res6(torch.cat([h, s1], dim=1), temb)
        return self.out_conv(F.silu(self.out_norm(h)))


#======= tools =============

def get_loader(data_dir, batch_size):
    """CIFAR-10 scaled to [-1,1] with random horizontal flips."""

    tf = TF.Compose([
        TF.RandomHorizontalFlip(),
        TF.ToTensor(),
        TF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, 
                    drop_last=True, pin_memory=True, persistent_workers=True)

class EMA:
    """Exponential moving average of model weights (paper: decay 0.9999)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

def extract(a, t, x_shape):
    """Gather from precomputed tensor a[T] at timesteps t[B], reshape for broadcasting."""
    return a.gather(0, t).view(-1, *([1] * (len(x_shape) - 1)))


def make_ema_model(model, ema_shadow):
    ema_model = copy.deepcopy(model)
    ema_model.load_state_dict(ema_shadow)
    ema_model.eval()
    return ema_model


def ensure_cifar10_ref(data_dir, ref_dir):
    """Export CIFAR-10 train split as PNGs for FID (paper uses train-set FID)."""
    marker = os.path.join(ref_dir, ".done")
    if os.path.exists(marker):
        return ref_dir
    os.makedirs(ref_dir, exist_ok=True)
    ds = torchvision.datasets.CIFAR10(
        data_dir, train=True, download=True, transform=TF.ToTensor())
    for i in tqdm(range(len(ds)), desc="export CIFAR-10 ref for FID"):
        img, _ = ds[i]
        save_image(img, os.path.join(ref_dir, f"{i:05d}.png"))
    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok\n")
    print(f"CIFAR-10 ref exported: {ref_dir} ({len(ds)} images)")
    return ref_dir


@torch.no_grad()
def generate_fid_samples(model, diffusion, device, out_dir, n, batch_size):
    """Generate n images with the EMA model and save as PNGs in out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    for old in os.listdir(out_dir):
        if old.endswith(".png"):
            os.remove(os.path.join(out_dir, old))
    idx = 0
    pbar = tqdm(total=n, desc="generate FID samples")
    while idx < n:
        bs = min(batch_size, n - idx)
        x, _ = diffusion.p_sample_loop(model, (bs, 3, 32, 32), device)
        imgs = ((x + 1) / 2).clamp(0, 1)
        for j in range(bs):
            save_image(imgs[j], os.path.join(out_dir, f"{idx + j:05d}.png"))
        idx += bs
        pbar.update(bs)
    pbar.close()


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Fréchet distance between two Gaussians (scipy>=1.14 compatible)."""
    import numpy as np
    from scipy import linalg

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2

    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():
        print(f"FID: singular product; adding {eps} to diagonal of cov estimates")
        eye = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + eye) @ (sigma2 + eye))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def compute_fid(fake_dir, ref_dir, device, batch_size=50):
    """Extract Inception features with pytorch-fid, Fréchet dist with our helper."""
    from pytorch_fid import fid_score
    from pytorch_fid.inception import InceptionV3

    dims = 2048
    block = max(1, min(batch_size, 256))
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[dims]]).to(device)
    model.eval()

    mu_g, sigma_g = fid_score.calculate_activation_statistics(
        fake_dir, model, block, dims, device, num_workers=0)
    mu_r, sigma_r = fid_score.calculate_activation_statistics(
        ref_dir, model, block, dims, device, num_workers=0)
    return _frechet_distance(mu_g, sigma_g, mu_r, sigma_r)


def run_fid_eval(model, ema_shadow, diffusion, device, args, step=None, writer=None):
    """Generate fake samples and compute FID against CIFAR-10 train set."""
    ref_dir = ensure_cifar10_ref(args.data_dir, os.path.join(args.data_dir, "cifar10_fid_ref"))
    fake_dir = os.path.join(args.out, "fid", "fake")
    ema_model = make_ema_model(model, ema_shadow)
    t0 = time.time()
    generate_fid_samples(ema_model, diffusion, device, fake_dir, args.n_fid, args.fid_batch)
    fid = compute_fid(fake_dir, ref_dir, device, batch_size=args.fid_batch)
    dt = time.time() - t0
    tag = f"step {step}" if step is not None else "final"
    print(f"FID ({tag}, n={args.n_fid}): {fid:.2f}  ({dt / 60:.1f} min)")
    if writer is not None and step is not None:
        writer.add_scalar("eval/fid", fid, step)
    return fid

#======= train & sample ====

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model = Unet(dim=args.dim, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net params: {n_params / 1e6:.2f}M, device: {device}")

    diffusion = GaussianDiffusion(T=args.T, sigma_type=args.sigma).to(device)
    ema = EMA(model, decay=args.ema_decay)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    step, losses = 0, []
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        ema.shadow = ckpt["ema"]
        step, losses = ckpt["step"], ckpt.get("losses", [])
        print(f"resumed from checkpoint: step {step}")

    loader = get_loader(args.data_dir, args.batch_size)
    tb_dir = os.path.join(args.out, "tb")
    writer = SummaryWriter(log_dir=tb_dir)
    writer.add_text("config", "\n".join(f"{k}: {v}" for k, v in sorted(vars(args).items())), 0)
    print(f"tensorboard: tensorboard --logdir {tb_dir}")

    def save_ckpt():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "ema": ema.shadow, "step": step, "losses": losses,
                    "args": vars(args)}, os.path.join(args.out, "ckpt.pt"))

    def quick_sample(tag):
        ema_model = make_ema_model(model, ema.shadow)
        x, _ = diffusion.p_sample_loop(ema_model, (args.n_samples, 3, 32, 32), device)
        imgs = ((x + 1) / 2).clamp(0, 1)
        save_image(imgs, os.path.join(args.out, f"samples_{tag}.png"),
                   nrow=8, value_range=(0, 1))
        writer.add_images("samples/grid", imgs, global_step=tag)

    t0 = time.time()
    while step < args.max_steps:
        for x, _ in loader:
            if step >= args.max_steps:
                break
            x = x.to(device, non_blocking=True)
            loss = diffusion.loss(model, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ema.update(model)
            step += 1

            if step % args.log_every == 0:
                losses.append(loss.item())
                dt = (time.time() - t0) / args.log_every * 1000
                t0 = time.time()
                print(f"step {step:>7d} | loss {loss.item():.4f} | {dt:.0f} ms/step")
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/ms_per_step", dt, step)

            if step % args.sample_every == 0:
                quick_sample(step)
                print(f"    saved samples_{step}.png")

            if args.fid_every > 0 and step % args.fid_every == 0:
                run_fid_eval(model, ema.shadow, diffusion, device, args, step=step, writer=writer)

            if step % args.ckpt_every == 0:
                save_ckpt()

    save_ckpt()
    if args.fid_final:
        run_fid_eval(model, ema.shadow, diffusion, device, args, step=step, writer=writer)
    writer.close()
    print(f"training done, {step} steps, checkpoint: {os.path.join(args.out, 'ckpt.pt')}")
            
def sample(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    model = Unet(dim=ckpt["args"]["dim"], dropout=ckpt["args"]["dropout"]).to(device)
    model.load_state_dict(ckpt["ema"])                     # sample with EMA weights
    model.eval()

    diffusion = GaussianDiffusion(T=ckpt["args"]["T"],
                                  sigma_type=ckpt["args"]["sigma"]).to(device)
    x, snaps = diffusion.p_sample_loop(model, (args.n_samples, 3, 32, 32), device,
                                       progress_every=args.progress_every)
    out = os.path.join(args.out, "samples_final.png")
    save_image((x + 1) / 2, out, nrow=8, value_range=(0, 1))
    print(f"samples: {out}")

    if snaps:                                              # progressive generation: one timestep per row
        grid = torch.stack(snaps)
        grid = (grid + 1) / 2
        out_p = os.path.join(args.out, "progression.png")
        save_image(grid.view(-1, 3, 32, 32), out_p, nrow=args.n_samples, value_range=(0, 1))
        print(f"progression: {out_p} (top to bottom: high noise -> low noise)")


def eval_fid(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ca = ckpt["args"]

    model = Unet(dim=ca["dim"], dropout=ca["dropout"]).to(device)
    model.load_state_dict(ckpt["ema"])
    diffusion = GaussianDiffusion(T=ca["T"], sigma_type=ca["sigma"]).to(device)

    fid_args = argparse.Namespace(
        data_dir=args.data_dir,
        out=ca.get("out", args.out),
        n_fid=args.n_fid,
        fid_batch=args.fid_batch,
    )
    os.makedirs(fid_args.out, exist_ok=True)
    run_fid_eval(model, ckpt["ema"], diffusion, device, fid_args)

#======= main ==============

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DDPM from scratch")
    # training
    p.add_argument("--max_steps", type=int, default=200_000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--resume", type=str, default=None)
    # model (small version)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--sigma", type=str, default="beta", choices=["beta", "beta_tilde"])
    # sampling
    p.add_argument("--sample", action="store_true", help="sample only (requires --ckpt)")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--progress_every", type=int, default=250)
    # FID eval (paper: 50k samples vs CIFAR-10 train set)
    p.add_argument("--eval_fid", action="store_true", help="FID only (requires --ckpt)")
    p.add_argument("--fid_every", type=int, default=20_000,
                   help="compute FID every N steps during training (0=disable)")
    p.add_argument("--fid_final", action="store_true",
                   help="run FID once after training finishes")
    p.add_argument("--n_fid", type=int, default=10_000,
                   help="number of generated images for FID (paper uses 50000)")
    p.add_argument("--fid_batch", type=int, default=64,
                   help="batch size for sample generation during FID")
    # misc
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--out", type=str, default="runs/ddpm_cifar")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--sample_every", type=int, default=5000)
    p.add_argument("--ckpt_every", type=int, default=5000)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.eval_fid:
        assert args.ckpt, "--eval_fid requires --ckpt"
        eval_fid(args)
    elif args.sample:
        assert args.ckpt, "--sample requires --ckpt"
        sample(args)
    else:
        train(args)