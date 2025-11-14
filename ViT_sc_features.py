import os
import math
import time
import random
import json
import gc
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from datasets import load_dataset

import matplotlib.pyplot as plt

gc.enable()

# ---------------------------------------------------------------------
# Global defaults (you can override via ViTConfig / function arguments)
# ---------------------------------------------------------------------
IMAGE_SIZE = 160
NUM_CLASSES = 10
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

ImageFile.LOAD_TRUNCATED_IMAGES = True  # allow partially corrupted files

# ===========================
# 1) Core ViT building blocks
# ===========================


class MSA(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.msa = MSA(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.msa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 256):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class PatchCMLP(nn.Module):
    """Optional patch-wise MLP (not used by default)."""

    def __init__(self, embed_dim_out: int = 256, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim_out * 4
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim_out, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim_out),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ===========================
# 2) Gabor / V1-related parts
# ===========================


def make_gabor_kernel(theta, ksize=15, sigma=3.0, lam=8.0, phase=0.0):
    ax = torch.arange(ksize) - (ksize - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="xy")
    Xr = xx * math.cos(theta) + yy * math.sin(theta)
    Yr = -xx * math.sin(theta) + yy * math.cos(theta)
    env = torch.exp(-(Xr ** 2 + Yr ** 2) / (2 * sigma ** 2))
    carrier = torch.cos(2 * math.pi * Xr / lam + phase)
    kernel = env * carrier
    kernel -= kernel.mean()
    kernel /= kernel.abs().sum() + 1e-6
    return kernel


class SimpleGaborEmbed(nn.Module):
    """
    Fixed Gabor bank -> rectified -> pool to patch grid -> tokens.
    """

    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_chans=3,
        n_gabor=8,
        ksize=15,
        sigma=3.0,
        lam=8.0,
        embed_dim=None,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.in_chans = in_chans
        self.n_gabor = n_gabor
        self.embed_dim = embed_dim or n_gabor

        thetas = torch.tensor(np.linspace(0, math.pi, n_gabor, endpoint=False), dtype=torch.float32)
        gabor_kernels = torch.stack([make_gabor_kernel(t, ksize, sigma, lam) for t in thetas])

        weight = gabor_kernels.unsqueeze(1).repeat(1, in_chans, 1, 1)
        self.register_buffer("weight", weight)
        self.bias = None
        self.pool = nn.AvgPool2d(patch_size, stride=patch_size)

        self.proj = None
        if embed_dim != n_gabor:
            self.proj = nn.Linear(n_gabor, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv2d(x, self.weight, self.bias, padding="same")
        y = F.relu(y)
        y = self.pool(y)

        B, D, Hp, Wp = y.shape
        y = y.flatten(2).transpose(1, 2)  # (B, N_patches, D)

        if self.proj is not None:
            y = self.proj(y)

        return y


def gabor_odd(theta, ksize=15, sigma=3.0, lam=8.0):
    ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="xy")
    ct, st = math.cos(float(theta)), math.sin(float(theta))
    Xr = xx * ct + yy * st
    Yr = -xx * st + yy * ct
    env = torch.exp(-(Xr ** 2 + Yr ** 2) / (2 * sigma ** 2))
    ker = env * torch.sin(2 * math.pi * Xr / lam)
    ker -= ker.mean()
    ker /= ker.abs().sum() + 1e-6
    return ker


class MonocularSimpleOddGaborEmbed(nn.Module):
    """
    Odd-phase Gabor simple cells with signed responses (polarity preserved).
    """

    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_chans=3,
        n_gabor=8,
        ksize=15,
        sigma=3.0,
        lam=8.0,
        embed_dim=None,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        assert ksize % 2 == 1
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.in_chans = in_chans
        self.n_gabor = n_gabor
        self.embed_dim = embed_dim or n_gabor
        self.pad = ksize // 2

        thetas = torch.tensor(np.linspace(0, math.pi, n_gabor, endpoint=False), dtype=torch.float32)
        gabor_kernels = torch.stack([gabor_odd(t, ksize, sigma, lam) for t in thetas])
        weight = gabor_kernels.unsqueeze(1).repeat(1, in_chans, 1, 1)
        self.register_buffer("weight", weight)
        self.bias = None

        self.pool = nn.AvgPool2d(patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        y = F.conv2d(x, self.weight, bias=self.bias, padding=self.pad)  # (B, n_gabor, H, W)
        mag = y.abs()
        ori_idx = mag.argmax(dim=1)  # (B, H, W)
        gather_idx = ori_idx.unsqueeze(1)
        sel = y.gather(1, gather_idx).squeeze(1)
        polarity = torch.sign(sel)

        yp = self.pool(y)
        B, D, Hp, Wp = yp.shape
        y_embed = yp.flatten(2).transpose(1, 2)  # (B, N_patches, n_gabor)

        return y_embed, y, polarity, ori_idx


class ComplexFromSimplePatches(nn.Module):
    """
    Complex-cell-like pooling from simple-cell embeddings.
    """

    def __init__(self, n_orient: int, embed_dim_out: Optional[int] = None, eps: float = 1e-8):
        super().__init__()
        self.n_orient = n_orient
        self.eps = eps
        self.embed_dim_out = embed_dim_out or n_orient
        self.proj = None
        if self.embed_dim_out != n_orient:
            self.proj = nn.Linear(n_orient, self.embed_dim_out)

    def _energy(self, even: torch.Tensor, odd: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(even * even + odd * odd + self.eps)

    def forward(self, x: torch.Tensor, input_layout: Optional[str] = None, mode: Optional[str] = None):
        if isinstance(x, tuple):
            even, odd = x
            B, N, D = even.shape
            assert odd.shape == even.shape
            assert D == self.n_orient
            y = self._energy(even, odd)
        else:
            B, N, D = x.shape
            if mode == "odd_only":
                assert D == self.n_orient
                y = torch.sqrt(x * x + self.eps)
            else:
                assert input_layout in {"concat", "interleave"}
                assert D == 2 * self.n_orient

                if input_layout == "concat":
                    even = x[:, :, : self.n_orient]
                    odd = x[:, :, self.n_orient :]
                else:
                    even = x[:, :, 0::2]
                    odd = x[:, :, 1::2]

                y = self._energy(even, odd)

        if self.proj is not None:
            y = self.proj(y)
        return y


# ====================
# 3) ViT configurations
# ====================


@dataclass
class ViTConfig:
    img_size: int = IMAGE_SIZE
    patch_size: int = 4
    in_chans: int = 3
    part_embed_dim: int = 256          # per-stream dim for ViT_sc_features
    embed_dim: int = part_embed_dim * 2
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    num_classes: int = NUM_CLASSES
    gabor: int = 8


# ===========================
# 4) ViT variants & main model
# ===========================


class ViT_sc_features(nn.Module):
    """
    Early-fusion three-stream ViT:
      - raw patches
      - simple Gabor tokens
      - complex-cell tokens (from odd simple responses)
    Concatenate along feature dim, then pass through a single ViT encoder.
    """

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg

        self.patch_embed = PatchEmbed(
            cfg.img_size, cfg.patch_size, cfg.in_chans, int(cfg.part_embed_dim)
        )

        self.simple_gabor_embed = SimpleGaborEmbed(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            n_gabor=cfg.gabor,
            embed_dim=cfg.part_embed_dim,
        )

        self.mono_simple = MonocularSimpleOddGaborEmbed(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            n_gabor=cfg.gabor,
            embed_dim=cfg.embed_dim,
            ksize=15,
            sigma=3.0,
            lam=8.0,
        )

        self.complex_embed = ComplexFromSimplePatches(
            n_orient=cfg.gabor, embed_dim_out=cfg.part_embed_dim
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.part_embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.num_patches, cfg.part_embed_dim)
        )
        self.pos_drop = nn.Dropout(cfg.dropout)

        mlp_dim = int(cfg.embed_dim * cfg.mlp_ratio)
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(cfg.embed_dim, cfg.num_heads, mlp_dim, dropout=cfg.dropout)
                for _ in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # simple stream
        xs = self.simple_gabor_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        xs = torch.cat([cls, xs], dim=1)
        xs = xs + self.pos_embed[:, : xs.size(1), :]
        xs = self.pos_drop(xs)

        # complex stream
        x1, signed_maps, polarity, ori_idx = self.mono_simple(x)
        xc = self.complex_embed(x1, mode="odd_only")
        cls = self.cls_token.expand(B, -1, -1)
        xc = torch.cat([cls, xc], dim=1)
        xc = xc + self.pos_embed[:, : xc.size(1), :]
        xc = self.pos_drop(xc)

        # raw stream
        xr = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        xr = torch.cat([cls, xr], dim=1)
        xr = xr + self.pos_embed[:, : xr.size(1), :]
        xr = self.pos_drop(xr)

        # concat along feature dim: each has part_embed_dim, so result has 3 * part_embed_dim
        x_cat = torch.cat([xr, xs, xc], dim=-1)

        for blk in self.blocks:
            x_cat = blk(x_cat)

        x_cat = self.norm(x_cat)
        cls_out = x_cat[:, 0]
        return self.head(cls_out)


class ViT(nn.Module):
    """Baseline ViT with standard patch embedding."""

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(
            cfg.img_size, cfg.patch_size, cfg.in_chans, int(cfg.embed_dim)
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.num_patches, cfg.embed_dim)
        )
        self.pos_drop = nn.Dropout(cfg.dropout)

        mlp_dim = int(cfg.embed_dim * cfg.mlp_ratio)
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(cfg.embed_dim, cfg.num_heads, mlp_dim, dropout=cfg.dropout)
                for _ in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)


def vit_tiny(img_size=IMAGE_SIZE, num_classes=NUM_CLASSES) -> ViT:
    return ViT(
        ViTConfig(
            img_size=img_size,
            patch_size=4,
            embed_dim=192,
            depth=6,
            num_heads=8,
            num_classes=num_classes,
            mlp_ratio=4.0,
        )
    )


def vit_sc_tiny(img_size=IMAGE_SIZE, num_classes=NUM_CLASSES) -> ViT_sc_features:
    return ViT_sc_features(
        ViTConfig(
            img_size=img_size,
            patch_size=4,
            embed_dim=64 * 3,
            part_embed_dim=64,
            depth=6,
            num_heads=8,
            num_classes=num_classes,
            mlp_ratio=4.0,
        )
    )


# ==========================
# 5) Utils: metrics & losses
# ==========================


def iter_regularized_params(model: nn.Module):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias"):
            continue
        yield param


def weight_regularization(model: nn.Module, reg_type="l2", scale=1.0):
    reg = 0.0
    if reg_type == "l2":
        for p in iter_regularized_params(model):
            reg = reg + p.pow(2).sum()
    elif reg_type == "l1":
        for p in iter_regularized_params(model):
            reg = reg + p.abs().sum()
    else:
        raise ValueError("reg_type must be 'l2' or 'l1'")
    return scale * reg


class LabelSmoothingXEnt(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        logprobs = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logprobs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * logprobs, dim=-1))


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k * (100.0 / target.size(0))).item())
    return res


def cosine_with_warmup(step, total_steps, base_lr, min_lr, warmup_steps=0):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


# ======================
# 6) Training / eval loop
# ======================


def train_one_epoch(
    epoch,
    model,
    loader,
    optimizer,
    loss_fn,
    scaler,
    device,
    lr_schedule=None,
    weight_reg_lambda: float = 1e-4,
):
    model.train()
    total_loss, total_top1 = 0.0, 0.0
    n = 0

    with tqdm(total=len(loader), desc=f"Train epoch {epoch}") as pbar:
        for step, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if lr_schedule is not None:
                lr = lr_schedule()
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                logits = model(images)
                ce = loss_fn(logits, targets)
                wreg = weight_regularization(model, "l2")
                loss = ce + weight_reg_lambda * wreg

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            top1, = accuracy(logits.detach(), targets, topk=(1,))
            bs = images.size(0)
            total_loss += loss.item() * bs
            total_top1 += top1 * bs
            n += bs

            pbar.update(1)
            pbar.set_postfix(loss=f"{(total_loss / n):.4f}")

    return total_loss / n, total_top1 / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_top1 = 0.0, 0.0
    n = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = loss_fn(logits, targets)
        top1, = accuracy(logits, targets, topk=(1,))
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_top1 += top1 * bs
        n += bs
    return total_loss / n, total_top1 / n


# ==============================
# 7) Plotting & history utilities
# ==============================


def plot_histories(
    histories: List[Dict[str, list]],
    save_path: Optional[str] = None,
    show: bool = False,
    scale_val_acc: float = 10.0,
    smooth_ema: Optional[float] = 0.9,
):
    assert len(histories) > 0, "Provide at least one history."

    def _ema(xs, alpha):
        if alpha is None or not xs:
            return xs
        out = []
        m = None
        for v in xs:
            m = v if m is None else alpha * m + (1 - alpha) * v
            out.append(m)
        return out

    keys = {"train_loss", "val_loss", "val_acc"}
    present = {k for k in keys if any(k in h and h[k] for h in histories)}
    if not present:
        raise ValueError("No metrics to plot.")

    plt.figure(figsize=(9, 6))

    def plot_metric(metric_name: str, linestyle: str):
        for h in histories:
            if metric_name not in h or len(h[metric_name]) == 0:
                continue
            y = h[metric_name]
            if metric_name == "val_acc" and scale_val_acc is not None:
                y = [v * scale_val_acc for v in y]
            y = _ema(y, smooth_ema)
            x = range(1, len(y) + 1)
            label = h.get("model_name", "model")
            plt.plot(x, y, label=f"{label} {metric_name}", linestyle=linestyle, linewidth=2)

    if "train_loss" in present:
        plot_metric("train_loss", "-")
    if "val_loss" in present:
        plot_metric("val_loss", "--")
    if "val_acc" in present:
        plot_metric("val_acc", ":")

    ylabel = "Loss / Acc"
    if "val_acc" in present and (scale_val_acc and scale_val_acc != 1.0):
        ylabel += f"  (val_acc × {scale_val_acc:g})"

    plt.title("Training Curves Across Models")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_history(history: Dict[str, list], out_dir: str, fname: str = "accuracy_vs_epoch.png"):
    epochs = range(1, len(history["train_acc"]) + 1)
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train acc")
    plt.plot(epochs, history["val_acc"], label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs. Epoch")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[saved plot] {path}")


def save_history(history: Dict[str, list], out_dir: str, fname: str = "training_history.json"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    with open(path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"[saved history] {path}")


# =====================
# 8) Dataset & loaders
# =====================


def build_transforms(img_size: int):
    mean, std = MEAN, STD
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 256 / img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tf, eval_tf


class HFDatasetTorch(torch.utils.data.Dataset):
    """Wrapper for HF datasets returning (image, label) pairs."""

    def __init__(self, hf_ds, transform=None, skip_bad=True):
        self.ds = hf_ds
        self.transform = transform
        self.skip_bad = skip_bad

    def __len__(self):
        return self.ds.num_rows

    def __getitem__(self, idx):
        try:
            ex = self.ds[idx]
            img = ex["image"]
            label = ex["label"]

            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))
            if img.mode != "RGB":
                img = img.convert("RGB")

            if self.transform:
                img = self.transform(img)

            return img, label

        except (UnidentifiedImageError, OSError, ValueError):
            if self.skip_bad:
                new_idx = np.random.randint(0, len(self.ds))
                return self.__getitem__(new_idx)
            raise


def make_dataloaders_hf(
    batch_size: int = 32,
    num_workers: int = 0,
    img_size: int = IMAGE_SIZE,
):
    train_tf, eval_tf = build_transforms(img_size)

    hf_train = load_dataset("frgfm/imagenette", "320px", split="train")
    split = hf_train.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
    hf_train, hf_val = split["train"], split["test"]

    train_set = HFDatasetTorch(hf_train, transform=train_tf)
    val_set = HFDatasetTorch(hf_val, transform=eval_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


@torch.no_grad()
def debug_one_batch(model, loader, device):
    images, targets = next(iter(loader))
    images, targets = images.to(device), targets.to(device)
    logits = model(images)
    preds = logits.argmax(1)

    print(
        "debug batch | logits shape:",
        logits.shape,
        "| unique preds:",
        torch.unique(preds).tolist(),
        "| unique targets:",
        torch.unique(targets).tolist(),
    )


# =====================
# 9) Training entrypoints
# =====================


def train_model_vit():
    seed = 42
    epochs = 100
    batch_size = 32
    base_lr = 3e-4
    min_lr = 1e-5
    weight_decay = 0.05
    warmup_epochs = 5
    label_smoothing = 0.1
    num_workers = 0
    out_dir = "./checkpoints_vit"

    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_dataloaders_hf(
        batch_size=batch_size, num_workers=num_workers
    )

    model = vit_tiny(num_classes=NUM_CLASSES).to(device)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.dim() == 1 or name.endswith(".bias") else decay).append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=base_lr,
        betas=(0.9, 0.999),
    )

    loss_fn = LabelSmoothingXEnt(smoothing=label_smoothing)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    step_counter = 0

    def lr_schedule():
        nonlocal step_counter
        lr = cosine_with_warmup(step_counter, total_steps, base_lr, min_lr, warmup_steps)
        step_counter += 1
        return lr

    history = {
        "model_name": "ViT",
        "epoch": [],
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
    }

    debug_one_batch(model, train_loader, device)
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            epoch, model, train_loader, optimizer, loss_fn, scaler, device, lr_schedule
        )
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        t1 = time.time()

        history["epoch"].append(epoch)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss {train_loss:.4f} acc {train_acc:.2f}% | "
            f"val_loss {val_loss:.4f} acc {val_acc:.2f}% | "
            f"{(t1 - t0):.1f}s"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(out_dir, "vit_best.pt")
            torch.save(
                {"model": model.state_dict(), "acc": best_acc, "epoch": epoch},
                ckpt_path,
            )

    torch.save(model.state_dict(), os.path.join(out_dir, "vit_last.pt"))
    save_history(history, out_dir)
    plot_history(history, out_dir)
    print(f"ViT training done. Best val acc: {best_acc:.2f}%")

    return history


def train_model_vit_sc():
    seed = 42
    epochs = 100
    batch_size = 32
    base_lr = 3e-4
    min_lr = 1e-5
    weight_decay = 0.05
    warmup_epochs = 5
    label_smoothing = 0.1
    num_workers = 0
    out_dir = "./checkpoints_vit_sc"

    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_dataloaders_hf(
        batch_size=batch_size, num_workers=num_workers
    )

    model = vit_sc_tiny(num_classes=NUM_CLASSES).to(device)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.dim() == 1 or name.endswith(".bias") else decay).append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=base_lr,
        betas=(0.9, 0.999),
    )

    loss_fn = LabelSmoothingXEnt(smoothing=label_smoothing)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    step_counter = 0

    def lr_schedule():
        nonlocal step_counter
        lr = cosine_with_warmup(step_counter, total_steps, base_lr, min_lr, warmup_steps)
        step_counter += 1
        return lr

    history = {
        "model_name": "ViT_sc",
        "epoch": [],
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
    }

    debug_one_batch(model, train_loader, device)
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            epoch, model, train_loader, optimizer, loss_fn, scaler, device, lr_schedule
        )
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        t1 = time.time()

        history["epoch"].append(epoch)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss {train_loss:.4f} acc {train_acc:.2f}% | "
            f"val_loss {val_loss:.4f} acc {val_acc:.2f}% | "
            f"{(t1 - t0):.1f}s"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(out_dir, "vit_sc_best.pt")
            torch.save(
                {"model": model.state_dict(), "acc": best_acc, "epoch": epoch},
                ckpt_path,
            )

    torch.save(model.state_dict(), os.path.join(out_dir, "vit_sc_last.pt"))
    save_history(history, out_dir)
    plot_history(history, out_dir)
    print(f"ViT_sc training done. Best val acc: {best_acc:.2f}%")

    return history


# ===========
# 10) Main
# ==========

if __name__ == "__main__":
    history_vit = train_model_vit()
    history_vit_sc = train_model_vit_sc()

    plot_histories(
        [history_vit_sc, history_vit],
        save_path=os.path.join(os.getcwd(), "vitc_comparison_result.png"),
    )
