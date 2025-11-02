# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import math, time, os, random
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import List, Dict, Optional

import json
import matplotlib.pyplot as plt
# ---------------------------
# 1) Reusable ViT definition
# ---------------------------

class MSA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
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
    def __init__(self, dim, mlp_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.msa = MSA(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.msa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=256):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):                 # (B, C, H, W)
        x = self.proj(x)                  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x





def make_gabor_kernel(theta, ksize=15, sigma=3.0, lam=8.0, phase=0.0):
    """Return a 2D Gabor kernel as a torch tensor."""
    ax = torch.arange(ksize) - (ksize - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing='xy')
    Xr =  xx * math.cos(theta) + yy * math.sin(theta)
    Yr = -xx * math.sin(theta) + yy * math.cos(theta)
    env = torch.exp(-(Xr**2 + Yr**2) / (2 * sigma**2))
    carrier = torch.cos(2 * math.pi * Xr / lam + phase)
    kernel = env * carrier
    kernel -= kernel.mean()  # zero-mean
    kernel /= kernel.abs().sum() + 1e-6  # normalize energy
    return kernel

class SimpleGaborEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, n_gabor=8,
                 ksize=15, sigma=3.0, lam=8.0, embed_dim=None):
        """
        Similar to PatchEmbed but uses fixed Gabor filters.
        Args:
            img_size: input image size (H=W)
            patch_size: stride for patch extraction
            in_chans: input channels (1=gray, 3=RGB)
            n_gabor: number of Gabor orientations
            embed_dim: optional linear projection dimension (default = n_gabor)
        """
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.in_chans = in_chans
        self.n_gabor = n_gabor
        self.embed_dim = embed_dim or n_gabor

        # ---- create fixed Gabor filter bank ----
      #  thetas = torch.linspace(0, math.pi, n_gabor, endpoint=False)
        thetas = torch.tensor(np.linspace(0, math.pi, n_gabor, endpoint=False), dtype=torch.float32)

        gabor_kernels = torch.stack([make_gabor_kernel(t, ksize, sigma, lam) for t in thetas])
        # shape (n_gabor, ksize, ksize)

        # Each Gabor acts on each input channel independently
        weight = gabor_kernels.unsqueeze(1).repeat(1, in_chans, 1, 1)
        self.register_buffer('weight', weight)  # not learnable

        self.bias = None#nn.Parameter(torch.zeros(n_gabor))
        self.pool = nn.AvgPool2d(patch_size, stride=patch_size)

        # Optional linear projection to embed_dim
        self.proj = None
        if embed_dim != n_gabor:
            self.proj = nn.Linear(n_gabor, embed_dim)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, N_patches, embed_dim)
        """
        # Gabor convolution
        y = F.conv2d(x, self.weight, self.bias, padding='same')

        y = torch.relu(y)  # rectify like complex-cell response

        # Downsample to patch grid
        y = self.pool(y)  # (B, n_gabor, H/P, W/P)

        # Flatten patches
        B, D, Hp, Wp = y.shape
        y = y.flatten(2).transpose(1, 2)  # (B, N_patches, D)

        # Optional projection
        if self.proj is not None:
            y = self.proj(y)  # (B, N_patches, embed_dim)

        return y


def _gabor_even_odd(theta, ksize=15, sigma=3.0, lam=8.0):
    """
    Returns a pair (even, odd) Gabor kernels as torch.FloatTensors of shape (ksize, ksize).
    - even: cosine carrier
    - odd : sine   carrier
    """
    ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing='xy')
    Xr =  xx * math.cos(theta) + yy * math.sin(theta)
    Yr = -xx * math.sin(theta) + yy * math.cos(theta)

    env = torch.exp(-(Xr**2 + Yr**2) / (2 * sigma**2))
    even = env * torch.cos(2 * math.pi * Xr / lam)   # phase = 0
    odd  = env * torch.sin(2 * math.pi * Xr / lam)   # phase = pi/2

    # zero-mean & energy normalization (helps stability)
    for k in (even, odd):
        k -= k.mean()
        k /= (k.abs().sum() + 1e-6)
    return even, odd



def gabor_odd(theta, ksize=15, sigma=3.0, lam=8.0):
    """Odd-phase (sine) Gabor: phase-sensitive edge detector (simple cell)."""
    ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing='xy')
    ct, st = math.cos(float(theta)), math.sin(float(theta))
    Xr =  xx * ct + yy * st
    Yr = -xx * st + yy * ct
    env = torch.exp(-(Xr**2 + Yr**2) / (2 * sigma**2))
    ker = env * torch.sin(2 * math.pi * Xr / lam)   # odd phase
    ker -= ker.mean()
    ker /= (ker.abs().sum() + 1e-6)
    return ker

class MonocularSimpleOddGaborEmbed(nn.Module):
    """
    (B, C, H, W) -> (B, N_patches, n_gabor)
    - Odd-phase Gabors (simple-cell, phase-sensitive)
    - Signed outputs (no ReLU) to keep light→dark / dark→light polarity

    Usage:
    emb, signed_maps, polarity, ori_idx = self.mono(x)

    emb.shape         # (B, N_patches, n_gabor)
    signed_maps.shape # (B, n_gabor, H, W) per-orientation signed responses
    polarity.shape    # (B, H, W)   +1/-1 polarity map
    ori_idx.shape     # (B, H, W)   strongest-orientation index
    """
    def __init__(self, img_size=32, patch_size=4, in_chans=3, n_gabor=8,
                 ksize=15, sigma=3.0, lam=8.0, embed_dim=None,
                 combine_channels="sum"):  # 'sum' or 'mean'
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
        self.combine_channels = combine_channels

        # orientations in [0, pi)
        thetas = torch.tensor(np.linspace(0, math.pi, n_gabor, endpoint=False), dtype=torch.float32)

        # Build fixed odd Gabor bank: shape (n_gabor, k, k)
        gabor_kernels = torch.stack([gabor_odd(t, ksize, sigma, lam) for t in thetas])

        # Weight to span color channels jointly: (n_gabor, in_chans, k, k)
        weight = gabor_kernels.unsqueeze(1).repeat(1, in_chans, 1, 1)
        self.register_buffer('weight', weight)   # fixed (no grad)
        self.bias = None                         # keep strictly linear

        self.pool = nn.AvgPool2d(patch_size, stride=patch_size)

        self.proj = None
        if self.embed_dim != n_gabor:
            self.proj = nn.Linear(n_gabor, self.embed_dim)

    def forward(self, x):
        """
        x: (B, C, H, W)  (RGB or gray). Channels are combined linearly by the filters.
        Returns:
          y_embed: (B, N_patches, embed_dim)    # patch embeddings (signed)
          signed_maps: (B, n_gabor, H, W)       # per-orientation signed responses
          polarity: (B, H, W) in {+1, 0, -1}    # sign of strongest-orientation response
          ori_idx:  (B, H, W)                   # index of strongest orientation
        """
        # Simple-cell linear filtering (keep SIGN; no ReLU)
        y = F.conv2d(x, self.weight, bias=self.bias, padding=self.pad)   # (B, n_gabor, H, W)

        # Find strongest orientation per pixel (by magnitude), then take its sign → polarity
        mag = y.abs()                                 # (B, n_gabor, H, W)
        ori_idx = mag.argmax(dim=1)                   # (B, H, W)
        B, _, H, W = y.shape
        gather_idx = ori_idx.unsqueeze(1)             # (B,1,H,W)
        sel = y.gather(1, gather_idx).squeeze(1)      # (B,H,W) signed response at best orientation
        polarity = torch.sign(sel)                    # +1 light→dark, -1 dark→light, 0 near-zero

        # Pool to patch grid -> (B, n_gabor, H/P, W/P)
        yp = self.pool(y)

        # Flatten to (B, N_patches, n_gabor)
        B, D, Hp, Wp = yp.shape
        y_embed = yp.flatten(2).transpose(1, 2)
        
        # Optional projection
     #   if self.proj is not None:
     #       with torch.no_grad():
    
     #           y_embed = self.proj(y_embed)

        return y_embed, y, polarity, ori_idx



class ComplexFromSimplePatches(nn.Module):
    """
    Patchwise complex cells from simple-cell embeddings.

    Inputs (choose ONE of the following formats):
      - forward((even, odd))                       # tuple: both (B, N_patches, n_orient)
      - forward(x, input_layout='concat')          # x: (B, N_patches, 2*n_orient) = [even, odd]
      - forward(x, input_layout='interleave')      # x: (B, N_patches, 2*n_orient) = e0,o0,e1,o1,...
      - forward(x, mode='odd_only')                # x: (B, N_patches, n_orient)   = odd only (fallback)

    Output:
      - (B, N_patches, embed_dim_out)
    """
    def __init__(self, n_orient, embed_dim_out=None, eps=1e-8):
        super().__init__()
        self.n_orient = n_orient
        self.eps = eps
        self.embed_dim_out = embed_dim_out or n_orient
        self.proj = None
        if self.embed_dim_out != n_orient:
            self.proj = nn.Linear(n_orient, self.embed_dim_out)

    def _energy(self, even, odd):
        # even/odd: (B, N, n_orient)
        energy = torch.sqrt(even * even + odd * odd + self.eps)
        return energy  # (B, N, n_orient)

    def forward(self, x, input_layout=None, mode=None):
        """
        x:
          - tuple: (even, odd), each (B, N, n_orient)
          - tensor: (B, N, 2*n_orient) with input_layout in {'concat','interleave'}
          - tensor: (B, N, n_orient) with mode='odd_only'
        """
        if isinstance(x, tuple):
            even, odd = x
            B, N, D = even.shape
            assert odd.shape == even.shape, "even and odd must match shapes"
            assert D == self.n_orient, f"last dim must be n_orient={self.n_orient}"
            y = self._energy(even, odd)

        else:
            B, N, D = x.shape
            if mode == 'odd_only':
                # fallback: |odd| ≈ sqrt(odd^2) (NOT true quadrature)
                assert D == self.n_orient, f"odd_only expects last dim == n_orient={self.n_orient}"
                y = torch.sqrt(x * x + self.eps)
            else:
                assert input_layout in {'concat','interleave'}, \
                    "Specify input_layout in {'concat','interleave'} or use mode='odd_only'"

                assert D == 2 * self.n_orient, \
                    f"{input_layout} expects last dim == 2*n_orient={2*self.n_orient}"

                if input_layout == 'concat':
                    even = x[:, :, :self.n_orient]
                    odd  = x[:, :, self.n_orient:]
                else:  # interleave
                    even = x[:, :, 0::2]
                    odd  = x[:, :, 1::2]

                y = self._energy(even, odd)  # (B, N, n_orient)

        if self.proj is not None:
            y = self.proj(y)  # (B, N, embed_dim_out)
        return y



class ComplexGaborEmbed(nn.Module):
    """
    Complex cell–like embedding:
      (B, C, H, W) -> (B, N_patches, embed_dim)

    Uses quadrature Gabors (even+odd) to form phase-invariant energy per orientation,
    then averages across color channels, pools to patch grid, and (optionally) projects.
    """
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_chans=3,
        n_orient=8,
        ksize=15,
        sigma=3.0,
        lam=8.0,
        embed_dim=None,
        pool_type="avg",      # 'avg' or 'max'
        aggregate_channels="mean"  # 'mean' or 'sum'
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        assert ksize % 2 == 1, "ksize should be odd so padding=ksize//2 is 'same'"

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.n_orient = n_orient
        self.ksize = ksize
        self.aggregate_channels = aggregate_channels
        self.num_patches = (img_size // patch_size) ** 2

        # Equally spaced orientations in [0, pi)
        # (Use torch.arange to avoid endpoint issues across PyTorch versions)
        thetas = torch.arange(n_orient, dtype=torch.float32) * (math.pi / n_orient)

        # Build filter banks (even & odd), then replicate per input channel for grouped conv
        even_list, odd_list = [], []
        for t in thetas:
            e, o = _gabor_even_odd(float(t), ksize=ksize, sigma=sigma, lam=lam)
            even_list.append(e)
            odd_list.append(o)

        even = torch.stack(even_list, dim=0)  # (n_orient, k, k)
        odd  = torch.stack(odd_list,  dim=0)  # (n_orient, k, k)

        # For grouped conv with groups=in_chans:
        #   weight shape must be (out_channels, 1, k, k)
        #   where out_channels is multiple of in_chans.
        # We create n_orient filters PER channel => out_channels = n_orient * in_chans.
        even = even.unsqueeze(1)  # (n_orient, 1, k, k)
        odd  = odd.unsqueeze(1)   # (n_orient, 1, k, k)
        even = even.repeat(in_chans, 1, 1, 1)  # (n_orient*in_chans, 1, k, k)
        odd  = odd.repeat(in_chans, 1, 1, 1)   # (n_orient*in_chans, 1, k, k)

        self.register_buffer("weight_even", even)  # non-learnable (fixed)
        self.register_buffer("weight_odd",  odd)   # non-learnable (fixed)

        # no bias for quadrature energy; keep everything symmetric
        self.padding = ksize // 2

        # pooling to patch grid
        if pool_type == "avg":
            self.pool = nn.AvgPool2d(patch_size, stride=patch_size)
        elif pool_type == "max":
            self.pool = nn.MaxPool2d(patch_size, stride=patch_size)
        else:
            raise ValueError("pool_type must be 'avg' or 'max'")

        # optional linear projection to embed_dim
        self.embed_dim = embed_dim or n_orient
        self.proj = None
        if self.embed_dim != n_orient:
            self.proj = nn.Linear(n_orient, self.embed_dim)

        # small epsilon for stable sqrt
        self.register_buffer("eps", torch.tensor(1e-8, dtype=torch.float32))

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, N_patches, embed_dim)
        """
        B, C, H, W = x.shape
        assert C == self.in_chans, f"Expected {self.in_chans} channels, got {C}"

        # Convolve per channel with groups=in_chans
        # Outputs: (B, n_orient*in_chans, H, W)
        e = F.conv2d(x, self.weight_even, bias=None, padding=self.padding, groups=self.in_chans)
        o = F.conv2d(x, self.weight_odd,  bias=None, padding=self.padding, groups=self.in_chans)

        # Quadrature energy (phase-invariant)
        energy = torch.sqrt(e * e + o * o + self.eps)

        # Reshape to (B, in_chans, n_orient, H, W) then aggregate across channels
        energy = energy.view(B, self.in_chans, self.n_orient, H, W)
        if self.aggregate_channels == "mean":
            energy = energy.mean(dim=1)   # (B, n_orient, H, W)
        elif self.aggregate_channels == "sum":
            energy = energy.sum(dim=1)
        else:
            raise ValueError("aggregate_channels must be 'mean' or 'sum'")

        # Pool to patch grid: (B, n_orient, H/P, W/P)
        y = self.pool(energy)

        # Flatten to (B, N_patches, n_orient)
        B, D, Hp, Wp = y.shape
        y = y.flatten(2).transpose(1, 2)  # (B, Hp*Wp, D)

        # Optional projection to embed_dim
        if self.proj is not None:
            y = self.proj(y)              # (B, N_patches, embed_dim)

        return y


@dataclass
class ViTConfig:
    img_size: int = 32
    patch_size: int = 4           # 32/4 -> 8x8 = 64 tokens
    in_chans: int = 3
    part_embed_dim: int = 256
    embed_dim: int =  part_embed_dim*2
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    num_classes: int = 10
    gabor:int=8

class ViT_sc_features(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg.img_size, cfg.patch_size, cfg.in_chans,int(cfg.embed_dim))

        self.simple_gabor_embed = SimpleGaborEmbed(img_size=cfg.img_size, 
                                      patch_size=cfg.patch_size, 
                                      in_chans=cfg.in_chans, 
                                      n_gabor=cfg.gabor,
                                      embed_dim=cfg.embed_dim
                                      
                                      )


        
        self.mono_simple = MonocularSimpleOddGaborEmbed(img_size=cfg.img_size, 
                                      patch_size=cfg.patch_size, 
                                      in_chans=cfg.in_chans, 
                                      n_gabor=cfg.gabor,
                                      embed_dim=cfg.part_embed_dim,
                                            ksize=15, sigma=3.0, lam=8.0,
                                    )
        self.complex_embed = ComplexFromSimplePatches(n_orient=cfg.gabor, embed_dim_out=cfg.embed_dim,)
   
        
        self.alpha_c = torch.nn.Parameter(torch.ones((int(cfg.part_embed_dim))))
        self.alpha_s= torch.nn.Parameter(torch.ones((int(cfg.part_embed_dim))))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, cfg.embed_dim))
        self.pos_drop = nn.Dropout(cfg.dropout)

        mlp_dim = int(cfg.embed_dim * cfg.mlp_ratio)
        self.blocks = nn.ModuleList([
            EncoderBlock(cfg.embed_dim, cfg.num_heads, mlp_dim, dropout=cfg.dropout)
            for _ in range(cfg.depth)
        ])
        self.blocks1 = nn.ModuleList([
            EncoderBlock(cfg.embed_dim, cfg.num_heads, mlp_dim, dropout=cfg.dropout)
            for _ in range(cfg.depth)
        ])
        self.blocks2 = nn.ModuleList([
            EncoderBlock(cfg.embed_dim, cfg.num_heads, mlp_dim, dropout=cfg.dropout)
            for _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):                          # x: (B, C, H, W)
        B = x.shape[0]
        xs= self.simple_gabor_embed(x)
        cls = self.cls_token.expand(B, -1, -1)     # (B, 1, D)

        xs = torch.cat([cls, xs], dim=1)             # prepend [CLS]
        xs = xs + self.pos_embed[:, : xs.size(1), :]  # add pos embed
        xs = self.pos_drop(xs)
     

        
        x1, signed_maps, polarity, ori_idx = self.mono_simple(x)     
        xc= self.complex_embed(x1,mode="odd_only")
        #  x= x2*self.alpha_c
      #  x = torch.cat([x, s*self.alpha_s, c*self.alpha_c], dim=-1) 
      #  x = torch.cat([x,c], dim=-1) 

        cls = self.cls_token.expand(B, -1, -1)     # (B, 1, D)
        xc = torch.cat([cls, xc], dim=1)             # prepend [CLS]
        xc = xc + self.pos_embed[:, : xc.size(1), :]  # add pos embed
        xc = self.pos_drop(xc)

        


        
        x = self.patch_embed(x)                    # (B, N, D) 
        cls = self.cls_token.expand(B, -1, -1)     # (B, 1, D)

        x = torch.cat([cls, x], dim=1)             # prepend [CLS]
        x = x + self.pos_embed[:, : x.size(1), :]  # add pos embed
        x = self.pos_drop(x)
        for blk, blk1, blk2 in zip(self.blocks, self.blocks1, self.blocks2):
            x = blk(x)
            xc= blk1(xc)
            xs= blk1(xs)
        x = self.norm(x+xc+xs)
        cls_out = x[:, 0]
        return self.head(cls_out)



def vitc_ti_4(num_classes=10):
    # Tiny-ish ViT suited for CIFAR-10 resolution (32x32)#ViT_sc_features
    return ViT_sc_features(ViTConfig(
        img_size=32, patch_size=4,
         depth=6, num_heads=8,
        num_classes=num_classes, mlp_ratio=4.0
    ))
  


def vitc_h14(num_classes=10):
    # "Huge" commonly uses patch_size=14 in many open-source releases
    return ViT_sc_features(ViTConfig(img_size=32, embed_dim=1280, depth=32, num_heads=16, patch_size=16, num_classes=num_classes,mlp_ratio=4.0))

# ---------------------------------
# 2) Utilities: metrics & schedules
# ---------------------------------


def iter_regularized_params(model):
    """
    Yields parameters suitable for weight regularization.
    Skips biases and normalization parameters (like LayerNorm, BatchNorm).
    """
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Skip biases and norm layers
        if name.endswith("bias") or isinstance(getattr(model, name.split('.')[0], None), (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            continue
        yield param


def weight_regularization(model, reg_type="l2", scale=1.0):
    """
    Computes L1 or L2 weight regularization for Transformer models.
    Args:
        model: nn.Module (Transformer or submodule)
        reg_type: "l1" or "l2"
        scale: scaling coefficient for the regularization term
    Returns:
        scalar Tensor regularization loss
    """
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

    def forward(self, pred, target):
        # pred: (B, K) logits; target: (B,)
        n_classes = pred.size(-1)
        logprobs = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logprobs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * logprobs, dim=-1))

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
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

# ---------------
# 3) Train / Eval
# ---------------

def train_one_epoch(epoch, model, loader, optimizer, loss_fn, scaler, device, 
                    lr_schedule=None, 
                    weight_reg_lambda: float = 1e-4 # strength
    ):
    model.train()
    total_loss, total_top1 = 0.0, 0.0
    n = 0
    with tqdm(total=len(loader), desc=f"Train model. Epochs: {epoch}") as pbar:

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
                loss= ce+ weight_reg_lambda*wreg
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
            pbar.set_postfix(loss= f'{(total_loss/n):.4f}')

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

# --------------
# 4) Main script
# --------------

def plot_histories(
    histories: List[Dict[str, list]],
    save_path: Optional[str] = None,
    show: bool = True,
    scale_val_acc: float = 10.0,
    smooth_ema: Optional[float] = None,  # e.g., 0.9 for EMA smoothing
):
    """
    Plot multiple training histories on a single figure.

    Args:
        histories: list of history dicts, e.g. each like:
                   {"train_loss": [...], "val_loss": [...], "val_acc": [...]}
        save_path: optional path to save the PNG (e.g., 'runs/curve_multi.png').
        show: if True, plt.show() at the end.
        scale_val_acc: multiply val_acc by this factor so lines share the y-axis (set to 1.0 to disable).
        smooth_ema: optional EMA factor in [0,1). If set (e.g., 0.9), applies EMA smoothing to each series.
    """
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

    # Determine which metrics exist overall
    keys = {"train_loss", "val_loss", "val_acc", "wreg", "treg"}
    present = {k for k in keys if any(k in h and h[k] for h in histories)}

    if not present:
        raise ValueError("No known metrics found in provided histories.")

    # Create one figure (single axes) as requested
    plt.figure(figsize=(9, 6))

    # Helper to plot a single metric for all histories
    def plot_metric(metric_name: str, linestyle: str):
        for h in histories:
            if metric_name not in h or len(h[metric_name]) == 0:
                continue
            y = h[metric_name]
            if metric_name == "val_acc" and scale_val_acc is not None:
                y = [v * scale_val_acc for v in y]
            y = _ema(y, smooth_ema)
            x = range(1, len(y) + 1)
            lab= history['model_name']
            plt.plot(x, y, label=f"{lab} {metric_name}", linestyle=linestyle, linewidth=2)

    # Plot common metrics with different linestyles for clarity
    if "train_loss" in present:
        plot_metric("train_loss", "-")
    if "val_loss" in present:
        plot_metric("val_loss", "--")
    if "val_acc" in present:
        plot_metric("val_acc", ":")

    # Optional regs if they exist
    if "wreg" in present:
        plot_metric("wreg", "-.")
    if "treg" in present:
        plot_metric("treg", (0, (3, 1, 1, 1)))  # dash-dot-dot

    # Labels / legend / grid
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

def plot_history(history, out_dir, fname="accuracy_vs_epoch.png"):
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

def save_history(history, out_dir, fname="training_history.json"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    with open(path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"[saved history] {path}")




def train_model_vitc():
    # ---- Config ----
    seed = 42
    epochs = 200
    batch_size = 64 
    base_lr = 3e-4
    min_lr = 1e-5
    weight_decay = 0.05
    warmup_epochs = 5
    label_smoothing = 0.1
    num_workers = 0
    out_dir = "./checkpoints_vit_cifar10"
    os.makedirs(out_dir, exist_ok=True)

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Data ----
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    # ---- Model ----
    model = vitc_ti_4(num_classes=10).to(device)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.dim() == 1 or name.endswith(".bias") else decay).append(p)

    optim = torch.optim.AdamW([
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=base_lr, betas=(0.9, 0.999))

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

    # ---- NEW: track history ----
    history = {"model_name":"vitc","epoch": [], "train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(epoch, model, train_loader, optim, loss_fn, scaler, device, lr_schedule)
        val_loss, val_acc = evaluate(model, test_loader, loss_fn, device)
        t1 = time.time()

        history["epoch"].append(epoch)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:03d}/{epochs} | "
              f"train_loss {train_loss:.4f} acc {train_acc:.2f}% | "
              f"val_loss {val_loss:.4f} acc {val_acc:.2f}% | "
              f"{(t1 - t0):.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(out_dir, "vit_cifar10_best.pt")
            torch.save({
                "model": model.state_dict(),
                "acc": best_acc,
                "epoch": epoch
            }, ckpt_path)

    # ---- Save model, plot, and history ----
    torch.save(model.state_dict(), os.path.join(out_dir, "vit_cifar10_last.pt"))
    save_history(history, out_dir)
    plot_history(history, out_dir)
    print(f"Done. Best val acc: {best_acc:.2f}%")

    return history





