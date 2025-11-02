# ViT_sc_features — Three-Stream ViT with V1 Simple/Complex Features

This repository implements a Vision Transformer with **three parallel streams**:
1) a **raw patch** stream (standard ViT tokens),
2) a **complex-cell** stream derived from odd-phase simple cells and complex pooling,
3) a **simple-cell** stream using a bank of fixed Gabor filters.

Each stream is encoded **independently** with its own Transformer stack and fused with a **late additive merge** before classification.

- **Simple cells**: fixed Gabor filtering + rectification → orientation & phase specific.
- **Complex cells**: magnitude/energy of simple responses (odd-only by default) → polarity-invariant.
- **Three encoders**: parallel Transformer stacks for raw, complex, and simple streams.
- **Late additive fusion**: `LayerNorm(raw + complex + simple)` before the classifier.

---

## ✨ What’s in this version

- **Three-stream architecture** (raw, complex, simple).
- **Shape-compatible tokens** by projecting complex/simple features to `cfg.embed_dim`.
- **Odd-only complex mode** for efficiency; switchable to full quadrature when needed.
- **Fixed Gabor bank** via `register_buffer` to enforce a strong edge/orientation inductive bias.

---

## 🔧 Architecture (High Level)

```
Image (B,C,H,W)
   ├─► PatchEmbed ───────────────► +[CLS], Pos ──► blocks   ──► x_raw
   ├─► MonocularSimpleOddGaborEmbed (odd)
   │     └─► ComplexFromSimplePatches (odd_only, proj→embed_dim)
   │            └─► +[CLS], Pos ──► blocks1 ──► x_comp
   └─► SimpleGaborEmbed (fixed Gabors, proj→embed_dim)
          └─► +[CLS], Pos ──► blocks2 ──► x_simp

Fusion: x = LayerNorm(x_raw + x_comp + x_simp)
Head:   y = Linear(x[:, CLS])
```

---

## 🧩 Components

### SimpleGaborEmbed
- Fixed orientations \\(\\theta \\in [0,\\pi)\\); `conv2d` per channel → `ReLU` → `AvgPool2d(patch_size)`.
- Optional `Linear(n_gabor → embed_dim)` to match ViT width.
- **Output:** `(B, N_patches, cfg.embed_dim)` when `embed_dim=cfg.embed_dim` (recommended for fusion).

### MonocularSimpleOddGaborEmbed → ComplexFromSimplePatches
- Odd-phase simple responses per orientation → complex energy.
- **Odd-only (used here):** \\( y_\\theta \\approx |S_{\\text{odd},\\theta}| = \\sqrt{S_{\\text{odd},\\theta}^2 + \\varepsilon} \\).
- Set `embed_dim_out = cfg.embed_dim` to align with ViT width.
- **Output:** `(B, N_patches, cfg.embed_dim)`.

### Transformer Stacks
- `blocks`  : raw stream encoder (width = `cfg.embed_dim`).
- `blocks1` : complex stream encoder (width = `cfg.embed_dim`).
- `blocks2` : simple stream encoder (width = `cfg.embed_dim`).

---

## 🏗️ Model: `ViT_sc_features` (Essential Flow)

```python
def forward(self, x):                      # x: (B, C, H, W)
    B = x.size(0)

    # --- simple stream ---
    xs = self.simple_gabor_embed(x)                            # (B, N, embed_dim)
    cls = self.cls_token.expand(B, -1, -1)
    xs  = torch.cat([cls, xs], dim=1)
    xs  = xs + self.pos_embed[:, : xs.size(1), :]
    xs  = self.pos_drop(xs)

    # --- complex stream ---
    x1, signed_maps, polarity, ori_idx = self.mono_simple(x)   # (B, N, n_orient)
    xc = self.complex_embed(x1, mode="odd_only")               # (B, N, embed_dim)
    xc = torch.cat([cls, xc], dim=1)
    xc = xc + self.pos_embed[:, : xc.size(1), :]
    xc = self.pos_drop(xc)

    # --- raw stream ---
    x  = self.patch_embed(x)                                   # (B, N, embed_dim)
    x  = torch.cat([cls, x], dim=1)
    x  = x + self.pos_embed[:, : x.size(1), :]
    x  = self.pos_drop(x)

    # --- parallel encoders ---
    for blk, blk1, blk2 in zip(self.blocks, self.blocks1, self.blocks2):
        x  = blk(x)
        xc = blk1(xc)
        xs = blk2(xs)   # NOTE: use blk2 for the simple stream

    # --- late fusion + head ---
    x = self.norm(x + xc + xs)
    cls_out = x[:, 0]
    return self.head(cls_out)
```
> **Note:** If your code currently uses `xs = blk1(xs)`, update it to `xs = blk2(xs)` to ensure the simple stream uses its own stack.

---

## 🔢 Shapes & Notation

- **Input image:** `(B, C, H, W)` with `H = W = cfg.img_size`  
- **Patches:** `N_patches = (H / cfg.patch_size)^2`  
- **Token width:** all three streams use `cfg.embed_dim`; fusion is elementwise.

---

## ⚙️ Configuration (`ViTConfig` fields)

| Field | Meaning |
|---|---|
| `img_size`, `patch_size`, `in_chans` | Image shape & tokenization. |
| `gabor` | Number of Gabor orientations (`n_orient`). |
| `part_embed_dim` | Dim for simple/complex inner computations (e.g., `mono_simple`), separate from fusion width. |
| `embed_dim` | **Shared token width** for all three streams and the ViT blocks. |
| `num_heads`, `mlp_ratio`, `depth`, `dropout` | ViT hyperparameters (applied to all three stacks). |
| `num_classes` | Classifier output size. |

---

## 🔍 Design Rationale

- **Modality-preserving**: Each stream keeps its own inductive bias (raw pixels vs V1 energy vs rectified Gabor) before interaction.
- **Late additive fusion**: Stable, parameter-free; relies on width alignment (`embed_dim`).
- **Odd-only complex**: Halves compute vs quadrature while maintaining strong edge sensitivity. Upgrade to full quadrature if invariance to small shifts is critical.

---

## 🧠 Training Tips

- Ensure **all three outputs** are in the same width (`embed_dim`) before fusion.
- Start with **odd-only** for speed; benchmark quadrature later.
- If VRAM is tight: reduce `depth` of auxiliary streams (`blocks1`, `blocks2`) or tie weights.
- Keep augmentations identical across streams since they share the same input image.

---

## 📦 Minimal Usage

```python
cfg = ViTConfig(
    img_size=32, patch_size=4, in_chans=3,
    gabor=8, part_embed_dim=16,
    embed_dim=256, num_heads=8, mlp_ratio=4.0, depth=6,
    dropout=0.1, num_classes=10
)

model = ViT_sc_features(cfg)
logits = model(images)   # (B, 3, 32, 32) -> (B, num_classes)
```

---

## 📚 Background
- Hubel & Wiesel (1962): V1 simple/complex cells and orientation tuning.  
- Energy model of complex cells: phase-invariant quadrature pooling.  
- **Dosovitskiy et al. (2020)** – *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.*  
  Introduced the Vision Transformer (ViT) architecture, which our model extends by integrating biologically inspired early vision mechanisms.
---

## License

MIT (or your preferred license).

---

## Citation

If you use this code, please cite this repository and relevant V1 literature.


---

## Citation

If you use this code, please cite this repository and relevant V1 literature.

