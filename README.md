# ViT_sc_features — V1 Simple/Complex Cells as Parallel Features for ViT

This repository implements a Vision Transformer front-end that injects **biologically inspired V1 features**. We compute **phase-sensitive (Simple)** and **phase-invariant (Complex)** orientation responses and feed them into a **parallel twin-stream encoder**: one stream processes **raw ViT patch tokens**, the other processes **complex-cell tokens** derived from odd-phase Gabor simple cells. The two streams are encoded **independently** and **fused late** by a parameter-free additive merge before classification.

- **Simple cells**: fixed Gabor filtering + rectification → orientation & phase specific.
- **Complex cells**: magnitude/energy of simple responses → phase/contrast polarity invariant.
- **Twin encoders**: independent Transformer stacks for raw and complex streams.
- **Late additive fusion**: `LayerNorm(raw + complex)` before the classifier.

---

## ✨ Highlights

- **Fixed Gabor bank** (strong inductive bias, fewer params, stable early training).
- **Patch-aligned features**: all streams output `(B, N_patches, D)`; easy to concatenate or fuse.
- **Odd-only complex** mode for **cheap phase invariance**; full **even+odd quadrature** is a drop-in change.
- Clean separation of components for ablations and extensions.

---

## 🔧 Architecture (High Level)

```
Image (B,C,H,W)
   ├─► PatchEmbed ──► +[CLS], Pos ──► blocks   ──► x_raw
   └─► MonocularSimpleOddGaborEmbed (odd) ──► ComplexFromSimplePatches (odd_only, proj→embed_dim)
                      └─► +[CLS], Pos ──► blocks1 ──► x_comp

Fusion: x = LayerNorm(x_raw + x_comp)
Head:   y = Linear(x[:, CLS])
```

---

## 🧩 Components

### SimpleGaborEmbed
Fixed, evenly spaced orientations \\(\\theta \\in [0,\\pi)\\).  
`conv2d` (per channel) → `ReLU` → `AvgPool2d(patch_size)` → optional `Linear(n_gabor → embed_dim)`  
**Output:** `(B, N_patches, embed_dim)`

### MonocularSimpleOddGaborEmbed
Patchwise **odd-phase** Gabor responses per orientation (odd ≈ spatial derivative).  
**Output:** `(B, N_patches, n_orient)` plus auxiliary maps (`signed_maps`, `polarity`, `ori_idx`).

### ComplexFromSimplePatches
Computes complex-cell responses per patch.  
- **Odd-only (used here):** \\( y_\\theta \\approx |S_{\\text{odd},\\theta}| = \\sqrt{S_{\\text{odd},\\theta}^2 + \\varepsilon} \\).  
- **Quadrature (optional):** \\( \\sqrt{S_e^2 + S_o^2 + \\varepsilon} \\) using even+odd pairs.  
- With `embed_dim_out = cfg.embed_dim` to match ViT width.  
**Output:** `(B, N_patches, cfg.embed_dim)`


### ViT Encoder Blocks
Two independent stacks (`blocks`, `blocks1`) of standard Transformer encoder blocks (same width `cfg.embed_dim`).

---

## 🏗️ Model: `ViT_sc_features` (Essential Flow)

```python
def forward(self, x):                      # x: (B, C, H, W)
    B = x.size(0)

    # Complex stream (V1 features)
    x1, signed_maps, polarity, ori_idx = self.mono_simple(x)        # (B,N,n_orient)
    xc = self.complex_embed(x1, mode="odd_only")                    # (B,N,embed_dim)

    cls = self.cls_token.expand(B, -1, -1)
    xc  = torch.cat([cls, xc], dim=1)
    xc  = xc + self.pos_embed[:, : xc.size(1), :]
    xc  = self.pos_drop(xc)

    # Raw stream
    x   = self.patch_embed(x)                                       # (B,N,embed_dim)
    x   = torch.cat([cls, x], dim=1)
    x   = x + self.pos_embed[:, : x.size(1), :]
    x   = self.pos_drop(x)

    # Twin encoders
    for blk, blk1 in zip(self.blocks, self.blocks1):
        x  = blk(x)
        xc = blk1(xc)

    # Late fusion + head
    x = self.norm(x + xc)
    cls_out = x[:, 0]
    return self.head(cls_out)
```

> `SimpleGaborEmbed` is computed as `s = self.simple_gabor_embed(x)` for ablation or auxiliary objectives; it is not fused by default in the snippet above.

---

## 🔢 Shapes & Notation

- **Input image:** `(B, C, H, W)` with `H = W = cfg.img_size`  
- **Patches:** `N_patches = (H / cfg.patch_size)^2`  
- **Token width:** both streams use `cfg.embed_dim`; fusion is elementwise.

---

## ⚙️ Configuration (`ViTConfig` fields)

| Field | Meaning |
|---|---|
| `img_size`, `patch_size`, `in_chans` | Image shape & tokenization. |
| `gabor` | Number of Gabor orientations (`n_orient`). |
| `part_embed_dim` | Dim for simple/complex branches (kept for ablations). |
| `embed_dim` | **ViT width** and complex stream projection width. |
| `num_heads`, `mlp_ratio`, `depth`, `dropout` | ViT hyperparameters for both stacks. |
| `num_classes` | Classifier output classes. |

---

## 🔍 Design Rationale

- **Independent encoding** preserves modality-specific inductive biases (raw vs V1 energy) before interaction.
- **Late additive fusion** is stable and parameter-free; aligns shapes by projecting complex tokens to `embed_dim`.
- **Odd-only complex** halves compute vs full quadrature yet keeps strong edge sensitivity. Quadrature is a drop-in replacement when needed.

---

## 🧪 Ablations & Extensions

1. **Use `SimpleGaborEmbed` in fusion:** `torch.cat([tokens, s], dim=-1)` with a bottleneck MLP back to `embed_dim`.  
2. **Quadrature complex:** call `complex_embed((even, odd))` or supply concatenated/interleaved even+odd.  
3. **Gated fusion:** `x + σ(g) ⊙ xc` with a learnable gate `g` (scalar/vector/token).  
4. **Periodic fusion:** fuse every `k` layers instead of only once at the end.  
5. **Separate positional embeddings:** have `pos_embed_raw` and `pos_embed_cmp` if distributions differ.

---

## 🧠 Training Tips

- Start with **odd_only** for speed; test quadrature for robustness.  
- If VRAM is tight: reduce `depth` of the complex stream or share weights across some layers.  
- Keep augmentation identical across streams since both derive from the same input.  
- Tune `ksize/sigma/lam` to dataset scale; increase `ksize` for higher-resolution images.

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

- Hubel & Wiesel (1962): orientation-selective simple/complex cells in V1.  
- Energy model of complex cells: phase-invariant quadrature pooling.

---

## License

MIT (or your preferred license).

---

## Citation

If you use this code, please cite this repository and relevant V1 literature.

