# V1-Simple-Complex-Cells-as-Parallel-Features-for-ViT

This variant integrates biologically inspired V1 features into a Vision Transformer using a parallel twin-stream encoder. One stream processes raw patch tokens; the other processes complex-cell tokens derived from Gabor simple cells (odd phase) and a complex pooling stage. The two streams are encoded independently and summed before the classifier head.

✨ What’s new in this version

Twin streams:

Raw stream: standard PatchEmbed → Transformer blocks (blocks).

Complex stream: MonocularSimpleOddGaborEmbed → ComplexFromSimplePatches(odd_only) → Transformer blocks (blocks1).

Late additive fusion: After layerwise independent encoding, we sum the two sequences and apply a shared LayerNorm and head.

Shape-compatible complex tokens: ComplexFromSimplePatches(embed_dim_out=cfg.embed_dim) ensures both streams have the same token dimension for clean fusion.

SimpleGaborEmbed available for ablation: computed as s = SimpleGaborEmbed(...) (not used by default here, but kept for experiments).

🔧 High-Level Flow
Image (B,C,H,W)
   ├─► PatchEmbed ──► +[CLS], Pos ──► blocks   ──► x_raw
   └─► MonocularSimpleOddGaborEmbed (odd) ──► ComplexFromSimplePatches (odd_only)
                      └─► +[CLS], Pos ──► blocks1 ──► x_comp

Fusion: x = LayerNorm(x_raw + x_comp)
Head:   y = Linear(x[:, CLS])



