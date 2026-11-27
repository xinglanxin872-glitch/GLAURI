# Input ablations

All current variants use the config-only trainer and geometry_mask_v2.
See [model variants](../model/VARIANTS.md) and
[training decision nodes](../model/TRAINING_DECISIONS.md) for the exact policy.

w/o struct retains one-hot/ESM and deterministic sequence adjacency. Geometric
scalars and vectors, edge features, GCP-internal RBF, local frames and RSA are
zeroed; coordinate noise is disabled.

Historical partial-input-ablation checkpoints have different semantics. Their
old flag must not be substituted for geometry_mask_v2 in new training.
