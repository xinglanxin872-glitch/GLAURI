# Training flow

Configuration → data and annotations → sampling → input features → model and loss
→ optimization → validation → checkpoint selection.

| Stage | Decision |
| --- | --- |
| Configuration | Merge common settings, defaults and the selected [model configuration](VARIANTS.md); resolve local data paths. |
| Data | Load the released splits, [training targets](../colabel/README.md#training-targets) and Cα alignment masks. |
| Sampling | Apply structure-set weighting, or the corresponding weighting ablation. |
| Inputs | Use the selected model's features. w/o struct removes structural information, including RSA, while retaining sequence adjacency. |
| Loss | Optimize the active interface tasks using their configured class/task weights. Single-task models optimize only their matching task. |
| Optimization | Apply the configured seed, Adam learning rate, epoch count and gradient clipping. |
| Validation | Evaluate a fixed H5 validation subset and compute the mean AUC across active tasks. |
| Checkpoints | Save best-AUC and last weights; ties in best AUC retain the latest checkpoint. |

Each training run writes its effective configuration and `parameter_decisions.json`
for the complete parameter-to-runtime mapping. Parameter values remain in the
[common configuration](config_fixed.json), [defaults](config_defaults.json) and
individual model configuration files.
