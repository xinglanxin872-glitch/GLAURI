"""Sequence-feature ablation (zeroseq) of the cat36_rsa baseline.

The only difference from the config_baseline baseline:
  at forward time, zero out the entire ESM3 embedding segment of the node scalars (the 1536 dims after the first 36 structure-scalar columns);
  the structure scalars (29 residue one-hot + 6 dihedrals), the RSA column, and the edge features are all kept unchanged. In effect throughout training/validation.

Model architecture, dimensions, parameter count, and seed are identical to the baseline and align parameter by parameter.
"""

from config_baseline import (  # noqa: F401
    config_data,
    config_model,
    config_runtime,
    tag,
)

# All three input-ablation switches are written out explicitly to avoid crosstalk from the shared dict when the baseline module is reused in the same process.
_mcfg = config_model["params"]["gcp"]["module_cfg"]
_mcfg["zero_seq_input"] = True
_mcfg["zero_struct_input"] = False
_mcfg["zero_rsa_input"] = False
