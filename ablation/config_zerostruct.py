"""Structure-feature ablation (zerostruct: one-hot + dihedrals + edge features) of the cat36_rsa baseline.

The only difference from the config_baseline baseline:
  at forward time, zero out the following feature blocks, keeping the ESM3 embedding segment and the RSA column unchanged:
    - the first 35 node-scalar columns (29-dim residue one-hot encoding + 6-dim dihedral encoding);
    - the edge feature scalars batch.e (distances and additional scalars from edge construction);
    - the edge feature vectors batch.xi (edge unit direction vectors).
  The edge_index topology and the f_ij local frames are kept (required for GCP message passing),
  and the node vector channels (backbone orientation, side-chain orientation) are kept as well, so this is not "full de-structuring"
  but "removing the explicit structure scalars and edge features". In effect throughout training/validation.

RSA is ablated separately by the zerorsa variant; this variant keeps RSA, so the two ablations are mutually orthogonal.
Model architecture, dimensions, parameter count, and seed are identical to the baseline.
"""

from config_baseline import (  # noqa: F401
    config_data,
    config_model,
    config_runtime,
    tag,
)

# All three input-ablation switches are written out explicitly to avoid crosstalk from the shared dict when the baseline module is reused in the same process.
_mcfg = config_model["params"]["gcp"]["module_cfg"]
_mcfg["zero_seq_input"] = False
_mcfg["zero_struct_input"] = True
_mcfg["zero_rsa_input"] = False
