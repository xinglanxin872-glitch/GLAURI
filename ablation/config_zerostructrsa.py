"""Full structure ablation (zerostructrsa = zerostruct + zerorsa) of the cat36_rsa baseline.

RSA itself is also a structure-derived quantity (SASA computed from isolated-chain coordinates, then normalized), so besides
ablating one-hot + dihedrals + edge features alone (zerostruct), this additional variant removes RSA as well,
answering "how much is left after removing all explicit structural information from the node scalars".

The only difference from the config_baseline baseline:
  at forward time, zero out the following feature blocks, leaving only the ESM3 embedding segment:
    - the first 35 node-scalar columns (29-dim residue one-hot + 6-dim dihedral encoding);
    - node-scalar column 36 (per-residue RSA);
    - the edge feature scalars batch.e and the edge feature vectors batch.xi.
  The edge_index topology, the f_ij local frames, and the node vector channels (backbone/side-chain orientation) are kept
  (required for GCP message passing), so this is still not "full de-structuring" but
  "removing all explicit structure descriptors from the node scalars and edge features". In effect throughout training/validation.

Complementary pair with zeroseq: zeroseq keeps only structure, this variant keeps only ESM3.
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
_mcfg["zero_rsa_input"] = True
