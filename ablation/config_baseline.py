"""Baseline config entry point (GCP + ESM3 concat + per-residue RSA).

Data/model pipeline:
  native structure scalars, 36 dims (29 residue types + 6 dihedrals + 1 RSA) + ESM3 1536 dims
  -> [structure 36, Dropout(0.3, ELU(Linear(1536 -> 128)))]
  -> 164-dim node scalar input
  -> 6 GCP layers (interaction residual-branch dropout=0.2)

All parameter values live in config.py (already aligned with this baseline); this module only
re-exports them so that training (`--config config_baseline.py`) and the ablation configs
(`from config_baseline import ...`) have a stable baseline entry point.

RSA is precomputed by processing/add_sasa_to_h5.py (Shrake-Rupley, isolated chain, heavy atoms)
into the ../bio_251210_rsa.h5 sidecar; data_handler reads it via rsa_filepath and cats it as the
last node-scalar column.
"""

from config import (  # noqa: F401
    config_data,
    config_model,
    config_runtime,
    tag,
)
