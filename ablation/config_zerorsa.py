"""RSA-feature ablation (zerorsa) of the cat36_rsa baseline.

The only difference from the config_baseline baseline:
  at forward time, zero out column 36 of the node scalars (per-residue RSA, read by data_handler
  from ../bio_251210_rsa.h5 and catted at the end of the structure scalars); the one-hot + dihedrals,
  edge features, and the ESM3 segment are all unchanged. In effect throughout training/validation.

Zeroing instead of falling back to the 35-dim cat35 config keeps parameter count, initialization, and batch order bit-aligned with the baseline;
the shrunk-architecture control is 20260723_..._cat35_gcpdrop02 (h_input_dim=35+1536).

model.py checks rsa_scalar_dim>0 and errors out directly if the baseline config does not set the RSA column width,
avoiding a fake ablation where "nothing was actually zeroed".
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
_mcfg["zero_struct_input"] = False
_mcfg["zero_rsa_input"] = True
