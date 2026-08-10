# infer_one.py smoketest

`apo_4x8m_A.pdb` is a genuinely single-chain structure (copied from
`case_study/apoholo_4x8m_4x8o/apo_4x8m_A.pdb`), used to verify `infer_one.py` end to end.

Reproduce:
```
conda activate boltz
cd /home/lanxin/data/GLAURI
python3 infer_one.py _smoketest/apo_4x8m_A.pdb --out-prefix _smoketest/apo_4x8m_A
```

Expected output (verified bit-for-bit against the independently-computed
`case_study/apoholo_4x8m_4x8o/apo_4x8m_A_GLAURI.pdb` reference at the time this was
written — 214/214 residues match to 2-decimal B-factor precision):

| channel | n | mean | max | n>=0.5 |
|---|---|---|---|---|
| protein | 214 | 0.148 | 0.858 | 19 |
| nn      | 214 | 0.014 | 0.286 | 0  |
| ion     | 214 | 0.117 | 0.717 | 4  |
| ligand  | 214 | 0.167 | 0.975 | 25 |

Note: `model/GLAURI/`'s own training-time config snapshots (`*_base_config.py` and
`*_config_v6_esm3_...py`) have twice gone missing from disk on their own this session
(the recurring volatility bug, not deliberate deletion -- see memory
`glauri-release-restructure.md` / `interface-prediction-dir-volatile`). `load_model()`
falls back to `model/config_baseline.py` when no snapshot is present in the checkpoint
dir, and that fallback has been verified to reproduce the exact same numbers above --
so this smoketest still passes, but the checkpoint's own provenance record is gone.
If exact reproducibility of *how this specific checkpoint was configured* ever matters,
that record no longer exists on disk.
