# GLAURI inference checkpoint

True-Cα weights with four outputs: protein, nucleic acid, ion and ligand.
Checkpoint SHA256:
`256a07f12616aeaadf08bf9bf145dc7d0e3be76ab9301d3a74c88e2ce3a0fbfd`.

Full geometry, sidechain vectors, RSA, spatial kNN64 and ESM3 are enabled.
The probability classification threshold is 0.5. For annotation preparation,
see [training targets](../../colabel/README.md#training-targets).

`effective_config.json` contains the actual inference configuration paired with
these unchanged weights. `release_manifest.json` checks their identity and usage.
These existing inference weights were not trained and selected using this source
release's validation protocol. Complete original training records are archived
separately, rather than rewritten as a new training run.

For training with the current dataset release, use `python -m model.train
--config GLAURI`. New outputs include their own best-AUC/last weights and complete
input/validation provenance. See [training decision nodes](../TRAINING_DECISIONS.md).
