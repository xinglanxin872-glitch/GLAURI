# Evaluation

Registered groups: standard_test, weighting, ablation. The full COACH420 is
the ligand test benchmark. Weighting uses protein_valid, nucleic_valid,
ion_valid and independent ligand_valid. All curated cohorts use bound structures
and common binary GT; their labels are independent of the evaluated model.
Single-task models evaluate matching heads only. AlloBench retains25 validation
and58 test targets; ablation includes AlloBench-test.


## Benchmark membership and data splits

The complete COACH420 is a test benchmark. Its sequence/native-label connected
components are excluded from H5 Train and Valid. Canonical unique structure
counts are Train=221549, Valid=36651, Test=96915. One additional pre-existing
held-out chain, 2r6a_c, remains held out with its three biological assemblies.

Of the 420 COACH chains, 402 are present in the effective H5 library and occur in
Test; the remaining 18 are external-only benchmark chains. The external labels
and bound/predicted structures cover all 420 chains. Boolean split masks align
with H5 metadata rows; the structure counts above count unique H5 keys. The split
manifest separately records contact-row, chain and cluster counts. Whole coarse
sequence/native co-label components stay in one fold.

The four curated validation cohorts contain 100 representatives each (390 unique
structures across tasks). Selection uses native-positive coarse 30% identity /
80% coverage clusters, seed 2025 plus task column, and one usable longest-sequence
representative per cluster, without model scores. Ligand has 1,055 candidate
clusters. Members, candidates, labels and selection provenance are under
`curation/curated_validation/`. Common binary evaluation truth combines native
and projected contacts; ligand projected pockets require support >=0.1.

## Run evaluation

```bash
python -m benchmark.paper_pipeline --inputs /path/to/evaluation_inputs.json --runs-root /path/to/paired_runs --output /path/to/new_results --stage plan
python -m benchmark.paper_pipeline --inputs /path/to/evaluation_inputs.json --runs-root /path/to/paired_runs --output /path/to/new_results --stage all
```

Relative paths resolve against the manifest, not the process working directory.
Use --asset-root project=/path/to/Interface_prediction for declared roots.
Duplicate members, incomplete cohorts and changed checkpoint/input hashes fail.
Output directories must be new or contain identical verified protocol outputs.

Outputs: metrics.csv, experiment_metrics.csv, icc_per_chain.csv,
pocket_jaccard.csv, per-model predictions and result_manifest.json. Plans derive
counts from actual registered models and active tasks. --limit is diagnostic.

ESM uses math attention and float32 caches. Legacy caches require explicit
--allow-legacy-embedding-cache and are recorded. ICC(A,1) calls src.metrics.icc_a1
with SSE/((n-1)*(k-1)); classification uses p>=0.5. External methods are not run:
their complete saved predictions are re-scored with actual method coverage.

MaSIF/DNA/RNA/Zn/AlloBench use full graph inference then scoring masks. COACH
retains its existing common-residue subgraph/shared-reference-ESM protocol;
expanding membership does not silently change that protocol. Cohort/protocol
identity must match before reusing any per-chain prediction.

model_benchmark_eval.py is a standalone compatibility CLI using shared checkpoint
and embedding APIs. bench_eval.py reads external predictions. Both require the
complete COACH420 membership. Plotting/SI/cases are in the separate paper package.
