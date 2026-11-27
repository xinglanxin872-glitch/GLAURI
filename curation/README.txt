Canonical dataset: coach420_curated_v1

Unique H5 structures: Train 221549; Valid 36651; Test 96915.
Counts are structures, not H5 contact-metadata rows; split_manifest.json also
records contact rows, unique chains and exact split/weight hashes.

COACH420 is entirely held out as test. Whole coarse sequence and native co-label
connected components stay in one fold. additional_test_chains.txt preserves an
existing held-out chain (2r6a_c) rather than returning its assemblies to training.
component_membership.json.gz supports independent cross-fold overlap checks.

The four curated validation cohorts each contain100 representatives from final
H5 Valid; ligand has its own independent membership in ligand_valid_chains.txt.
curated_validation/ contains candidate clusters, representatives, exclusions,
composition, common evaluation labels and a source/selection manifest. Existing
protein/nucleic/ion representatives were retained after eligibility checks.
Selection uses native-positive metadata, 30%-identity/80%-coverage clusters,
seed2025+task column, longest sequence first, without model scores.
AlloBench25/58-target memberships stay unchanged.

Boolean *_m.pkl masks align with H5 metadata/keys. *_h5_keys.txt preserves
first-occurrence Dataset order. Train membership/order and inverse-native-cluster
sampler weights are unchanged. CA masks validate sequence/residue identities;
validation masks are restricted to the current fold and record its split SHA.

Verify: python curation/generate_final_splits.py --check-only
Rebuild from source inputs:
  python curation/generate_final_splits.py --project-root /path/to/Interface_prediction --output-dir /path/to/new_splits
Source hashes are recorded in split_manifest.json. The metadata supplement also
supports --source-inputs /path/to/source_inputs --h5 /path/to/bio_251210.h5
instead of --project-root. Rebuild curate with:
  python curation/build_curated_validation.py --coarse-clusters /path/to/coarse_clusters.txt --existing-representatives curation/curated_validation/representatives.csv --output /path/to/new_curated

Training labels use the fixed GLAURI definition. Curated evaluation labels are
common binary GT and have their separate definition in the curated manifest.
Do not treat binary contact GT or set union operations as alternative models.
