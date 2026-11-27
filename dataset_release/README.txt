Lossless H5 dataset packaging

prepare_zenodo.py compresses each original H5 with Zstandard level 19 and a
1 GiB long-distance window, splits it into at most 5 GB parts, and verifies a
full decompression against the original SHA256. Original H5 files are read-only.

Example, from the repository root:
  python dataset_release/prepare_zenodo.py \
    --source data/bio_251210_rsa.h5 data/bio_251210.h5 \
    --output /path/to/new_upload_directory --threads 12

Requires Python 3.8+ and the zstd command-line utility. The output directory must
be empty. --resume rechecks and reuses completed files; an interrupted compressed
stream restarts from the beginning. Compression parameters must stay the same.

Upload all output parts together with manifest.json, README.txt,
ORIGINAL_SHA256SUMS.txt, SHA256SUMS.txt and restore_h5.py. The package is ready only
when manifest.json has "complete": true. Every full H5 has then passed round-trip
size and SHA256 checks. Parts are segments of a compressed stream, not independent
HDF5 files. Keep them all together.

Download all package files and restore with:
  python restore_h5.py --output /path/to/restored_h5
The restore script checks each compressed part and the complete original H5.
It refuses existing outputs. Allow about 2 GB RAM and enough disk space for the
original files. --only bio_251210_rsa.h5 restores just the sidecar.

The same package works with Zenodo or other file hosts. Zenodo's record quota is
for the sum of all files; splitting does not reduce the total. Check the current
quota in the upload draft: https://help.zenodo.org/docs/deposit/manage-quota/

Hugging Face upload

Use a current huggingface_hub installation (the uploader verifies individual file uploads).
Authenticate locally with a token that can write the destination dataset:
  hf auth login
  python dataset_release/upload_huggingface.py \
    --folder /path/to/prepared_dataset \
    --repo-id sindax/GLAURI_processed

The destination repository must already exist. Only the 16 manifest-listed
compressed parts and five restoration/verification files are uploaded. No remote
files are deleted and repository visibility is preserved. Re-run the same command
after an interruption to resume. Files are committed individually; already
verified remote files are skipped. Transient connection and service errors are
retried up to 12 times per operation with increasing delays (30 to 300 seconds).
--max-attempts and --retry-delay override these settings.
Remote file sizes and compressed-part SHA256
metadata are checked after upload. Progress state is recorded locally in
~/.cache/glauri_hf_upload/status.json; "complete" means remote verification passed.
completed_bytes and completed_files count verified remote files, not network
traffic; current_file identifies the active file during transfer.
For a server handoff, the folder may contain only the remaining parts and the
five metadata files. Parts already present remotely with matching size/SHA256
need not exist locally. The manifest must still describe the entire release.
An "authentication_required" state means no upload was started.

Versioned metadata supplement

Raw H5/RSA bytes and their restoration checksums do not change when external
split masks change. The code release coach420_curated_v1 uses complete COACH420
for testing and independently curated H5 validation representatives. Obtain the
matching metadata ZIP in addition to the original H5 restoration package.
Build and verify the supplement locally:
  python dataset_release/prepare_metadata.py --data-root /path/to/GLAURI/data --project-root /path/to/Interface_prediction --output /path/to/new_metadata_package

The supplement includes masks, members, candidate/representative/label records,
reference sequences, contact inputs, CA alignment masks and original construction
inputs. metadata_manifest.json checks every decompressed entry and the ZIP.
The H5 uploader's existing part list does not upload this supplement. Publish
its ZIP, metadata manifest and README as explicit additional versioned assets;
update the dataset card and code/data availability links to the same version.
Do not reuse H5 restoration SHA256 entries for this separate metadata archive.
