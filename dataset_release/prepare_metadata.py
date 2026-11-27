"""Package versioned splits, curated truth and training inputs separately from raw H5."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

REPO=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--project-root',type=Path,required=True,help='Original construction inputs')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    files={}
    for path in (REPO/'curation').rglob('*'):
        if path.is_file() and path.suffix in ('.pkl','.txt','.csv','.json','.gz'):
            files[str(path.relative_to(REPO))]=path
    data=args.data_root
    for relative in ['all_seq_dict.pkl','ligand_pocket_interactions.csv.gz','label_stores/reference_labels.pkl',
                     'alignment_masks_ca_v1/mask_dict_train.pkl','alignment_masks_ca_v1/mask_dict_train.json',
                     'alignment_masks_ca_v1/mask_dict_valid.pkl','alignment_masks_ca_v1/mask_dict_valid.json']:
        files['data/'+relative]=data/relative
    project=args.project_root
    files['source_inputs/coarse_clusters.txt']=project/'cluster/pdb_seqres_pro_clean_cluster_cluster_30_80_forSplit.txt'
    files['source_inputs/native_mapping.pkl']=project/'model/20260917_mmseq_longest_soft_ml/support/colabel_mask_all.pkl'
    files['source_inputs/allobench_test_pockets.csv']=project/'colabel/allobench_pockets_testset.csv'
    for role in ['train','valid','test']:
        files[f'source_inputs/base_masks/{role}_m.pkl']=project/f'processing/data/datasets/cluster_30_80_benchclean_{"val" if role=="valid" else role}_m.pkl'
    manifest={'schema':'glauri_metadata_supplement_v1','dataset_version':'coach420_curated_v1',
              'raw_h5_modified':False,'files':[]}
    archive=args.output/'GLAURI_metadata_coach420_v1.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as package:
        for relative,path in sorted(files.items()):
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            manifest['files'].append(dict(path=relative,sha256=digest,size_bytes=path.stat().st_size))
            package.write(path,relative)
        package.writestr('metadata_manifest.json',json.dumps(manifest,indent=2)+'\n')
    # Verify decompressed bytes without retaining a second extracted copy.
    with zipfile.ZipFile(archive) as package:
        for row in manifest['files']:
            digest=hashlib.sha256();count=0
            with package.open(row['path']) as stream:
                while block:=stream.read(8*1024*1024):digest.update(block);count+=len(block)
            if digest.hexdigest()!=row['sha256'] or count!=row['size_bytes']:raise ValueError(row['path'])
    manifest['archive_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest['archive_size_bytes']=archive.stat().st_size;manifest['verified']=True
    (args.output/'metadata_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'README.md').write_text('''# GLAURI dataset metadata supplement

Data version: coach420_curated_v1. Raw H5/RSA files are unchanged and must be
restored using their independent compressed-data manifest. Extract the metadata
ZIP into a fresh directory; verify entries against metadata_manifest.json.
The curation directory supplies canonical folds, members and fixed curated GT.
The data directory supplies reference sequences, non-ligand labels, pocket
interactions and verified CA alignment masks. Copy these into the corresponding
locations of the matching code release, or provide explicit paths.json.

Build the single GLAURI ligand label store with python -m model.prepare --config all.
Reconstruct splits from source_inputs with:
python curation/generate_final_splits.py --source-inputs /path/to/source_inputs --h5 /path/to/bio_251210.h5 --output-dir /path/to/new_splits

The ZIP is an addition to the H5 restoration package, not a replacement. Its
archive hash and individual decompressed-file hashes are independently verified.
''')
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))


if __name__=='__main__':main()
