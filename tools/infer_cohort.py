"""Resume structure-cohort inference through the same API as infer_one.

Cohort JSON: [{"id":"protein1", "path":"relative/input.cif", "chain":"A"}, ...].
All paths are relative to the cohort file unless absolute. Outputs contain all
four task probabilities and full-precision Cα coordinates; no plotting occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.checkpoint import load_checkpoint, sha256
from src.embeddings import EmbeddingProvider
from src.predict import predict_structure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--device')
    args = parser.parse_args()
    rows = json.loads(args.cohort.read_text())
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate cohort IDs')
    for row in rows:
        if Path(row['id']).name != row['id'] or row['id'] in ('.', '..'):
            raise ValueError('Cohort IDs must be plain filenames')
    loaded = load_checkpoint(args.model_dir, device=args.device)
    provider = EmbeddingProvider(loaded.metadata['device'], backend='math', memory_size=0)
    if provider.identity.get('weights_sha256') is None:
        raise ValueError('Cache the identified ESM3 model weights before cohort inference')
    protocol = dict(checkpoint_sha256=loaded.metadata['checkpoint_sha256'],
        feature_protocol=loaded.metadata['feature_protocol'], input_policy=loaded.metadata['input_policy'],
        embedding_protocol=provider.protocol, cohort_sha256=sha256(args.cohort),
        task_columns=['protein', 'nucleic_acid', 'ion', 'ligand'])
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError('Output directory belongs to a different cohort/model/protocol')
    if not manifest.exists() and any(args.output.iterdir()):
        raise ValueError('Nonempty output directory has no verified protocol manifest')
    manifest.write_text(json.dumps(protocol, indent=2)+'\n')
    complete = 0
    for row in rows:
        path = Path(row['path'])
        path = path if path.is_absolute() else args.cohort.resolve().parent/path
        digest = sha256(path)
        output = args.output/(row['id']+'.npz')
        if output.exists():
            with np.load(output, allow_pickle=False) as old:
                p = old['probs']
                if (str(old['source_sha256']) != digest or
                    str(old['checkpoint_sha256']) != loaded.metadata['checkpoint_sha256'] or
                    p.shape != (len(str(old['seq'])), 4) or not np.isfinite(p).all() or
                    not ((p >= 0) & (p <= 1)).all()):
                    raise ValueError(f'Input/checkpoint/prediction identity differs: {path}')
        else:
            probabilities, sequence, positions = predict_structure(loaded, path, row['chain'], provider)
            fd, temp = tempfile.mkstemp(dir=args.output, suffix='.npz')
            os.close(fd)
            try:
                np.savez_compressed(temp, seq=np.asarray(sequence), probs=probabilities[:, 1:].numpy(),
                    node_positions=positions, source_sha256=np.asarray(digest),
                    checkpoint_sha256=np.asarray(loaded.metadata['checkpoint_sha256']))
                os.replace(temp, output)
            finally:
                Path(temp).unlink(missing_ok=True)
        complete += 1
        if complete == 1 or complete % 25 == 0 or complete == len(rows):
            print(f'INFERENCE {complete}/{len(rows)}', flush=True)
    (args.output/'status.json').write_text(json.dumps(dict(status='complete', proteins=complete), indent=2)+'\n')


if __name__ == '__main__':
    main()
