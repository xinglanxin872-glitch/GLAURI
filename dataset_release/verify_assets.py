"""Verify local release assets; large dataset digests come from the restoration manifest."""
import argparse
import json
from pathlib import Path
from src.checkpoint import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--sha256sums', type=Path, help='Restored dataset ORIGINAL_SHA256SUMS.txt')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo/'assets/manifest.json').read_text())
    expected = {}
    if args.sha256sums:
        for line in args.sha256sums.read_text().splitlines():
            if line.strip():
                digest, name = line.split(maxsplit=1)
                expected[Path(name.lstrip('*')).name] = digest
    rows = []
    for item in manifest['files']:
        path = (repo if item['root'] == 'repo' else args.data_root)/item['path']
        digest = item.get('sha256') or expected.get(path.name)
        status = 'missing' if not path.is_file() else 'present_unverified'
        if path.is_file() and digest:
            status = 'verified' if sha256(path) == digest else 'checksum_mismatch'
        rows.append(dict(path=str(path), status=status))
    result = dict(status='PASS' if all(r['status']=='verified' for r in rows) else 'INCOMPLETE', files=rows)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
