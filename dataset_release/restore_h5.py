#!/usr/bin/env python3
"""Restore and verify the original HDF5 files from downloaded Zstandard parts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import threading

BLOCK = 4 * 1024 * 1024


def restore(record, source, output, zstd):
    target = output / record['name']
    partial = output / (record['name'] + '.partial')
    if target.exists() or partial.exists():
        raise FileExistsError(f'Refusing to overwrite {target} or {partial}')
    if shutil.disk_usage(output).free < record['original_bytes'] + 1024**3:
        raise OSError('Insufficient output space for ' + record['name'])
    process = subprocess.Popen([zstd, '-d', '-q', '--long=30', '-c'],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    errors = []
    def feed():
        try:
            for part in record['parts']:
                path = source / part['name']
                if path.stat().st_size != part['bytes']:
                    raise ValueError('Part size mismatch: ' + path.name)
                digest = hashlib.sha256()
                with path.open('rb') as f:
                    while chunk := f.read(BLOCK):
                        digest.update(chunk); process.stdin.write(chunk)
                if digest.hexdigest() != part['sha256']:
                    raise ValueError('Part SHA256 mismatch: ' + path.name)
        except BaseException as error:
            errors.append(error)
        finally:
            process.stdin.close()
    worker = threading.Thread(target=feed, daemon=True); worker.start()
    digest = hashlib.sha256(); count = 0
    try:
        with partial.open('xb') as f:
            while chunk := process.stdout.read(BLOCK):
                f.write(chunk); digest.update(chunk); count += len(chunk)
        code = process.wait(); worker.join()
        if errors: raise errors[0]
        if code or count != record['original_bytes'] or digest.hexdigest() != record['original_sha256']:
            raise ValueError('Restored file SHA256/size mismatch: ' + record['name'])
        partial.replace(target)
        print(f'Verified: {target} ({count:,} bytes)', flush=True)
    finally:
        if process.poll() is None: process.kill(); process.wait()
        process.stdout.close()
        if partial.exists(): partial.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--zstd', default=shutil.which('zstd'))
    parser.add_argument('--only', nargs='+', help='Optional original H5 filenames to restore')
    args = parser.parse_args()
    if not args.zstd: parser.error('Install zstd and add it to PATH, or pass --zstd /path/to/zstd')
    manifest = json.loads((args.source/'manifest.json').read_text())
    if not manifest.get('complete'): parser.error('Package creation has not completed')
    available = {r['name'] for r in manifest['files']}
    if args.only and not set(args.only) <= available: parser.error('--only contains an unknown original filename')
    args.output.mkdir(parents=True, exist_ok=True)
    for record in manifest['files']:
        if args.only and record['name'] not in args.only: continue
        restore(record, args.source, args.output, args.zstd)


if __name__ == '__main__':
    main()
