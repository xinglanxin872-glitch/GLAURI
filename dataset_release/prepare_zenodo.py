#!/usr/bin/env python3
"""Losslessly compress H5 files, split the byte stream, and verify original SHA256."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

BLOCK = 4 * 1024 * 1024


def save_manifest(path, manifest):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(manifest, indent=2) + '\n')
    temp.replace(path)


def verify_parts(parts, directory, expected, zstd):
    process = subprocess.Popen([zstd, '-d', '-q', '--long=30', '-c'],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    errors = []
    def feed():
        try:
            for part in parts:
                digest = hashlib.sha256()
                with (directory / part['name']).open('rb') as f:
                    while chunk := f.read(BLOCK):
                        digest.update(chunk)
                        process.stdin.write(chunk)
                if digest.hexdigest() != part['sha256']:
                    raise ValueError('Part checksum mismatch: ' + part['name'])
        except BaseException as error:
            errors.append(error)
        finally:
            process.stdin.close()
    worker = threading.Thread(target=feed, daemon=True); worker.start()
    digest = hashlib.sha256(); count = 0
    try:
        while chunk := process.stdout.read(BLOCK):
            digest.update(chunk); count += len(chunk)
        code = process.wait(); worker.join()
        if errors: raise errors[0]
        if code or count != expected['original_bytes'] or digest.hexdigest() != expected['original_sha256']:
            raise ValueError('Round-trip verification failed for ' + expected['name'])
    finally:
        if process.poll() is None: process.kill(); process.wait()
        process.stdout.close()
    return count


def compress(source, directory, *, zstd, level, threads, part_bytes):
    before = source.stat()
    command = [zstd, f'-{level}', f'-T{threads}', '--long=30', '--check',
               f'--stream-size={before.st_size}', '-q', '-c']
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    original = hashlib.sha256(); counters = {'read': 0}; errors = []
    def feed():
        try:
            with source.open('rb') as f:
                while chunk := f.read(BLOCK):
                    original.update(chunk); process.stdin.write(chunk)
                    counters['read'] += len(chunk)
        except BaseException as error:
            errors.append(error)
        finally:
            process.stdin.close()
    worker = threading.Thread(target=feed, daemon=True); worker.start()
    parts = []; handle = None; part_hash = None; part_size = 0
    total = 0; started = last = time.monotonic()
    try:
        while chunk := process.stdout.read(BLOCK):
            if shutil.disk_usage(directory).free < 10 * 1024**3:
                raise OSError('Stopped with less than 10 GiB free; original inputs are untouched')
            offset = 0
            while offset < len(chunk):
                if handle is None:
                    name = f'{source.name}.zst.part{len(parts):03d}'
                    handle = (directory / name).open('xb')
                    part_hash = hashlib.sha256(); part_size = 0
                take = min(len(chunk) - offset, part_bytes - part_size)
                data = chunk[offset:offset + take]
                handle.write(data); part_hash.update(data)
                part_size += take; total += take; offset += take
                if part_size == part_bytes:
                    handle.close(); handle = None
                    parts.append(dict(name=name, bytes=part_size, sha256=part_hash.hexdigest()))
            now = time.monotonic()
            if now - last >= 20:
                print(f'{source.name}: input {counters["read"]/1e9:.2f}/{before.st_size/1e9:.2f} GB; '
                      f'compressed {total/1e9:.2f} GB; elapsed {(now-started)/60:.1f} min', flush=True)
                last = now
        code = process.wait(); worker.join()
        if errors: raise errors[0]
        if code: raise RuntimeError(f'zstd exited with status {code}')
        if handle is not None:
            handle.close(); handle = None
            parts.append(dict(name=name, bytes=part_size, sha256=part_hash.hexdigest()))
        after = source.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise RuntimeError('Source changed during compression: ' + str(source))
        record = dict(name=source.name, original_bytes=before.st_size,
                      original_sha256=original.hexdigest(), compressed_bytes=total,
                      compression_ratio=total/before.st_size, parts=parts,
                      compression_seconds=round(time.monotonic()-started, 2), verified=False)
        print(f'{source.name}: compressed to {total/1e9:.3f} GB; verifying full decompression...', flush=True)
        verify_parts(parts, directory, record, zstd)
        record['verified'] = True
        print(f'{source.name}: SHA256 round-trip VERIFIED; {len(parts)} part(s)', flush=True)
        return record
    finally:
        if process.poll() is None: process.kill(); process.wait()
        if handle is not None: handle.close()
        process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--level', type=int, default=19, choices=range(1, 20))
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--part-bytes', type=int, default=5_000_000_000)
    parser.add_argument('--zstd', default=shutil.which('zstd'))
    parser.add_argument('--resume', action='store_true', help='Verify completed files and restart incomplete files')
    args = parser.parse_args()
    if not args.zstd: parser.error('Install the zstd command-line program first')
    if args.part_bytes <= 0: parser.error('--part-bytes must be positive')
    if len({p.name for p in args.source}) != len(args.source): parser.error('Source basenames must be unique')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        if not args.resume: parser.error('Use an empty output directory, or --resume')
        manifest = json.loads((args.output/'manifest.json').read_text())
        if manifest.get('complete'): parser.error('The package is already complete')
        if (manifest.get('level'), manifest.get('long_window_log'), manifest.get('part_bytes')) != (args.level, 30, args.part_bytes):
            parser.error('Resume compression parameters must match the original manifest')
        completed = {record['name']: record for record in manifest['files']}
        if not set(completed) <= {p.name for p in args.source}:
            parser.error('Resume sources differ from the original manifest')
    else:
        manifest = dict(format_version=1, complete=False, compression='zstd',
                        level=args.level, long_window_log=30, part_bytes=args.part_bytes,
                        zstd_version=subprocess.check_output([args.zstd, '--version'], text=True).strip(), files=[])
        completed = {}
    save_manifest(args.output/'manifest.json', manifest)
    for source in args.source:
        if source.name in completed:
            record = completed[source.name]
            print(f'Verifying completed file before reuse: {source.name}', flush=True)
            digest = hashlib.sha256()
            with source.open('rb') as handle:
                while chunk := handle.read(BLOCK): digest.update(chunk)
            if source.stat().st_size != record['original_bytes'] or digest.hexdigest() != record['original_sha256']:
                raise ValueError('Completed source has changed: ' + str(source))
            verify_parts(record['parts'], args.output, record, args.zstd)
            print(f'Reusing verified completed file: {source.name}', flush=True)
            continue
        leftovers = sorted(args.output.glob(source.name + '.zst.part[0-9][0-9][0-9]'))
        if leftovers:
            print(f'Restarting incomplete {source.name} from the beginning; removing {len(leftovers)} incomplete parts', flush=True)
            for path in leftovers:
                if path.is_symlink() or not path.is_file():
                    raise ValueError('Unexpected partial-file type: ' + str(path))
                path.unlink()
        print(f'Starting {source} ({source.stat().st_size/1e9:.3f} GB)', flush=True)
        manifest['files'].append(compress(source, args.output, zstd=args.zstd,
            level=args.level, threads=args.threads, part_bytes=args.part_bytes))
        save_manifest(args.output/'manifest.json', manifest)
    shutil.copyfile(Path(__file__).with_name('restore_h5.py'), args.output/'restore_h5.py')
    original_checksums = ''.join(f'{r["original_sha256"]}  {r["name"]}\n' for r in manifest['files'])
    (args.output/'ORIGINAL_SHA256SUMS.txt').write_text(original_checksums)
    total = sum(r['compressed_bytes'] for r in manifest['files'])
    (args.output/'README.txt').write_text(
        'GLAURI structure/contact dataset and RSA sidecar\n\n'
        f'Lossless Zstandard compression, split into at most {args.part_bytes:,}-byte parts.\n'
        'The original HDF5 bytes, dataset paths, attributes, dtypes and values are preserved.\n'
        'Parts are segments of one compressed stream per H5, not independent HDF5 files.\n'
        'Download all parts and metadata into one folder, then run:\n\n'
        '  python3 restore_h5.py --output /path/to/restored_h5\n\n'
        'Requires Python 3.8+ and the zstd command-line program (no Python packages).\n'
        'On Windows, Python and zstd must be installed and available on PATH.\n'
        'Restoration needs approximately 2 GB RAM plus disk space for the original H5 files.\n'
        'Compressed-part and restored-H5 SHA256 checks are automatic. Existing output H5s\n'
        'are never overwritten. Compression and full decompression were verified before release.\n\n'
        + '\n'.join(f'{r["name"]}: {r["original_bytes"]:,} -> {r["compressed_bytes"]:,} bytes; '
                    f'{len(r["parts"])} part(s)' for r in manifest['files'])
        + f'\n\nTotal compressed bytes: {total:,}\n'
        'Upload every file in this folder to ONE Zenodo record. Splitting does not reduce\n'
        'the record total. Default quota: 50 GB; allocate additional storage in the draft\n'
        'if necessary: https://help.zenodo.org/docs/deposit/manage-quota/\n')
    manifest['complete'] = True; save_manifest(args.output/'manifest.json', manifest)
    sums = []
    known = {p['name']:p['sha256'] for r in manifest['files'] for p in r['parts']}
    for path in sorted(args.output.iterdir()):
        if path.name == 'SHA256SUMS.txt': continue
        digest = known.get(path.name) or hashlib.sha256(path.read_bytes()).hexdigest()
        sums.append(f'{digest}  {path.name}\n')
    (args.output/'SHA256SUMS.txt').write_text(''.join(sums))
    print(f'COMPLETE: {total/1e9:.3f} GB compressed, {sum(len(r["parts"]) for r in manifest["files"])} parts; {args.output}', flush=True)


if __name__ == '__main__':
    main()
