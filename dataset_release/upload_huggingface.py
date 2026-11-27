#!/usr/bin/env python3
"""Upload the verified compressed H5 release and check remote sizes/SHA256.

Authenticate first with `hf auth login` in the same Python environment.
Re-run this script to resume an interrupted upload.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--repo-id', default='sindax/GLAURI_processed')
    parser.add_argument('--status', type=Path,
                        default=Path.home() / '.cache/glauri_hf_upload/status.json')
    parser.add_argument('--max-attempts', type=int, default=12)
    parser.add_argument('--retry-delay', type=float, default=30)
    args = parser.parse_args()
    if args.max_attempts < 1 or args.retry_delay < 0:
        parser.error('max-attempts must be positive and retry-delay nonnegative')
    args.status.parent.mkdir(parents=True, exist_ok=True)

    def status(state, **details):
        record = dict(state=state, repo_id=args.repo_id, folder=str(args.folder),
                      updated_at=datetime.now(timezone.utc).isoformat(), **details)
        tmp = args.status.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, indent=2) + '\n')
        os.replace(tmp, args.status)

    try:
        manifest = json.loads((args.folder / 'manifest.json').read_text())
        if not manifest.get('complete') or not all(f.get('verified') for f in manifest['files']):
            raise ValueError('Release manifest must be complete and verified')
        expected = {}
        for entry in manifest['files']:
            for part in entry['parts']:
                expected[part['name']] = (part['bytes'], part['sha256'])
        for name in ('README.txt', 'restore_h5.py', 'manifest.json',
                     'SHA256SUMS.txt', 'ORIGINAL_SHA256SUMS.txt'):
            path = args.folder / name
            expected[name] = (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        total = sum(size for size, _ in expected.values())
        progress = dict(files=len(expected), bytes=total, completed_files=0, completed_bytes=0)

        def retry(operation, action):
            for attempt in range(1, args.max_attempts + 1):
                try:
                    return action()
                except Exception as exc:
                    code = getattr(getattr(exc, 'response', None), 'status_code', None)
                    transient = (code in (408, 429, 500, 502, 503, 504)
                                 or isinstance(exc, (ConnectionError, TimeoutError))
                                 or type(exc).__name__ in ('ConnectionError', 'Timeout', 'ReadTimeout',
                                     'ConnectTimeout', 'ConnectError', 'ReadError', 'WriteError',
                                     'RemoteProtocolError'))
                    if not transient or attempt == args.max_attempts:
                        raise
                    delay = min(args.retry_delay * attempt, 300)
                    status('retrying', operation=operation, attempt=attempt,
                           retry_in_seconds=delay, error_type=type(exc).__name__, **progress)
                    print(f'{operation}: {type(exc).__name__}; retry {attempt + 1} '
                          f'in {delay:g} seconds', flush=True)
                    time.sleep(delay)

        api = HfApi()
        try:
            account = retry('authentication', api.whoami)['name']
        except Exception as exc:
            code = getattr(getattr(exc, 'response', None), 'status_code', None)
            if code not in (401, 403):
                raise
            status('authentication_required')
            print('Hugging Face authentication failed. Run hf auth login with a write token, '
                  'then re-run this script. Do not paste the token into chat.', file=sys.stderr)
            return 2
        # The destination already exists; leave its visibility and unrelated files unchanged.
        def remote_info():
            return retry('remote metadata', lambda: api.repo_info(
                args.repo_id, repo_type='dataset', files_metadata=True, timeout=30))

        info = remote_info()
        remote = {file.rfilename: file for file in info.siblings}
        print(f'Account: {account}; destination: {args.repo_id}', flush=True)
        print(f'Uploading {len(expected)} files ({total:,} bytes)', flush=True)

        def matches(file, name):
            size, sha256 = expected[name]
            if file is None or file.size != size:
                return False
            if file.lfs is not None:
                return file.lfs.sha256 == sha256
            return '.zst.part' not in name

        # A handoff server only needs uncommitted parts plus the five metadata
        # files; remote parts with verified hashes need not be copied locally.
        for name, (size, _) in expected.items():
            if matches(remote.get(name), name):
                continue
            path = args.folder / name
            if not path.is_file() or path.stat().st_size != size:
                raise ValueError(f'Missing or incorrect local file not verified remotely: {name}')

        # Commit each file separately so a connection failure cannot hold back
        # the whole release. Smaller documentation files are committed first.
        for name in sorted(expected, key=lambda n: (expected[n][0], n)):
            if not matches(remote.get(name), name):
                status('uploading', current_file=name, **progress)
                retry(f'upload {name}', lambda: api.upload_folder(
                    repo_id=args.repo_id, repo_type='dataset', folder_path=str(args.folder),
                    allow_patterns=[name], commit_message=f'Upload {name}'))
                info = remote_info()
                remote = {file.rfilename: file for file in info.siblings}
                if not matches(remote.get(name), name):
                    raise RuntimeError(f'Remote verification failed: {name}')
            progress['completed_files'] += 1
            progress['completed_bytes'] += expected[name][0]
            status('uploading', last_completed_file=name, **progress)
            print(f"Verified {progress['completed_files']}/{len(expected)}: {name}; "
                  f"{progress['completed_bytes'] / total:.1%} of total bytes", flush=True)
        status('verifying', **progress)
        info = remote_info()
        remote = {file.rfilename: file for file in info.siblings}
        for name, (size, sha256) in expected.items():
            file = remote.get(name)
            if file is None or file.size != size:
                raise RuntimeError(f'Remote file missing or incorrect size: {name}')
            if file.lfs is not None:
                if file.lfs.sha256 != sha256:
                    raise RuntimeError(f'Remote SHA256 mismatch: {name}')
            elif '.zst.part' in name:
                raise RuntimeError(f'Remote SHA256 metadata unavailable: {name}')
        url = f'https://huggingface.co/datasets/{args.repo_id}/tree/main'
        status('complete', files=len(expected), bytes=total, commit=info.sha,
               compressed_parts_sha256_verified=True, url=url)
        print(f'Upload and remote verification complete: {url}', flush=True)
        return 0
    except Exception as exc:
        # Avoid recording HTTP exception bodies or signed upload URLs in the status file.
        status('failed', error_type=type(exc).__name__)
        print(f'Upload failed ({type(exc).__name__}). Re-run to resume.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
