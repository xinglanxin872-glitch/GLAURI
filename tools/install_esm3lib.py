"""Prepare a versioned esm3lib namespace from a local upstream SDK wheel/source.

This does not fetch models, install dependencies, or overwrite an existing target.
Install the upstream SDK dependencies separately while retaining fair-esm's esm.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Local esm/ or esm3lib/ SDK directory')
    parser.add_argument('--wheel', type=Path, help='Downloaded upstream esm SDK wheel')
    parser.add_argument('--target', required=True, type=Path, help='Parent directory that will contain esm3lib/')
    parser.add_argument('--version', default='3.2.3', help='Expected SDK __version__')
    args = parser.parse_args()
    if bool(args.source) == bool(args.wheel):
        parser.error('Choose exactly one of --source or --wheel')
    output = args.target.resolve()/'esm3lib'
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory() as td:
        source = args.source
        if args.wheel:
            with zipfile.ZipFile(args.wheel) as archive:
                for member in archive.infolist():
                    parts = Path(member.filename).parts
                    if member.filename.startswith('/') or '..' in parts:
                        raise ValueError('Unsafe archive member')
                archive.extractall(td)
            source = Path(td)/'esm'
        source = source.resolve()
        version = re.search(r'__version__\s*=\s*[\'"]([^\'"]+)', (source/'__init__.py').read_text())
        if not version or version.group(1) != args.version:
            raise ValueError('SDK version differs from the requested reference')
        licenses = list(source.glob('LICENSE*')) + list(source.parent.glob('LICENSE*'))
        licenses += list(Path(td).glob('*.dist-info/licenses/LICENSE*'))
        if not licenses:
            raise ValueError('Keep the upstream SDK license next to the source before preparing it')
        args.target.mkdir(parents=True, exist_ok=True)
        staging = Path(td)/'prepared'
        shutil.copytree(source, staging, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for path in staging.rglob('*.py'):
            text = path.read_text()
            text = re.sub(r'\besm(?=\.)', 'esm3lib', text)
            text = re.sub(r'\bfrom esm import\b', 'from esm3lib import', text)
            text = re.sub(r'\bimport esm\b(?!\.)', 'import esm3lib', text)
            ast.parse(text, filename=str(path))
            path.write_text(text)
        shutil.copy2(licenses[0], staging/'LICENSE.upstream')
        inventory = {str(p.relative_to(staging)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in staging.rglob('*') if p.is_file()}
        (staging/'GLAURI_NAMESPACE.json').write_text(json.dumps(dict(version=args.version,
            changes='Python esm namespace renamed to esm3lib; weights excluded', files=inventory), indent=2)+'\n')
        shutil.copytree(staging, output)
    print(f'Prepared {output}. Add its parent to PYTHONPATH; model weights are external assets.')


if __name__ == '__main__':
    main()
