#!/usr/bin/env python3
"""Read-only binding-mode lookup by chain name, using a portable SQLite index.

Build once from the original cluster membership CSV and four contact tables.
Querying/serving needs only Python's standard library and the resulting database.
The index summarizes existing pocket IDs; it does not recluster or alter labels.
"""
import argparse
import ast
from collections import Counter
from contextlib import closing
import csv
import gzip
import hashlib
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
from urllib.parse import parse_qs, urlparse

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
TYPES = ('protein', 'nn', 'ion', 'ligand')
SCHEMA = 'glauri_binding_modes_v2'


def chain_name(key):
    parts = key.split('/')
    if len(parts) != 3:
        raise ValueError('Expected a structure key bucket/PDB/chain:model: ' + key)
    return parts[1].upper() + '_' + parts[2].split(':')[0]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def contact_medoid(vectors):
    """Exact record-weighted Jaccard medoid; ties retain source record order.

    Identical nonempty vectors are compressed to (count, first contact). Their
    multiplicity is retained in the objective, so this equals evaluating all
    original contact records, including duplicates and the zero self-distance.
    """
    entries = list(vectors.items())
    if not entries or any(bits == 0 or value[0] < 1 for bits, value in entries):
        raise ValueError('Medoid needs nonempty contact vectors and positive counts')
    totals = [0.] * len(entries)
    for i, (bits, (count, _)) in enumerate(entries):
        for j in range(i):
            other, (other_count, _) = entries[j]
            distance = 1. - (bits & other).bit_count()/(bits | other).bit_count()
            totals[i] += other_count * distance
            totals[j] += count * distance
    records = sum(value[0] for _, value in entries)
    lowest = min(totals)
    # Only numerical ties (<=1e-12 in mean distance) use source order.
    winner = next(i for i, score in enumerate(totals) if score <= lowest + records*1e-12)
    return dict(entries[winner][1][1], mean_jaccard_distance=totals[winner]/records,
                method='record_weighted_jaccard_medoid', tie_rule='first source contact within 1e-12 mean distance')


def connect(path):
    connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    version = connection.execute('SELECT value FROM metadata WHERE key=?', ('schema',)).fetchone()
    if version is None or version[0] != SCHEMA:
        connection.close()
        raise ValueError('Unsupported binding-mode index')
    return connection


def build(cluster_csv, tables, output):
    """Keep all modes; apply residue/support filters at query time only."""
    if output.exists():
        raise FileExistsError(output)
    paths = [cluster_csv] + [tables / (name + '_pocket_interactions.csv.gz') for name in TYPES]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=output.name + '.', dir=output.parent, delete=False) as temp:
        temporary = Path(temp.name)
    db = sqlite3.connect(temporary)
    try:
        db.executescript('''
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE structure_sets(id TEXT PRIMARY KEY, member_count INTEGER NOT NULL);
            CREATE TABLE members(key TEXT PRIMARY KEY, chain TEXT NOT NULL, set_id TEXT NOT NULL);
            CREATE INDEX members_chain ON members(chain);
            CREATE INDEX members_chain_folded ON members(lower(chain));
            CREATE INDEX members_key_folded ON members(lower(key));
            CREATE INDEX members_set ON members(set_id);
            CREATE TABLE interfaces(set_id TEXT, type TEXT, ref_id TEXT, ref_seq TEXT,
                receptor_count INTEGER, record_count INTEGER, PRIMARY KEY(set_id,type));
            CREATE TABLE modes(set_id TEXT, type TEXT, pocket_id INTEGER,
                record_count INTEGER, receptor_count INTEGER, residue_count INTEGER,
                positions TEXT, consensus_positions TEXT, examples TEXT, representative TEXT,
                PRIMARY KEY(set_id,type,pocket_id));
            CREATE TABLE mode_receptors(set_id TEXT, type TEXT, pocket_id INTEGER, key TEXT,
                PRIMARY KEY(set_id,type,pocket_id,key));
        ''')
        with cluster_csv.open(newline='') as stream:
            for row in csv.DictReader(stream):
                members = ast.literal_eval(row['seq_name'])
                set_id = row['cluster_names']
                if len(members) != int(row['cluster_size']) or len(members) != len(set(members)) or set_id not in members:
                    raise ValueError('Invalid structure-set membership: ' + set_id)
                db.execute('INSERT INTO structure_sets VALUES (?,?)', (set_id, len(members)))
                db.executemany('INSERT INTO members VALUES (?,?,?)',
                               [(key, chain_name(key), set_id) for key in members])
        member_sets = dict(db.execute('SELECT key,set_id FROM members'))
        sources = {path.name: {'sha256': digest(path), 'size_bytes': path.stat().st_size} for path in paths}
        summary = {}
        ones = re.compile('1')
        for kind in TYPES:
            groups = {}
            with gzip.open(tables / (kind + '_pocket_interactions.csv.gz'), 'rt', newline='') as stream:
                for row in csv.DictReader(stream):
                    key = row['cluster_name']
                    receptor = row['receptor_key']
                    if member_sets.get(receptor) != key:
                        raise ValueError('Contact receptor is outside its structure set: ' + receptor)
                    group = groups.setdefault(key, dict(ref_id=row['ref_id'], ref_seq=row['ref_seq'],
                                                         receptors=set(), records=0, modes={}))
                    if (group['ref_id'], group['ref_seq']) != (row['ref_id'], row['ref_seq']):
                        raise ValueError('Inconsistent reference: ' + key)
                    if member_sets.get(row['ref_id']) != key:
                        raise ValueError('Reference is outside its structure set: ' + key)
                    label = row['ref_label']
                    if len(label) != len(group['ref_seq']) or not label or set(label) - {'0', '1'} or '1' not in label:
                        raise ValueError('Invalid/empty projected label: ' + row['ckey'])
                    pid = int(row['pocket_id'])
                    if pid < 0:
                        raise ValueError('Negative pocket ID')
                    mode = group['modes'].setdefault(pid, dict(receptors=set(), records=0, coverage=Counter(), examples=[], vectors={}))
                    mode['receptors'].add(receptor)
                    mode['records'] += 1
                    mode['coverage'].update(match.start() + 1 for match in ones.finditer(label))
                    bits = int(label, 2)
                    if bits not in mode['vectors']:
                        mode['vectors'][bits] = [0, dict(ckey=row['ckey'], receptor_key=receptor, partner_key=row['ligand_key'])]
                    mode['vectors'][bits][0] += 1
                    if len(mode['examples']) < 3:
                        mode['examples'].append(dict(ckey=row['ckey'], receptor_key=receptor, partner_key=row['ligand_key']))
                    group['receptors'].add(receptor)
                    group['records'] += 1
            counts = dict(structure_sets=len(groups), modes=0, modes_at_least_4_residues=0,
                          modes_at_least_4_residues_support_gt_0_1=0, contact_records=0)
            for key, group in groups.items():
                db.execute('INSERT INTO interfaces VALUES (?,?,?,?,?,?)',
                           (key, kind, group['ref_id'], group['ref_seq'], len(group['receptors']), group['records']))
                counts['contact_records'] += group['records']
                for pid, mode in group['modes'].items():
                    positions = sorted(mode['coverage'])
                    consensus = sorted(pos for pos, n in mode['coverage'].items() if 2*n >= mode['records'])
                    representative = contact_medoid(mode['vectors'])
                    db.execute('INSERT INTO modes VALUES (?,?,?,?,?,?,?,?,?,?)',
                               (key, kind, pid, mode['records'], len(mode['receptors']), len(positions),
                                json.dumps(positions), json.dumps(consensus), json.dumps(mode['examples']), json.dumps(representative)))
                    db.executemany('INSERT INTO mode_receptors VALUES (?,?,?,?)',
                                   [(key, kind, pid, receptor) for receptor in sorted(mode['receptors'])])
                    counts['modes'] += 1
                    if len(positions) >= 4:
                        counts['modes_at_least_4_residues'] += 1
                        if len(mode['receptors']) / len(group['receptors']) > .1:
                            counts['modes_at_least_4_residues_support_gt_0_1'] += 1
            summary[kind] = counts
            db.commit()
            print(kind + ': ' + json.dumps(counts), file=sys.stderr, flush=True)
            del groups
        manifest = dict(schema=SCHEMA, structure_sets=db.execute('SELECT count(*) FROM structure_sets').fetchone()[0],
                        member_structures=len(member_sets), sources=sources, interfaces=summary,
                        residue_definition='Union of projected contact positions within the existing mode, reference sequence, 1-based',
                        support_definition='Distinct receptor keys in mode / distinct receptor keys with retained contacts in set and interface',
                        contact_grouping='Existing pocket_id, unchanged; no reclustering', default_min_residues=4,
                        representative_definition='Original contact record minimizing mean Jaccard distance to all records in that mode, duplicates weighted by their record count; self-distance zero; ties within 1e-12 mean distance retain input order',
                        default_support_strictly_greater_than=0)
        db.execute('INSERT INTO metadata VALUES (?,?)', ('schema', SCHEMA))
        db.execute('INSERT INTO metadata VALUES (?,?)', ('manifest', json.dumps(manifest)))
        db.commit()
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('SQLite integrity check failed')
        db.close()
        # Exclusive creation also protects against another build choosing this output.
        with output.open('xb') as stream, temporary.open('rb') as source:
            import shutil
            shutil.copyfileobj(source, stream)
        manifest['database_sha256'] = digest(output)
        output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        return manifest
    finally:
        db.close()
        temporary.unlink(missing_ok=True)


def lookup(path, chain, min_residues=4, min_support=0):
    if min_residues < 1 or not 0 <= min_support <= 1:
        raise ValueError('min-residues must be positive; min-support must be within [0,1]')
    chain = chain.strip()
    if not chain:
        raise ValueError('Enter a chain name or full structure key')
    with closing(connect(path)) as db:
        field = 'key' if '/' in chain else 'chain'
        if field == 'chain':
            parts = chain.split('_', 1)
            if len(parts) != 2:
                raise ValueError('Use PDB_chain (6CBZ_B) or bucket/PDB/chain:model')
            chain = parts[0].upper() + '_' + parts[1]
        matches = list(db.execute(f'SELECT key,set_id FROM members WHERE {field}=? ORDER BY key', (chain,)))
        match_policy = 'exact_chain_case'
        if not matches:
            matches = list(db.execute(f'SELECT key,set_id FROM members WHERE lower({field})=lower(?) ORDER BY key', (chain,)))
            match_policy = 'case_insensitive_fallback; verify chain identity'
        grouped = {}
        for row in matches:
            grouped.setdefault(row['set_id'], []).append(row['key'])
        results = []
        for set_id, keys in grouped.items():
            members = [row[0] for row in db.execute('SELECT key FROM members WHERE set_id=? ORDER BY key', (set_id,))]
            interfaces = {}
            for kind in TYPES:
                info = db.execute('SELECT * FROM interfaces WHERE set_id=? AND type=?', (set_id, kind)).fetchone()
                modes = []
                total = 0
                if info:
                    for row in db.execute('SELECT * FROM modes WHERE set_id=? AND type=? ORDER BY pocket_id', (set_id, kind)):
                        total += 1
                        support = row['receptor_count'] / info['receptor_count']
                        if row['residue_count'] < min_residues or support <= min_support:
                            continue
                        receptors = [r[0] for r in db.execute('SELECT key FROM mode_receptors WHERE set_id=? AND type=? AND pocket_id=? ORDER BY key', (set_id, kind, row['pocket_id']))]
                        positions = json.loads(row['positions'])
                        representative = json.loads(row['representative'])
                        modes.append(dict(pocket_id=row['pocket_id'], mode_id=row['pocket_id']+1,
                                          representative_chain=chain_name(representative['receptor_key']),
                                          representative=representative,
                                          residue_count=row['residue_count'], reference_positions_1based=positions,
                                          reference_residues=[info['ref_seq'][pos-1]+str(pos) for pos in positions],
                                          consensus_positions_1based=json.loads(row['consensus_positions']),
                                          contact_records=row['record_count'], supporting_receptors=row['receptor_count'],
                                          support=support, receptor_keys=receptors,
                                          query_receptor_keys=sorted(set(keys) & set(receptors)),
                                          examples=json.loads(row['examples'])))
                interfaces[kind] = dict(reference_key=info['ref_id'] if info else None,
                                        reference_sequence=info['ref_seq'] if info else None,
                                        denominator_receptors=info['receptor_count'] if info else 0,
                                        total_modes=total, retained_modes=len(modes), modes=modes)
            results.append(dict(structure_set=set_id, member_count=len(members), members=members,
                                matched_structure_keys=keys, interfaces=interfaces))
        return dict(query=chain, match_policy=match_policy, min_residues=min_residues,
                    support_strictly_greater_than=min_support,
                    positions='1-based reference sequence; not PDB residue numbers', results=results)


def report(result):
    lines = [f"Query: {result['query']} ({result['match_policy']})",
             f"Filter: union residues >= {result['min_residues']}; support > {result['support_strictly_greater_than']}",
             'Positions: 1-based reference sequence, NOT PDB residue numbers.']
    if not result['results']:
        lines.append('No matching chain in the indexed structure-set membership.')
    for group in result['results']:
        lines += ['', f"Structure set: {group['structure_set']} ({group['member_count']} members)",
                  'Matched keys: ' + ', '.join(group['matched_structure_keys'])]
        for kind, info in group['interfaces'].items():
            lines.append(f"  {kind}: {info['retained_modes']}/{info['total_modes']} modes; ref={info['reference_key']}")
            for mode in info['modes']:
                lines.append(f"    mode {mode['mode_id']} (pocket_id={mode['pocket_id']}): "
                             f"{mode['residue_count']} residues; support={mode['support']:.4f} "
                             f"({mode['supporting_receptors']}/{info['denominator_receptors']}); "
                             f"records={mode['contact_records']}; representative={mode['representative_chain']}; "
                             f"query chain observed={bool(mode['query_receptor_keys'])}")
                lines.append('      ' + ', '.join(mode['reference_residues']))
    return '\n'.join(lines)


def export_modes(path, output, min_residues=4, min_support=0):
    """Export one row per retained mode across all structure sets/types."""
    if min_residues < 1 or not 0 <= min_support <= 1:
        raise ValueError('Invalid residue/support filter')
    count = 0
    with closing(connect(path)) as db, output.open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['structure_set', 'interface_type', 'reference_key', 'pocket_id', 'mode_id',
                         'representative_chain', 'representative_key', 'representative_contact', 'mean_jaccard_distance',
                         'participating_residues', 'supporting_receptors', 'denominator_receptors',
                         'support', 'contact_records', 'reference_positions_1based'])
        rows = db.execute('''SELECT m.*, i.ref_id, i.receptor_count AS denominator
            FROM modes m JOIN interfaces i ON m.set_id=i.set_id AND m.type=i.type
            WHERE m.residue_count>=? ORDER BY m.set_id,m.type,m.pocket_id''', (min_residues,))
        for row in rows:
            support = row['receptor_count']/row['denominator']
            if support <= min_support:
                continue
            representative = json.loads(row['representative'])
            writer.writerow([row['set_id'], row['type'], row['ref_id'], row['pocket_id'], row['pocket_id']+1,
                             chain_name(representative['receptor_key']), representative['receptor_key'], representative['ckey'],
                             repr(representative['mean_jaccard_distance']),
                             row['residue_count'], row['receptor_count'], row['denominator'],
                             repr(support), row['record_count'], row['positions']])
            count += 1
    return dict(rows=count, min_residues=min_residues, support_strictly_greater_than=min_support)


def structure_payload(path, h5, set_id, kind, key):
    if h5 is None:
        raise ValueError('Structure display is disabled; restart serve with --h5 /path/to/bio_251210.h5')
    if kind not in TYPES:
        raise ValueError('Unknown interface type')
    with closing(connect(path)) as db:
        if not db.execute('SELECT 1 FROM members WHERE set_id=? AND key=?', (set_id, key)).fetchone():
            raise ValueError('Selected structure is not a member of this structure set')
        info = db.execute('SELECT ref_id,ref_seq FROM interfaces WHERE set_id=? AND type=?', (set_id, kind)).fetchone()
        if info is None:
            raise ValueError('No reference for this interface in the selected structure set')
    try:
        from colabel.binding_structure import read_monomer
    except ModuleNotFoundError as error:
        if error.name != 'colabel':
            raise
        from binding_structure import read_monomer
    return dict(read_monomer(str(h5.resolve()), key, info['ref_seq']),
                structure_set=set_id, interface_type=kind, reference_key=info['ref_id'])


def browser_capabilities(h5):
    file_available = h5 is not None and h5.is_file()
    missing = [name for name in ('numpy', 'h5py', 'Bio') if importlib.util.find_spec(name) is None]
    return dict(reference_tracks=True, structure_available=file_available and not missing,
                h5_available=file_available, missing_structure_dependencies=missing,
                default_h5_location='data/bio_251210.h5')


def serve(path, port, h5=None):
    page = Path(__file__).with_name('binding_modes.html').read_bytes()
    capabilities = browser_capabilities(h5)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path == '/':
                data, content_type, status = page, 'text/html; charset=utf-8', 200
            elif url.path == '/api/capabilities':
                data, content_type, status = json.dumps(capabilities).encode(), 'application/json; charset=utf-8', 200
            elif url.path in ('/static/3Dmol-min.js', '/static/binding_viewer.js'):
                asset = Path(__file__).with_name('static')/url.path.rsplit('/', 1)[1]
                if asset.is_file():
                    data, content_type, status = asset.read_bytes(), 'text/javascript; charset=utf-8', 200
                else:
                    data, content_type, status = b'Missing local viewer asset', 'text/plain', 404
            elif url.path == '/api/structure':
                try:
                    params = parse_qs(url.query)
                    result = structure_payload(path, h5, params.get('set', [''])[0],
                                               params.get('type', [''])[0], params.get('key', [''])[0])
                    data, status = json.dumps(result).encode(), 200
                except (ValueError, OSError, ImportError, sqlite3.Error) as error:
                    data, status = json.dumps({'error': str(error)}).encode(), 400
                content_type = 'application/json; charset=utf-8'
            elif url.path == '/api/search':
                try:
                    params = parse_qs(url.query)
                    result = lookup(path, params.get('chain', [''])[0],
                                    int(params.get('min_residues', ['4'])[0]),
                                    float(params.get('min_support', ['0'])[0]))
                    data, status = json.dumps(result).encode(), 200
                except (ValueError, sqlite3.Error) as error:
                    data, status = json.dumps({'error': str(error)}).encode(), 400
                content_type = 'application/json; charset=utf-8'
            else:
                data, content_type, status = b'Not found', 'text/plain', 404
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store' if url.path == '/' or url.path.startswith('/api/') else 'public, max-age=86400')
            self.end_headers()
            self.wfile.write(data)
    with closing(connect(path)):
        pass
    print(f'Binding-mode lookup: http://127.0.0.1:{port} (SSH: forward the same port)', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build_parser = commands.add_parser('build')
    build_parser.add_argument('--clusters', type=Path, required=True)
    build_parser.add_argument('--tables', type=Path, required=True)
    build_parser.add_argument('--output', type=Path, required=True)
    query_parser = commands.add_parser('query')
    query_parser.add_argument('chain')
    query_parser.add_argument('--db', type=Path, required=True)
    query_parser.add_argument('--min-residues', type=int, default=4)
    query_parser.add_argument('--min-support', type=float, default=0,
                              help='Strictly greater than this support; 0.1 reproduces the old support filter')
    query_parser.add_argument('--json', action='store_true')
    query_parser.add_argument('--output', type=Path)
    export_parser = commands.add_parser('export')
    export_parser.add_argument('--db', type=Path, required=True)
    export_parser.add_argument('--output', type=Path, required=True)
    export_parser.add_argument('--min-residues', type=int, default=4)
    export_parser.add_argument('--min-support', type=float, default=0)
    serve_parser = commands.add_parser('serve')
    serve_parser.add_argument('--db', type=Path, required=True)
    serve_parser.add_argument('--port', type=int, default=8765)
    coordinate_options = serve_parser.add_mutually_exclusive_group()
    coordinate_options.add_argument('--h5', type=Path, default=Path(__file__).resolve().parents[1]/'data/bio_251210.h5',
                              help='Read-only native H5 coordinates for the optional monomer cartoon view')
    coordinate_options.add_argument('--index-only', action='store_true',
                                    help='Use reference-sequence tracks without H5, coordinate dependencies or WebGL')
    args = parser.parse_args()
    if args.command == 'build':
        print(json.dumps(build(args.clusters, args.tables, args.output), indent=2))
    elif args.command == 'export':
        print(json.dumps(export_modes(args.db, args.output, args.min_residues, args.min_support)))
    elif args.command == 'query':
        result = lookup(args.db, args.chain, args.min_residues, args.min_support)
        text = json.dumps(result, indent=2) if args.json else report(result)
        if args.output:
            with args.output.open('x') as stream:
                stream.write(text + '\n')
        else:
            print(text)
    else:
        serve(args.db, args.port, args.h5 if not args.index_only and args.h5.is_file() else None)


if __name__ == '__main__':
    main()
