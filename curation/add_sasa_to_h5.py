"""
  python add_sasa_to_h5.py --limit 64 --workers 4 --out /tmp/rsa_test.h5   # small-scale test
  python add_sasa_to_h5.py --workers 32                                    # full run (resumable: re-run skips completed keys)
"""

import argparse
import json
import os
import time
from datetime import datetime
from multiprocessing import Pool

import h5py
import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Constant tables
# ---------------------------------------------------------------------------

# Tien et al. 2013, Table 1 "Theoretical" column, unit Å²
MAX_ASA = {
    'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0, 'CYS': 167.0,
    'GLU': 223.0, 'GLN': 225.0, 'GLY': 104.0, 'HIS': 224.0, 'ILE': 197.0,
    'LEU': 201.0, 'LYS': 236.0, 'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0,
    'SER': 155.0, 'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0,
}
# Fallback max ASA for nucleotides (G/A/C/U/DG/DA/DT/DC) and unknown residues (the last qr class).
# A conservative value just above TRP's ~285; the model only needs a consistent scale for these classes.
FALLBACK_MAX_ASA = 290.0

# Atomic vdW radii (Å); Bondi (1964) for main-group elements. Metals have no Bondi standard value,
# so common force-field approximations are used — metals are extremely rare in protein chains, and a ±0.2 Å radius error has negligible effect on RSA.
VDW_RADII = {
    'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80, 'F': 1.47,
    'Cl': 1.75, 'Br': 1.85, 'I': 1.98, 'Se': 1.90, 'As': 1.85, 'B': 1.92,
    'Na': 2.27, 'K': 2.75, 'Mg': 1.73, 'Ca': 2.31, 'Sr': 2.49, 'Ba': 2.68,
    'Zn': 1.39, 'Fe': 1.56, 'Mn': 1.61, 'Cu': 1.40, 'Ni': 1.63, 'Co': 1.58,
    'Cd': 1.58, 'Hg': 1.55, 'W': 2.10, 'Mo': 2.10, 'Pt': 1.75,
}
FALLBACK_RADIUS = 1.70  # qe unknown element class -> treat as carbon

PROBE_RADIUS = 1.4      # water probe radius, Å
N_SPHERE_POINTS = 960   # consistent with the original paper / freesasa default

# ---------------------------------------------------------------------------
# Shrake-Rupley
# ---------------------------------------------------------------------------


def fibonacci_sphere(n):
    """Approximately uniform points on the unit sphere (golden-angle spiral)."""
    i = np.arange(n, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (2.0 * i + 1.0) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * i
    return np.stack([r * np.cos(theta), y, r * np.sin(theta)], axis=1)


_SPHERE_PTS = fibonacci_sphere(N_SPHERE_POINTS)


def shrake_rupley_per_atom(X, radii, probe=PROBE_RADIUS, atom_chunk=128):
    """X: (N,3) Å; radii: (N,) vdW radii. Returns per-atom SASA (N,) Å².

    Fully vectorized implementation: the neighbor table is padded to (N, J_max), and per atom chunk (atom_chunk)
    the (m, P, J) distance matrix is expanded in one pass for the occlusion test, keeping peak memory bounded.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    R = (radii + probe).astype(np.float64)
    n = X.shape[0]
    r_max = float(2.0 * R.max())
    pairs = cKDTree(X).query_pairs(r_max, output_type='ndarray')

    # ---- build the padded neighbor table (n, J_max) vectorized; -1 = empty slot ----
    if pairs.size:
        src = np.concatenate([pairs[:, 0], pairs[:, 1]])
        dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
        order = np.argsort(src, kind='stable')
        src_s, dst_s = src[order], dst[order]
    else:
        src_s = np.zeros(0, dtype=np.int64)
        dst_s = np.zeros(0, dtype=np.int64)
    counts = np.bincount(src_s, minlength=n)
    j_max = int(counts.max()) if counts.size else 0
    nbr = np.full((n, max(j_max, 1)), -1, dtype=np.int64)
    if j_max:
        starts = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=starts[1:])
        pos = np.arange(dst_s.shape[0]) - starts[src_s]
        nbr[src_s, pos] = dst_s

    # ---- chunked occlusion test: d2 = |p|² + |x|² - 2 p·x ----
    pts_sphere = _SPHERE_PTS.astype(np.float64)
    sasa = np.zeros(n, dtype=np.float64)
    const = 4.0 * np.pi / N_SPHERE_POINTS
    for lo in range(0, n, atom_chunk):
        hi = min(lo + atom_chunk, n)
        nb = nbr[lo:hi]                                   # (m,J)
        valid = nb >= 0
        # neighbor columns actually used by this chunk (drop all-empty trailing columns, so the dense core's J_max doesn't burden surface chunks)
        cols = np.where(valid.any(axis=0))[0]
        if cols.size == 0:
            sasa[lo:hi] = const * R[lo:hi] ** 2 * N_SPHERE_POINTS
            continue
        nb = nb[:, :cols[-1] + 1]
        valid = valid[:, :cols[-1] + 1]
        nbc = np.where(valid, nb, 0)
        # float32: halves bandwidth and speeds up GEMM; the precision loss in the threshold comparison (~0.01 Å) is far below the sphere-sampling noise
        pts = (X[lo:hi, None, :] + R[lo:hi, None, None] * pts_sphere[None, :, :]).astype(np.float32)  # (m,P,3)
        Xn = X[nbc].astype(np.float32)                            # (m,J,3)
        d2 = ((pts ** 2).sum(-1)[:, :, None] + (Xn ** 2).sum(-1)[:, None, :]
              - 2.0 * np.matmul(pts, Xn.transpose(0, 2, 1)))  # (m,P,J)
        buried = ((d2 < (R[nbc] ** 2)[:, None, :]) & valid[:, None, :]).any(axis=-1)
        sasa[lo:hi] = const * R[lo:hi] ** 2 * (N_SPHERE_POINTS - buried.sum(axis=1))
    return sasa


def compute_rsa(g, elem_radius_lut, resname_maxasa_lut):
    """Compute per-residue RSA from an h5 structure group; returns float32 (N_res,)."""
    M = np.asarray(g['M'])              # (N,2) (atom_idx, res_idx)
    n_res = int(g.attrs['M_shape'][1])
    res_of_atom = M[:, 1].astype(np.int64)
    qe = np.asarray(g['qe'])[:, 1]      # per-atom element class index
    qr = np.asarray(g['qr'])[:, 1]      # per-atom residue-name class index
    X = np.asarray(g['X'], dtype=np.float64)

    atom_sasa = shrake_rupley_per_atom(X, elem_radius_lut[qe])

    res_sasa = np.zeros(n_res, dtype=np.float64)
    np.add.at(res_sasa, res_of_atom, atom_sasa)

    # take the residue-name class of each residue's first atom (all atoms in the same residue share the class)
    _, first = np.unique(res_of_atom, return_index=True)
    res_maxasa = resname_maxasa_lut[qr[first]]

    rsa = res_sasa / res_maxasa
    return np.clip(rsa, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# GPU path (torch): same Shrake-Rupley math, single process reads h5 + GPU compute
# ---------------------------------------------------------------------------

_SPHERE_PTS_T = None   # lazily initialized torch version of the sphere points


def shrake_rupley_per_atom_gpu(X_t, R_t, atom_chunk=1024):
    """X_t: (N,3) float32 cuda; R_t: (N,) = vdW+probe. Returns per-atom SASA (N,) float32."""
    global _SPHERE_PTS_T
    import torch
    if _SPHERE_PTS_T is None or _SPHERE_PTS_T.device != X_t.device:
        _SPHERE_PTS_T = torch.from_numpy(_SPHERE_PTS).to(X_t.device, torch.float32)
    pts_sphere = _SPHERE_PTS_T
    P = pts_sphere.shape[0]
    n = X_t.shape[0]

    # neighbor table: D(i,j) < R_i + R_j
    D = torch.cdist(X_t, X_t)
    nbr_mask = (D < (R_t[:, None] + R_t[None, :]))
    nbr_mask.fill_diagonal_(False)
    counts = nbr_mask.sum(1)
    j_max = int(counts.max())
    sasa = torch.empty(n, dtype=torch.float32, device=X_t.device)
    const = 4.0 * np.pi / P
    if j_max == 0:
        return const * R_t ** 2 * P
    # per-row padded indices of the first j_max neighbors (sort puts True first)
    order = torch.argsort((~nbr_mask).to(torch.uint8), dim=1, stable=True)
    nbr = order[:, :j_max]                                   # (n,J)
    valid = nbr_mask.gather(1, nbr)                          # (n,J)

    for lo in range(0, n, atom_chunk):
        hi = min(lo + atom_chunk, n)
        nb = nbr[lo:hi]
        va = valid[lo:hi]
        cols = torch.where(va.any(0))[0]
        if cols.numel() == 0:
            sasa[lo:hi] = const * R_t[lo:hi] ** 2 * P
            continue
        nb = nb[:, :int(cols[-1]) + 1]
        va = va[:, :int(cols[-1]) + 1]
        pts = X_t[lo:hi, None, :] + R_t[lo:hi, None, None] * pts_sphere[None, :, :]  # (m,P,3)
        Xn = X_t[nb]                                          # (m,J,3)
        d2 = ((pts ** 2).sum(-1)[:, :, None] + (Xn ** 2).sum(-1)[:, None, :]
              - 2.0 * torch.matmul(pts, Xn.transpose(1, 2)))  # (m,P,J)
        buried = ((d2 < (R_t[nb] ** 2)[:, None, :]) & va[:, None, :]).any(-1)
        sasa[lo:hi] = const * R_t[lo:hi] ** 2 * (P - buried.sum(-1).to(torch.float32))
    return sasa


def compute_rsa_gpu(g, elem_radius_lut, resname_maxasa_lut, device):
    """GPU version of compute_rsa; returns float32 (N_res,) numpy."""
    import torch
    M = np.asarray(g['M'])
    n_res = int(g.attrs['M_shape'][1])
    res_of_atom = M[:, 1].astype(np.int64)
    qe = np.asarray(g['qe'])[:, 1]
    qr = np.asarray(g['qr'])[:, 1]
    X = np.asarray(g['X'], dtype=np.float32)

    X_t = torch.from_numpy(X).to(device)
    R_t = torch.from_numpy(elem_radius_lut[qe].astype(np.float32) + PROBE_RADIUS).to(device)
    atom_sasa = shrake_rupley_per_atom_gpu(X_t, R_t).cpu().numpy().astype(np.float64)

    res_sasa = np.zeros(n_res, dtype=np.float64)
    np.add.at(res_sasa, res_of_atom, atom_sasa)
    _, first = np.unique(res_of_atom, return_index=True)
    res_maxasa = resname_maxasa_lut[qr[first]]
    rsa = res_sasa / res_maxasa
    return np.clip(rsa, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Multiprocessing: workers read the main h5 read-only; only the main process writes the sidecar
# ---------------------------------------------------------------------------

_H5 = None          # worker-global read-only handle
_ELEM_LUT = None
_RES_LUT = None


def _worker_init(h5_path):
    global _H5, _ELEM_LUT, _RES_LUT
    # limit each worker's BLAS to a single thread to avoid multiprocessing x multithreading oversubscription
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    _H5 = h5py.File(h5_path, 'r')
    std_elements = np.array(_H5['metadata/std_elements']).astype('U')
    std_resnames = np.array(_H5['metadata/std_resnames']).astype('U')
    # LUT length = number of classes + 1 (the last class is unknown)
    _ELEM_LUT = np.array([VDW_RADII.get(e, FALLBACK_RADIUS) for e in std_elements]
                         + [FALLBACK_RADIUS], dtype=np.float64)
    _RES_LUT = np.array([MAX_ASA.get(r, FALLBACK_MAX_ASA) for r in std_resnames]
                        + [FALLBACK_MAX_ASA], dtype=np.float64)


def _worker(keys):
    out = []
    for key in keys:
        try:
            rsa = compute_rsa(_H5['data/structures/' + key], _ELEM_LUT, _RES_LUT)
            out.append((key, rsa))
        except Exception as e:  # a single-structure failure must not sink the batch; failures are reported together at the end
            out.append((key, e))
    return out


def collect_done(out_path):
    """Resume support: skip keys that already have an rsa dataset in the sidecar."""
    done = set()
    if os.path.isfile(out_path):
        with h5py.File(out_path, 'r') as fo:
            def _visit(name, obj):
                if isinstance(obj, h5py.Dataset) and name.endswith('/rsa'):
                    done.add(name[:-len('/rsa')])
            fo.visititems(_visit)
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    default_h5 = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'bio_251210.h5'))
    ap.add_argument('--h5', default=default_h5, help='main h5 (read-only)')
    ap.add_argument('--out', default=None, help='sidecar output path; default <main h5 minus .h5>_rsa.h5')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--chunksize', type=int, default=64, help='number of keys per task chunk')
    ap.add_argument('--limit', type=int, default=0, help='compute only the first N keys (for testing)')
    ap.add_argument('--keys-from', default=None, help='.npy/.txt key list; compute only these keys (for testing/backfilling)')
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                    help="cuda=single-process GPU compute (default, requires torch+cuda); cpu=multiprocess CPU")
    args = ap.parse_args()

    if args.out:
        out_path = args.out
    elif args.h5.endswith('.h5'):
        out_path = args.h5[:-3] + '_rsa.h5'
    else:
        out_path = args.h5 + '_rsa.h5'

    t0 = time.time()
    if args.keys_from:
        if args.keys_from.endswith('.npy'):
            all_keys = np.load(args.keys_from).astype('U').tolist()
        else:
            with open(args.keys_from) as fh:
                all_keys = [ln.strip() for ln in fh if ln.strip()]
    else:
        with h5py.File(args.h5, 'r') as hf:
            all_keys = np.unique(np.array(hf['metadata/keys']).astype('U')).tolist()
    if args.limit:
        all_keys = all_keys[:args.limit]
    print(f'[rsa] {len(all_keys)} unique structures in {args.h5}')

    done = collect_done(out_path)
    todo = [k for k in all_keys if ('data/structures/' + k) not in done]
    print(f'[rsa] done={len(done)}  todo={len(todo)}  out={out_path}')
    if not todo:
        return

    chunks = [todo[i:i + args.chunksize] for i in range(0, len(todo), args.chunksize)]

    fo = h5py.File(out_path, 'a')
    fo.attrs['source_h5'] = os.path.abspath(args.h5)
    fo.attrs['probe_radius'] = PROBE_RADIUS
    fo.attrs['n_sphere_points'] = N_SPHERE_POINTS
    fo.attrs['fallback_max_asa'] = FALLBACK_MAX_ASA
    fo.attrs['max_asa_table'] = json.dumps(MAX_ASA)
    fo.attrs['note'] = 'per-residue RSA (Shrake-Rupley, heavy atoms, isolated chain); see add_sasa_to_h5.py'
    fo.attrs['updated'] = datetime.now().isoformat(timespec='seconds')

    n_ok, n_fail, fails = 0, 0, []

    def _write(key, rsa):
        nonlocal n_ok
        grp = fo.require_group('data/structures/' + key)
        if 'rsa' in grp:
            del grp['rsa']
        grp.create_dataset('rsa', data=rsa, compression='lzf')
        n_ok += 1

    def _log(force=False):
        dt = time.time() - t0
        rate = n_ok / max(dt, 1e-9)
        eta = (len(todo) - n_ok - n_fail) / max(rate, 1e-9)
        print(f'[rsa] {n_ok + n_fail}/{len(todo)}  ({rate:.0f}/s, ETA {eta / 60:.1f} min)',
              flush=True)

    if args.device == 'cuda':
        # single process: read h5 sequentially + GPU compute + write sidecar
        import torch
        assert torch.cuda.is_available(), '--device cuda but torch.cuda is not available'
        hf = h5py.File(args.h5, 'r')
        std_elements = np.array(hf['metadata/std_elements']).astype('U')
        std_resnames = np.array(hf['metadata/std_resnames']).astype('U')
        elem_lut = np.array([VDW_RADII.get(e, FALLBACK_RADIUS) for e in std_elements]
                            + [FALLBACK_RADIUS], dtype=np.float64)
        res_lut = np.array([MAX_ASA.get(r, FALLBACK_MAX_ASA) for r in std_resnames]
                           + [FALLBACK_MAX_ASA], dtype=np.float64)
        # Memory fragmentation is prevented via PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
        # Note: calling empty_cache() in this mode triggers an internal assert in torch 2.12-dev's
        # CUDACachingAllocator (observed: 240k structures failed as collateral damage); do not add it back.
        log_every = max(1, len(todo) // 200)
        for i, key in enumerate(todo):
            try:
                rsa = compute_rsa_gpu(hf['data/structures/' + key], elem_lut, res_lut,
                                      torch.device('cuda'))
                _write(key, rsa)
            except Exception as e:
                n_fail += 1
                fails.append((key, repr(e)))
            if i % log_every == 0:
                fo.flush()
                _log()
        hf.close()
    else:
        chunks = [todo[i:i + args.chunksize] for i in range(0, len(todo), args.chunksize)]
        log_every = max(1, len(chunks) // 200)   # ~0.5% granularity
        with Pool(args.workers, initializer=_worker_init, initargs=(args.h5,)) as pool:
            for ci, results in enumerate(pool.imap_unordered(_worker, chunks)):
                for key, rsa in results:
                    if isinstance(rsa, Exception):
                        n_fail += 1
                        fails.append((key, repr(rsa)))
                        continue
                    _write(key, rsa)
                if ci % log_every == 0:
                    fo.flush()
                    _log()
    fo.close()

    print(f'[rsa] finished: ok={n_ok} fail={n_fail}  elapsed={(time.time() - t0) / 60:.1f} min')
    if fails:
        print('[rsa] FAILED keys (training will error when reading these keys; fix them and re-run):')
        for key, msg in fails[:50]:
            print('  ', key, msg)
        if len(fails) > 50:
            print(f'   ... {len(fails)} in total')


if __name__ == '__main__':
    main()
