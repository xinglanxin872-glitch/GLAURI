"""Read native contacts and sequences without importing the training model.

Metadata selection and contact-class definitions match the September 2026
MMseqs-longest pipeline. All HDF5 access is read-only.
"""
from collections import OrderedDict

import h5py
import numpy as np

AA = dict(zip(
    'ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(),
    'ARNDCQEGHILKMFPSTWYV',
))
PROTEIN = tuple(AA)
PARTNERS = (
    PROTEIN,
    ('DA', 'DT', 'DG', 'DC', 'A', 'U', 'G', 'C'),
    ('MG', 'ZN', 'CA', 'MN', 'CD', 'CU', 'FE', 'FE2', 'NI', 'SR', 'CO', 'HG', 'SF4', 'F3S', 'FES'),
    ('ligand',),
)


def parse_ckey(ckey):
    p = str(ckey).split('/')
    if len(p) != 4:
        raise ValueError(f'Invalid contact key: {ckey!r}')
    return '/'.join(p[:3]), '/'.join(p[:2] + [p[3]])


def reference_sequence(key, sequence_dict):
    p = key.split('/')
    pdb, chain = p[1].upper(), p[2].split(':')[0]
    for candidate in (f'{pdb}_{chain}', f'{pdb}_{chain.lower()}', f'{pdb}_{chain.upper()}'):
        if candidate in sequence_dict:
            return str(sequence_dict[candidate])
    raise KeyError(f'No full reference sequence for {key}')


def choose_ref_id(members, mapping, sequence_dict):
    """Longest usable full sequence; retain input member order on length ties."""
    usable = []
    for key in members:
        if key in mapping:
            try:
                usable.append((key, len(reference_sequence(key, sequence_dict))))
            except KeyError:
                continue
    if not usable:
        raise ValueError('No usable cluster member has a full reference sequence')
    return max(usable, key=lambda item: item[1])[0]


class H5Contacts:
    def __init__(self, h5_path, min_num_res=48, max_atoms=None):
        self.path = str(h5_path)
        with h5py.File(self.path, 'r') as hf:
            keys = hf['metadata/keys'][:].astype(str)
            ckeys = hf['metadata/ckeys'][:].astype(str)
            sizes = hf['metadata/sizes'][:]
            ctypes = hf['metadata/ctypes'][:]
            mids = hf['metadata/mids'][:].astype(str)
            self.resnames = hf['metadata/std_resnames'][:].astype(str)
        self.protein_ids = np.flatnonzero(np.isin(mids, PROTEIN))
        self.partner_ids = [np.flatnonzero(np.isin(mids, types)) for types in PARTNERS]
        receptor_rows = ctypes[np.isin(ctypes[:, 1], self.protein_ids), 0].astype(np.int64)
        mask = np.zeros(len(keys), dtype=bool)
        mask[receptor_rows] = True
        mask &= sizes[:, 1] >= min_num_res
        if max_atoms is not None:
            mask &= sizes[:, 0] <= max_atoms
        self.ckeys_map = OrderedDict()
        for key, ckey in zip(keys[mask], ckeys[mask]):
            self.ckeys_map.setdefault(str(key), []).append(str(ckey))
        self.ukeys = list(self.ckeys_map)

    def structure_sequence(self, group):
        """Same first-atom-per-residue convention as model.data_handler.load_seq."""
        m, qr = group['M'][:], group['qr'][:]
        if not len(m):
            return ''
        if len(m) != len(qr) or not np.array_equal(m[:, 0], qr[:, 0]):
            raise ValueError(f'M/qr atom indices differ: {group.name}')
        _, indices = np.unique(m[:, 1], return_index=True)
        codes = qr[np.sort(indices), 1].astype(int)
        return ''.join(AA.get(self.resnames[i], 'X') if 0 <= i < len(self.resnames) else 'X' for i in codes)

    def collect_contacts(self, members, contact_class, hf):
        binding = OrderedDict()
        for key in members:
            for ckey in self.ckeys_map.get(key, []):
                ids = hf[f'data/contacts/{ckey}/Y'][:]
                selected = np.isin(ids[:, 2], self.protein_ids) & np.isin(ids[:, 3], self.partner_ids[contact_class])
                positions = np.unique(ids[selected, 0]).astype(int)
                if len(positions):
                    binding[ckey] = [(int(i), contact_class) for i in positions]
        return binding
