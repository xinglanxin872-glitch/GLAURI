"""Read one H5 monomer and map reference contact positions onto real coordinates.

Imported only for the optional structure API. No model/checkpoint is needed.
H5 lacks original PDB residue numbering: display residue IDs are sequential H5
structure-sequence positions, explicitly distinct from reference-sequence IDs.
"""
from functools import lru_cache
import numpy as np
import h5py

try:
    from .contact_io import AA
    from .leftmost_align import leftmost_align
except ImportError:  # CLI executed as python colabel/binding_modes.py
    from contact_io import AA
    from leftmost_align import leftmost_align


def decode_names(values):
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


@lru_cache(maxsize=24)
def read_monomer(h5_path, key, reference_sequence):
    with h5py.File(h5_path, 'r') as handle:
        path = 'data/structures/' + key
        if path not in handle:
            raise ValueError('No coordinates in the H5 for this structure key: ' + key)
        group = handle[path]
        xyz = np.asarray(group['X'], dtype=float)
        columns = {}
        for name in ('M', 'qn', 'qr', 'qe'):
            sparse = np.asarray(group[name])
            if sparse.shape != (len(xyz), 2) or not np.array_equal(sparse[:, 0], np.arange(len(xyz))):
                raise ValueError('Unexpected H5 sparse atom order: ' + name)
            columns[name] = sparse[:, 1].astype(int)
        atom_names = decode_names(handle['metadata/std_names'][:]) + ['UNK']
        residue_names = decode_names(handle['metadata/std_resnames'][:]) + ['UNX']
        elements = decode_names(handle['metadata/std_elements'][:]) + ['X']
    if xyz.shape != (len(xyz), 3) or not np.isfinite(xyz).all():
        raise ValueError('Invalid atomic coordinates')
    _, first = np.unique(columns['M'], return_index=True)
    first = np.sort(first)
    if not len(first) or len(first) > 9999 or len(xyz) > 99999:
        raise ValueError('Structure exceeds the monomer PDB display limits or is empty')
    order = {int(columns['M'][i]): pos+1 for pos, i in enumerate(first)}
    sequence = ''.join(AA.get(residue_names[columns['qr'][i]], 'X') for i in first)
    rp, cp = leftmost_align(sequence, reference_sequence)
    to_reference = {int(c)+1: int(r)+1 for r, c in zip(rp, cp)}
    residues = [dict(structure_position=i+1, resname=residue_names[columns['qr'][atom]],
                     amino_acid=sequence[i], reference_position=to_reference.get(i+1), ca=None)
                for i, atom in enumerate(first)]
    pdb = []
    for index, coord in enumerate(xyz):
        atom = atom_names[columns['qn'][index]]
        resname = residue_names[columns['qr'][index]]
        element = elements[columns['qe'][index]]
        position = order[int(columns['M'][index])]
        if atom == 'CA' and residues[position-1]['ca'] is None:
            residues[position-1]['ca'] = [float(value) for value in coord]
        # Single chain A, original coordinates, sequential structure positions.
        name = atom.rjust(4) if len(atom) == 4 else (' '+atom.ljust(3))
        pdb.append(f'ATOM  {index+1:5d} {name} {resname:>3s} A{position:4d}    '
                   f'{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00          {element:>2s}  ')
    matches = sum(sequence[int(c)] == reference_sequence[int(r)] for r, c in zip(rp, cp))
    return dict(structure_key=key, structure_sequence=sequence, reference_length=len(reference_sequence),
                reference_to_structure=[[int(r)+1, int(c)+1] for r, c in zip(rp, cp)],
                aligned_positions=len(rp), aligned_identity=matches/len(rp) if len(rp) else None,
                residues=residues, atom_count=len(xyz), pdb='\n'.join(pdb)+'\nTER\nEND\n',
                numbering='Structure-sequence positions, 1-based; original PDB residue numbers are not stored in this H5',
                coordinates='Original H5 monomer coordinates; no invented coordinates for missing reference residues')
