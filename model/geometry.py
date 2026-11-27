"""Shared residue geometry for training, mask construction and inference."""
import numpy as np
import torch
from graphein.protein.tensor.angles import dihedrals

CA_PROTOCOL = 'ca_ncac_sidechain_v1'
LEGACY_PROTOCOL = 'legacy_nodes_angles_sidechain_v1_20261102'

def select_backbone(atom_residue, atom_type, n_residues, protocol=CA_PROTOCOL):
    if protocol not in (CA_PROTOCOL, LEGACY_PROTOCOL):
        raise ValueError(f'Unknown geometry protocol: {protocol}')
    atom_residue=np.asarray(atom_residue,dtype=np.int64)
    atom_type=np.asarray(atom_type,dtype=np.int64)
    # Vocabulary is CA=0, N=1, C=2. The historical protocol swapped CA and N.
    order=(1,0,2) if protocol==CA_PROTOCOL else (0,1,2)
    counts=np.zeros((n_residues,3),np.int32)
    indices=np.full((n_residues,3),-1,np.int64)
    for col,kind in enumerate(order):
        ids=np.flatnonzero(atom_type==kind); resid=atom_residue[ids]
        np.add.at(counts[:,col],resid,1); indices[resid,col]=ids
    residues=np.flatnonzero((counts==1).all(axis=1))
    if not len(residues):raise ValueError('No residues with unique N, CA and C atoms')
    return residues, indices[residues]

def residue_geometry(X, atom_residue, atom_type, n_residues, protocol=CA_PROTOCOL):
    X=torch.as_tensor(X,dtype=torch.float32)
    residues,indices=select_backbone(atom_residue,atom_type,n_residues,protocol)
    backbone=X[torch.from_numpy(indices)]
    node_indices=indices[:,1]
    positions=X[torch.from_numpy(node_indices)]
    vectors=torch.zeros_like(positions)
    atom_residue=np.asarray(atom_residue);atom_type=np.asarray(atom_type)
    side=(atom_type>=4)&(atom_type<=35)
    sums=torch.zeros((n_residues,3),dtype=X.dtype)
    counts=torch.zeros(n_residues,dtype=X.dtype)
    ri=torch.from_numpy(atom_residue[side]).long()
    sums.index_add_(0,ri,X[torch.from_numpy(np.flatnonzero(side))])
    counts.index_add_(0,ri,torch.ones(len(ri)))
    selected_counts=counts[torch.from_numpy(residues)]
    available=selected_counts>0
    vectors[available]=sums[torch.from_numpy(residues)][available]/selected_counts[available,None]-positions[available]
    norms=vectors.norm(dim=1,keepdim=True)
    vectors=torch.where(norms>1e-6,vectors/norms.clamp_min(1e-6),torch.zeros_like(vectors))
    angles=dihedrals(backbone,sparse=True)
    return dict(residues=residues,node_indices=node_indices,positions=positions,
                backbone=backbone,angles=angles,sidechain=vectors,protocol=protocol)
