"""Versioned geometry ablation shared by training and inference.

Sequence adjacency retains graph message passing. The alternative 'none' uses
zero edges, not a list of repeated 0->0 edges. No real coordinates are read.
"""
from functools import lru_cache
import torch
from torch import nn

PROTOCOL = 'geometry_mask_v2'

@lru_cache(maxsize=32)
def _sequence_edges_cpu(n, k):
    k = min(k, max(0, n - 1))
    if k == 0:
        return torch.empty((2, 0), dtype=torch.long)
    center = torch.arange(n)
    start = (center - (k + 1) // 2).clamp(0, n - k - 1)
    candidates = start[:, None] + torch.arange(k + 1)[None, :]
    # Rank by (sequence distance, neighbor index), deterministically favoring
    # the preceding residue when an odd k cuts a symmetric tie.
    key = (candidates - center[:, None]).abs() * (n + 1) + candidates
    key[candidates == center[:, None]] = torch.iinfo(torch.long).max
    neighbors = candidates.gather(1, key.argsort(dim=1, stable=True)[:, :k])
    return torch.stack((center.repeat_interleave(k), neighbors.reshape(-1)))

def sequence_edge_index(batch_index, k):
    pieces = []
    for group in torch.unique(batch_index, sorted=True):
        nodes = torch.where(batch_index == group)[0]
        local = _sequence_edges_cpu(len(nodes), k).to(batch_index.device)
        pieces.append(nodes[local])
    return torch.cat(pieces, dim=1) if pieces else torch.empty((2, 0), dtype=torch.long, device=batch_index.device)

class ZeroRBF(nn.Module):
    """Explicitly bypass radial.compute_rbf; zero coordinates alone do not."""
    def __init__(self, num_rbf):
        super().__init__()
        self.num_rbf = int(num_rbf)

    def forward(self, lengths):
        return lengths.new_zeros((lengths.shape[0], self.num_rbf))

def apply_geometry_policy(batch, edge_mode, max_nn, node_vector_channels, zero_rsa):
    if batch.x.shape[1] != 1572 or node_vector_channels != 3:
        raise ValueError('geometry_mask_v2 requires 29 AA + 6 angles + 1 RSA + 1536 ESM, and 3 vector channels')
    if edge_mode == 'sequence':
        edges = sequence_edge_index(batch.batch, max_nn)
    elif edge_mode == 'none':
        edges = torch.empty((2, 0), dtype=torch.long, device=batch.x.device)
    else:
        raise ValueError('Geometry ablation supports sequence or none edges only')
    x = batch.x
    rsa = torch.zeros_like(x[:, 35:36]) if zero_rsa else x[:, 35:36]
    batch.x = torch.cat((x[:, :29], torch.zeros_like(x[:, 29:35]), rsa, x[:, 36:]), dim=1)
    n, e = len(x), edges.shape[1]
    batch.pos = x.new_zeros((n, 3))
    batch.sc = x.new_zeros((n, 3))
    batch.bb_coords = x.new_zeros((n, 3, 3))
    batch.edge_index = edges
    batch.x_vector_attr = x.new_zeros((n, node_vector_channels, 3))
    batch.edge_attr = x.new_zeros((e, 3))
    batch.edge_vector_attr = x.new_zeros((e, 1, 3))
    batch.f_ij = x.new_zeros((e, 3, 3))
    # Retire a stale preprocessing spatial-neighbor cache. The sole downstream
    # adjacency is the explicit sequence/empty edge_index just constructed.
    if 'ids_topk' in batch:
        del batch.ids_topk
    batch.h, batch.chi, batch.e, batch.xi = batch.x, batch.x_vector_attr, batch.edge_attr, batch.edge_vector_attr
    return batch
