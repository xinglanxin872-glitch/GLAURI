"""September 2026 contact-label clustering: Jaccard distance, average linkage."""
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


def tanimoto_distances(labels):
    if labels.dtype != np.bool_:
        raise TypeError(f'labels must be boolean, got {labels.dtype}')
    return pdist(labels, metric='jaccard')


def cluster_labels(labels, *, threshold=0.8, method='average'):
    """Cut a hierarchical tree at distance threshold; largest pocket is ID 0.

    Distance 0.8 corresponds to pairwise similarity 0.2. Average linkage cuts
    the mean inter-cluster distance, not a minimum similarity for every pair.
    A distance threshold of 0.2 instead corresponds to pairwise similarity 0.8.
    Pocket support filtering (e.g. 0.2) is a separate downstream operation.
    """
    if labels.ndim != 2 or labels.dtype != np.bool_:
        raise TypeError('labels must be a 2-D boolean matrix')
    if not 0 <= threshold <= 1:
        raise ValueError('distance threshold must lie in [0, 1]')
    if method not in ('average', 'single', 'complete'):
        raise ValueError('method must be average, single, or complete')
    if len(labels) < 2:
        return np.zeros(len(labels), dtype=np.int64), np.zeros((0, 4)), np.zeros(0)
    distance = tanimoto_distances(labels)
    tree = linkage(distance, method=method)
    raw = fcluster(tree, t=threshold, criterion='distance')
    order = [lab for lab, _ in sorted(zip(*np.unique(raw, return_counts=True)), key=lambda item: (-item[1], item[0]))]
    remap = {lab: i for i, lab in enumerate(order)}
    return np.asarray([remap[v] for v in raw], dtype=np.int64), tree, distance
