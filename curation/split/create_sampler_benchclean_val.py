"""
Generate sample weights for WeightedRandomSampler on the benchclean validation set
(same method as create_sampler_benchclean.py).

weight = 1 / (number of validation samples in that sequence cluster), for
cluster-balanced sampling (large clusters are not oversampled).
Sequence clusters come from colabel/high_id_cluster/all_cluster.txt (members are
ukeys, e.g. ZL/5ZLP/A:0).

Only difference from the train version: mask uses cluster_30_80_benchclean_val_m.pkl,
output is val_weight.pkl.
The ukey dedup order matches the validation DataLoader in main.py (same h5 keys +
same val mask, first-occurrence order); weights correspond one-to-one with validation
samples, which main.py uses for fixed weighted subset sampling.
"""
import pickle
from pathlib import Path
import numpy as np
import torch as pt

from data_handler import Dataset
from src.dataset import select_by_interface_types
from sids_sel import config_data   # correct dataset_filepath / min_num_res / l_types / r_types

PROJECT_ROOT = Path("/home/lanxin/data/Interface_prediction")
ALL_CLUSTER = PROJECT_ROOT / "colabel/high_id_cluster/all_cluster.txt"
VAL_M = PROJECT_ROOT / "processing/data/datasets/cluster_30_80_benchclean_val_m.pkl"
OUT = PROJECT_ROOT / "processing/data/datasets/cluster_30_80_benchclean_val_weight.pkl"

# 1) chainid(ukey) -> cluster center id
chainid_to_center = {}
with open(ALL_CLUSTER) as t:
    for i, line in enumerate(t):
        line = line.strip()
        if not line:
            continue
        for tok in line.split():
            chainid_to_center[tok.strip()] = i
print(f"[cluster] all_cluster.txt: clusters={i+1}  tokens={len(chainid_to_center)}")

# 2) Load benchclean validation mask, replicate the extra filtering
# (idempotent for benchclean, consistent with main.py after mask loading)
with open(VAL_M, "rb") as v:
    m = pickle.load(v)
dataset = Dataset(config_data)
m = np.asarray(m).astype(bool)
m &= (dataset.sizes[:, 1] >= config_data['min_num_res'])
m &= select_by_interface_types(dataset, config_data['l_types'])
dataset.update_mask(m)
dataset.set_types(config_data['l_types'], config_data['r_types'])
print(f"[dataset] val ukey count: {len(dataset.ukeys)}")

# 3) each ukey -> cluster id (coverage should be 100%, otherwise report missing)
missing = [u for u in dataset.ukeys if u not in chainid_to_center]
assert not missing, f"{len(missing)} ukeys not in all_cluster.txt, e.g.: {missing[:5]}"
clusterids = np.array([chainid_to_center[u] for u in dataset.ukeys])
uniq = np.unique(clusterids)
print(f"[cluster] clusters covered in val set: {len(uniq)}")

# 4) weight = 1 / number of val samples in the cluster
class_sample_count = np.array([np.sum(clusterids == t) for t in uniq])
weight = 1.0 / class_sample_count
w_dict = {c: w for c, w in zip(uniq, weight)}
samples_weight = pt.from_numpy(np.array([w_dict[c] for c in clusterids])).double()
print(f"[weight] samples_weight.shape = {tuple(samples_weight.shape)}  "
      f"min={samples_weight.min():.3g} max={samples_weight.max():.3g}")

with open(OUT, "wb") as v:
    pickle.dump(samples_weight, v)
print(f"[done] written: {OUT}")
