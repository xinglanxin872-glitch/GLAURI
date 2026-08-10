"""STEP 3 of 3 (colabel pocket pipeline) -- 压成训练用稀疏 pkl。
    step 1: step1_batch_cluster_pocket_labels.py
    step 2: step2_pocket_binary_labels.py
    step 3: step3_build_pocket_store.py  (this file)

把 4 个 per-pocket 二值标签 CSV 压成一个训练可直接用的稀疏 pkl。

背景
----
`pocket_binary_labels_final/{protein,nn,ion,ligand}_pocket_binary_labels.csv`
每行是"一个 cluster 的一个口袋", `pocket_label` 是与 `ref_seq` 等长的 0/1 串。
4 个 CSV 合计 ~95MB / 14.5 万行, 8 个 dataloader worker 各解析一遍不可接受,
所以离线压成稀疏下标形式(只存正例位置), 训练时直接 pickle.load。

产物格式
--------
    {
      cluster_id: {
        'ref_len': int,                  # == len(ref_seq) == colabel[cluster].shape[0]
        'pockets': {                     # 任务 -> 该任务下的口袋列表
            0: [np.ndarray(int32), ...], # protein  (contact_class 0)
            1: [...],                    # nn       (contact_class 1)
            2: [...],                    # ion      (contact_class 2)
            3: [...],                    # ligand   (contact_class 3)
        },
      },
      ...
      ('__meta__',): {...}
    }

任务编号 0/1/2/3 与 `config_data['r_types']` 的列序一致:
[protein, dna+rna, ion, ligand], 也与 colabel 的 4 列一一对应(已验证)。
某 cluster 在某任务上没有口袋 -> `pockets` 里没有该键(= 空标签, 训练时全负例)。

用法
----
    conda activate boltz
    python3 step3_build_pocket_store.py                       # 默认写 ./pocket_store.pkl
    python3 step3_build_pocket_store.py --out /path/x.pkl
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# 任务编号 -> CSV 文件名。编号即 contact_class, 与 colabel 列序一致。
TASK_FILES = {
    0: "protein_pocket_binary_labels.csv",
    1: "nn_pocket_binary_labels.csv",
    2: "ion_pocket_binary_labels.csv",
    3: "ligand_pocket_binary_labels.csv",
}
TASK_NAMES = {0: "protein", 1: "nn", 2: "ion", 3: "ligand"}


def parse_label(s):
    """'0110...' -> np.int32 正例下标数组。CSV 是 CRLF, 必须 strip。"""
    s = str(s).strip()
    a = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
    return np.flatnonzero(a == ord("1")).astype(np.int32), len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_dir", default=os.path.join(HERE, "pocket_binary_labels_final"))
    ap.add_argument("--out", default=os.path.join(HERE, "pocket_store.pkl"))
    ap.add_argument("--colabel", default=os.path.join(HERE, "all_colabel.pkl"),
                    help="用于交叉校验 ref_len; 传空字符串跳过校验")
    args = ap.parse_args()

    colabel = None
    if args.colabel and os.path.isfile(args.colabel):
        with open(args.colabel, "rb") as f:
            colabel = pickle.load(f)
        print(f"[check] loaded colabel: {len(colabel)} clusters")

    store = {}
    stats = {}

    for task, fname in TASK_FILES.items():
        path = os.path.join(args.label_dir, fname)
        df = pd.read_csv(path, usecols=["cluster_id", "pocket_index", "pocket_label", "ref_seq"])
        # pocket_index 决定口袋顺序; 排序后 groupby 保证列表顺序稳定可复现
        df = df.sort_values(["cluster_id", "pocket_index"], kind="mergesort")

        n_pockets = 0
        n_len_mismatch = 0
        n_ref_conflict = 0
        n_colabel_mismatch = 0

        for cid, grp in df.groupby("cluster_id", sort=False):
            ref_len = len(str(grp["ref_seq"].iloc[0]).strip())

            plist = []
            for lab in grp["pocket_label"]:
                idx, L = parse_label(lab)
                if L != ref_len:
                    n_len_mismatch += 1
                    continue
                plist.append(idx)
            if not plist:
                continue

            ent = store.setdefault(cid, {"ref_len": ref_len, "pockets": {}})
            # 同一 cluster 的 ref_len 必须在 4 个任务间一致
            if ent["ref_len"] != ref_len:
                n_ref_conflict += 1
                continue
            if colabel is not None and cid in colabel:
                if int(np.asarray(colabel[cid]).shape[0]) != ref_len:
                    n_colabel_mismatch += 1

            ent["pockets"][task] = plist
            n_pockets += len(plist)

        stats[task] = dict(pockets=n_pockets, len_mismatch=n_len_mismatch,
                           ref_conflict=n_ref_conflict, colabel_mismatch=n_colabel_mismatch)
        print(f"[{TASK_NAMES[task]:7s}] pockets={n_pockets:6d}  "
              f"len_mismatch={n_len_mismatch}  ref_conflict={n_ref_conflict}  "
              f"colabel_len_mismatch={n_colabel_mismatch}")

    # ---- 汇总 ----
    npock_per_task = {t: [] for t in TASK_FILES}
    for ent in store.values():
        for t, pl in ent["pockets"].items():
            npock_per_task[t].append(len(pl))

    print(f"\nclusters in store: {len(store)}")
    for t in TASK_FILES:
        v = npock_per_task[t]
        if v:
            v = np.asarray(v)
            print(f"  {TASK_NAMES[t]:7s}: clusters={len(v):6d}  "
                  f"mean_pockets={v.mean():.2f}  max={v.max()}  multi={(v > 1).sum()}")

    store[("__meta__",)] = {
        "format": "pocket_store_v1",
        "task_names": TASK_NAMES,
        "source": args.label_dir,
        "stats": stats,
        "n_clusters": len(store),
    }

    with open(args.out, "wb") as f:
        pickle.dump(store, f, protocol=4)
    print(f"\n[write] {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
