import os
import json
import numpy as np
import torch as pt
import sys
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import pickle
import argparse
import shutil
import importlib.util
import random
from src.logger import Logger
from src.scoring import bc_scoring, bc_score_names, nanmean
from config import config_data, config_model, config_runtime,tag
from data_handler import Dataset, collate_batch_data_key_pyg, resolve_project_path
from model import GCPNetModel
from src.utils_eval import *
#from torch_geometric.data import Data, Batch
from torch.utils.data import WeightedRandomSampler, Subset

def setup_dataloader(config_data, logger, colabel_path=None, sel_m=None,
                     train_weight=None, shuffle=True, cluster_labels_path=None,
                     label_binarize_threshold=None, subset_size=None, subset_seed=42,
                     sample_weight=None, mask_dict_path=None,
                     pocket_sample=False, pocket_store_path=None, pocket_native_prob=0.2,
                     return_key=False):
    loader_config = dict(config_data)
    loader_config["colabel_path"] = colabel_path
    # native-only GCP training still needs to return the key so that the pairwise mask / PLM
    # embedding can be aligned by structure; return_key only changes what the batch carries, not the labels.
    loader_config["return_key"] = return_key
    # pairwise mask: when provided, Dataset aligns labels at the data layer (new_format); ignored for old format / when omitted
    loader_config["mask_dict_path"] = mask_dict_path
    # single-pocket label sampling (off by default). Enabled only for the [training set]; the validation set is always union, keeping metrics comparable.
    loader_config["pocket_sample"] = pocket_sample
    loader_config["pocket_store_path"] = pocket_store_path
    loader_config["pocket_native_prob"] = pocket_native_prob
    if cluster_labels_path:
        loader_config["cluster_labels_path"] = cluster_labels_path
        loader_config["label_noise_std"] = 0.0
        loader_config["label_binarize_threshold"] = label_binarize_threshold
    dataset = Dataset(loader_config)
    # pocket sampling must not silently fall back to union on old masks / non-CA_BB preprocessing; when explicitly requested, the code path must be available.
    if pocket_sample and not getattr(dataset, "mask_new_format", False):
        raise RuntimeError(
            "--pocket_sample requires a pairwise mask and preprocess=CA_BB; "
            f"current mask_new_format={getattr(dataset, 'mask_new_format', False)}, "
            f"preprocess={getattr(dataset, 'preprocess', None)!r}"
        )
    with open(sel_m,'rb') as id:
        m=pickle.load(id)
    dataset.update_mask(m)

    # set dataset types for labels
    dataset.set_types(config_data['l_types'], config_data['r_types'])

    # fixed subset (mainly for the validation set): draw subset_size indices with a fixed seed, so every round evaluates this same batch of samples
    if subset_size is not None and subset_size < len(dataset):
        if sample_weight is not None:
            # weighted fixed subset: weighted sampling without replacement by cluster weights (1/cluster size),
            # fixed seed -> same batch every round, so the validation subset's effective distribution matches the training set's weighted sampling (WeightedRandomSampler).
            with open(sample_weight, 'rb') as wf:
                w = pickle.load(wf)
            w = pt.as_tensor(np.asarray(w)).double().flatten()
            assert len(w) == len(dataset), (
                f"valid sample_weight length {len(w)} != valid dataset {len(dataset)}; "
                f"mask or ukey order mismatch; regenerate with create_sampler_benchclean_val.py on the same val mask"
            )
            g = pt.Generator().manual_seed(subset_seed)
            perm = pt.multinomial(w, num_samples=subset_size, replacement=False, generator=g).tolist()
            dataset = Subset(dataset, sorted(perm))
            logger.print(f"> fixed weighted subset: {len(dataset)} samples (seed={subset_seed}, cluster-balanced)")
        else:
            g = pt.Generator().manual_seed(subset_seed)
            perm = pt.randperm(len(dataset), generator=g)[:subset_size].tolist()
            dataset = Subset(dataset, sorted(perm))
            logger.print(f"> fixed subset: {len(dataset)} samples (seed={subset_seed})")

    # define data loader: always go through key_pyg (GCP path; the batch carries ukey for pairwise mask/PLM alignment)
    if not (colabel_path or return_key):
        raise ValueError(
            "setup_dataloader requires colabel_path or return_key=True; "
            "the old key-less collate (collate_batch_data) was removed together with the PeSTo path"
        )
    if train_weight:
        with open(train_weight,'rb')as t:
            train_weight=pickle.load(t)
        sampler = WeightedRandomSampler(train_weight, num_samples=len(train_weight), replacement=True)
        dataloader = pt.utils.data.DataLoader(dataset, batch_size=config_runtime['batch_size'],
                                            sampler=sampler, num_workers=8,
                                            collate_fn=collate_batch_data_key_pyg,
                                            pin_memory=True, prefetch_factor=2)
    else:
        dataloader = pt.utils.data.DataLoader(dataset, batch_size=config_runtime['batch_size'], shuffle=shuffle, num_workers=8, collate_fn=collate_batch_data_key_pyg, pin_memory=True, prefetch_factor=2)

    return dataloader

def _valid_train_step(y, z):
    """False -> skip this step (no forward loss, no pos_ratios update)."""
    if y.numel() == 0 or z.numel() == 0:
        return False
    if y.shape[0] != z.shape[0]:
        return False
    if y.ndim != 2 or z.ndim != 2:
        return False
    if not pt.isfinite(y).all() or not pt.isfinite(z).all():
        return False
    if not pt.isfinite(pt.mean(y, dim=0)).all():
        return False
    return True


def _normalize_active_tasks(active_tasks, n_tasks):
    """Validate task indices while preserving CLI order."""
    if active_tasks is None:
        return tuple(range(n_tasks))
    tasks = tuple(int(i) for i in active_tasks)
    if not tasks:
        raise ValueError("active_tasks must not be empty")
    if len(set(tasks)) != len(tasks):
        raise ValueError(f"active_tasks contains duplicate indices: {tasks}")
    invalid = [i for i in tasks if i < 0 or i >= n_tasks]
    if invalid:
        raise ValueError(f"active_tasks out of range: {invalid}; valid range 0..{n_tasks - 1}")
    return tasks


def _loss_factors(pos_ratios, active_tasks=None):
    """Normalize column weights over active tasks only; inactive columns stay zero."""
    tasks = _normalize_active_tasks(active_tasks, pos_ratios.shape[0])
    factors = pt.zeros_like(pos_ratios)
    values = pos_ratios[list(tasks)]
    if pt.isfinite(values).all() and pt.sum(values) > 0:
        factors[list(tasks)] = values / pt.sum(values)
    else:
        factors[list(tasks)] = 1.0 / len(tasks)
    return factors.reshape(1, -1)


def eval_step(model, device, batch_data, criterion, pos_ratios, pos_weight_factor, global_step,ref_db=None,mapping=None,mask=None,plm_type=None,plm_pack=None,plm_fusion="concat",h5_path=None,h5_file=None,is_train=True,active_tasks=None):
    # With single-pocket sampling (--pocket_sample), Dataset attaches two node attributes to the pyg batch:
    #   batch.pm = loss mask (1=included, 0=ignore unselected real pocket residues)
    #   batch.yu = union labels (native ∪ colabel), fed to metrics only, not to the loss
    # when disabled, neither exists -> the whole flow stays line-by-line identical to before.
    pocket_mask = None
    y_metric = None

    # unpack data
    # only the new pairwise mask (dict) is supported: labels are already aligned to the N_ca structure nodes
    # in the dataset, no more graph subgraphing; just scatter the fasta-computed PLM embedding onto the N_ca canvas (zero-padded where uncovered).
    if mapping is None:
        raise ValueError("eval_step requires mapping (chain_mapping); the old mapping-less path has been removed")
    data_list= [data for data in batch_data]
    ukey=data_list[-1]
    md = mask[ukey[0]]

    chainname=ukey[0].split("/")[1].upper()+"_"+ukey[0].split("/")[2].split(":")[0]
    fasta_seq=ref_db[chainname]

    if not isinstance(md, dict):
        raise TypeError(
            f"mask[{ukey[0]!r}] is not in pairwise dict format; the old three-way MSA mask is no longer supported, "
            "rebuild the mask with build_mask_dict_pairwise.py"
        )
    y = data_list[-2].to(device)          # already N_ca, one-to-one with the graph nodes
    batch = data_list[0].to(device)
    n_ca = batch.num_nodes
    if n_ca == 0:
        return "failed", None, None
    if plm_type:
        plm_embedding = get_embedding(
            fasta_seq, device, plm_type, plm_pack, ukey=ukey[0],
            h5_path=None if h5_file is not None else h5_path, h5_file=h5_file,
        )
        m_fasta = pt.as_tensor(np.asarray(md['fasta'][0])).bool()
        m_struct_f = pt.as_tensor(np.asarray(md['fasta'][1])).bool()
        if m_struct_f.numel() != n_ca or int(m_fasta.sum()) != int(m_struct_f.sum()):
            return "failed", None, None
        emb_canvas = pt.zeros((n_ca, plm_embedding.shape[1]),
                              device=device, dtype=plm_embedding.dtype)
        if int(m_fasta.sum()) > 0:
            emb_canvas[m_struct_f.to(device)] = plm_embedding[m_fasta].to(device)
        if plm_fusion == "series":
            batch.x = emb_canvas
        else:
            batch.x = pt.cat([batch.x, emb_canvas], dim=1)
    z = model.forward(batch)["node_embedding"]
    M = batch.M
    assert batch.pos.shape[0] == batch.x.shape[0] == M.shape[0]
    # y and z are both N_ca and aligned one-to-one, no further masking needed
    pocket_mask = getattr(batch, 'pm', None)   # None when single-pocket sampling is off
    y_metric = getattr(batch, 'yu', None)

    if not _valid_train_step(y, z):
        return "failed", None, None

    if pocket_mask is not None and pocket_mask.shape != y.shape:
        print(f"[eval_step] WARN pocket_mask {tuple(pocket_mask.shape)} != y {tuple(y.shape)}; ignoring mask for this step")
        pocket_mask = None
    # metrics always use union labels (comparable with baseline / historical runs); when sampling is off, yu does not exist -> fall back to y
    if y_metric is None or y_metric.shape != y.shape:
        y_metric = y

    active_tasks = _normalize_active_tasks(active_tasks, y.shape[1])
    active_mask = pt.zeros(y.shape[1], dtype=pt.bool, device=y.device)
    active_mask[list(active_tasks)] = True

    if is_train:
        # training: update pos_ratios (EMA), set pos_weight, and weight across columns by loss_factors
        if pocket_mask is not None:
            # the denominator counts only residues [included in the loss]. Otherwise ignored positives
            # stay in the denominator -> pos_ratio is underestimated and pos_weight is pushed too high.
            denom = pocket_mask.sum(dim=0).clamp(min=1.0)                   # (4,)
            batch_mean = ((y * pocket_mask).sum(dim=0) / denom).detach()
        else:
            denom = None
            batch_mean = pt.mean(y, dim=0).detach()
        valid_cols = pt.isfinite(batch_mean) & active_mask
        pos_ratios[valid_cols] += (batch_mean[valid_cols] - pos_ratios[valid_cols]) / (
            1.0 + np.sqrt(global_step)
        )
        # BCE still produces four output columns, but inactive tasks contribute neither weight nor final loss to training.
        pos_weight = pt.ones_like(pos_ratios)
        pos_weight[active_mask] = (
            pos_weight_factor * (1.0 - pos_ratios[active_mask])
            / (pos_ratios[active_mask] + 1e-6)
        )
        criterion.pos_weight = pos_weight

        dloss = criterion(z, y)

        loss_factors = _loss_factors(pos_ratios, active_tasks)
        if pocket_mask is not None:
            # normalize [per column], not by a uniform division by N: a column like ion, with many pockets
            # and many ignores, would be systematically shrunk if still divided by N. clamp(min=1) avoids division by zero when a column is fully ignored.
            losses = loss_factors * (dloss * pocket_mask) / denom.reshape(1, -1)
        else:
            losses = (loss_factors * dloss) / dloss.shape[0]
    else:
        # validation: do not update pos_ratios/pos_weight; use plain (unweighted) BCE so valid loss stays comparable across steps.
        # the validation set does [not] do single-pocket sampling (y is the union label), so metrics are directly comparable with historical runs.
        criterion.pos_weight = None
        dloss = criterion(z, y)
        losses = dloss / dloss.shape[0]

    # key control: inactive outputs may be forwarded, but contribute no gradient or aggregated loss.
    losses = losses * active_mask.reshape(1, -1).to(losses.dtype)

    return losses, y_metric.detach(), pt.sigmoid(z).detach()



def scoring(eval_results, device=pt.device('cpu'), active_tasks=None):
    # compute sum losses and scores for each entry
    sum_losses, scores = [], []
    for losses, y, p in eval_results:
        
        if pt.isnan(y).any() or pt.isnan(p).any():
            print("NaN detected in evaluation batch. Skipping...")
            print(y,p)
            continue
        tasks = _normalize_active_tasks(active_tasks, y.shape[1])
        y_active = y[:, list(tasks)]
        p_active = p[:, list(tasks)]
        y_np = y_active.cpu().numpy()
        p_np = p_active.cpu().numpy()

        valid = True
        for m in range(y_np.shape[1]):
            if len(np.unique(p_np[:, m])) < 2:
                valid = False
                break

        if not valid:
            print("Only one class present. Skipping batch...")
            continue
        sum_losses.append(pt.sum(losses[:, list(tasks)], dim=0))
        scores.append(bc_scoring(y_active, p_active))


    # average scores
    m_losses = pt.mean(pt.stack(sum_losses, dim=0), dim=0).numpy()
    m_scores = nanmean(pt.stack(scores, dim=0)).numpy()

    # pack scores
    scores = {'loss': float(np.sum(m_losses))}
    for local_i, task_i in enumerate(tasks):
        scores[f'{task_i}/loss'] = m_losses[local_i]
        for j in range(m_scores.shape[0]):
            scores[f'{task_i}/{bc_score_names[j]}'] = m_scores[j,local_i]

    return scores


def logging(logger, writer, scores, global_step, pos_ratios, step_type, active_tasks=None):
    # debug print
    tasks = _normalize_active_tasks(active_tasks, pos_ratios.shape[0])
    pr_str = ', '.join([f"{i}:{pos_ratios[i]:.4f}" for i in tasks])
    logger.print(f"{step_type}> [{global_step}] loss={scores['loss']:.4f}, active_pos_ratios=[{pr_str}]")

    # store statistics
    summary_stats = {k:scores[k] for k in scores if not np.isnan(scores[k])}
    summary_stats['global_step'] = int(global_step)
    summary_stats['pos_ratios'] = list(pos_ratios.cpu().numpy())
    summary_stats['step_type'] = step_type
    logger.store(**summary_stats)

    # detailed information
    for key in scores:
        writer.add_scalar(step_type+'/'+key, scores[key], global_step)

    # debug print
    for c in np.unique([key.split('/')[0] for key in scores if len(key.split('/')) == 2]):
        logger.print(f'[{c}] loss={scores[c+"/loss"]:.3f}, ' + ', '.join([f'{sn}={scores[c+"/"+sn]:.3f}' for sn in bc_score_names]))

def train(config_data, config_model, config_runtime,output_path,ref_db,
           model_name="model.pt",
          train_m=None,valid_m=None,
          colabel_path_train=None,
          colabel_path_valid=None,
          h5_train=None,
          h5_valid=None,
          train_weight=None,
          valid_weight=None,
          cluster_labels_path=None,
          label_binarize_threshold=0.5,
          valid_eval_size=2048,
          valid_subset_seed=42,
          pocket_sample=False,
          pocket_store_path=None,
          pocket_native_prob=0.2,
          train_native_only=False,
          active_tasks=None):
    # create logger

    logger = Logger(output_path, f'train')

    # print configuration
    logger.print(">>> Configuration")
    logger.print(config_data)
    logger.print(config_runtime)

    n_types = len(config_data['r_types'])
    active_tasks = _normalize_active_tasks(active_tasks, n_types)
    task_names = ("protein", "nucleic_acid", "ion", "ligand")
    active_names = [task_names[i] if i < len(task_names) else str(i) for i in active_tasks]
    logger.print(f"> active tasks: {list(active_tasks)} ({', '.join(active_names)})")

    # optional full training seed. valid_subset_seed only fixes the validation subset; it does not fix
    # model initialization or the WeightedRandomSampler. Enabled only when an A/B config explicitly provides a seed; old configs behave as before.
    run_seed = config_runtime.get("seed", None)
    if run_seed is not None:
        run_seed = int(run_seed)
        random.seed(run_seed)
        np.random.seed(run_seed)
        pt.manual_seed(run_seed)
        if pt.cuda.is_available():
            pt.cuda.manual_seed_all(run_seed)
        logger.print(f"> training seed: {run_seed}")

    # define device
    device = pt.device(config_runtime['device'])

    # create model
    model = GCPNetModel(config_model)

    # debug print
    logger.print(">>> Model")
    logger.print(model)
    logger.print(f"> {sum([int(pt.prod(pt.tensor(p.shape))) for p in model.parameters()])} parameters")
    
    plm_fusion=config_data.get('plm_fusion',"concat")
    print("plm_fusion",plm_fusion)
    #plm_embedonly=config_data.get('plm_embedonly',False)
    # reload model if configured
    model_filepath = os.path.join(output_path,model_name)
    if os.path.isfile(model_filepath) and config_runtime["reload"]:
        logger.print("Reloading model from save file")
        model.load_state_dict(pt.load(model_filepath))
        # get last global step
        global_step = json.loads([l for l in open(logger.log_lst_filepath, 'r')][-1])['global_step']
        # dynamic positive weight
        pos_ratios = pt.from_numpy(np.array(json.loads([l for l in open(logger.log_lst_filepath, 'r')][-1])['pos_ratios'])).float().to(device)
        print(f"Previous global step is {global_step}")
        print(f"Previous pos_ratios is {pos_ratios}")
        reloaded = True
    else:
        # starting global step
        global_step = 0
        # dynamic positive weight
        pos_ratios = 0.5*pt.ones(len(config_data['r_types']), dtype=pt.float).to(device)
        reloaded = False

    # debug print
    logger.print(">>> Loading data")
    logger.print(
        "> sampling switches: "
        f"train_weight={'ON' if train_weight else 'OFF'}, "
        f"valid_weight={'ON' if valid_weight else 'OFF'}, "
        f"train_pocket={'ON' if pocket_sample else 'OFF'}, "
        "valid_pocket=OFF"
    )
    logger.print(
        "> label sources: "
        f"train={'native contacts only' if train_native_only else 'configured projected labels + native'}, "
        "valid=configured projected labels + native"
    )

    # setup dataloaders
    # the mask paths are computed up front and passed to Dataset (with the new pairwise format, labels are
    # aligned directly at the data layer); the same pkl is loaded later for eval_step's embedding scattering.
    mask_path_train = os.path.join(out_dir, "mask_dict_train.pkl")
    mask_path_valid = os.path.join(out_dir, "mask_dict_valid.pkl")

    dataloader_train = setup_dataloader(
        config_data, logger,
        colabel_path=colabel_path_train, sel_m=train_m, train_weight=train_weight,
        cluster_labels_path=None if train_native_only else cluster_labels_path,
        label_binarize_threshold=label_binarize_threshold,
        mask_dict_path=mask_path_train,
        return_key=train_native_only,
        # single-pocket sampling is enabled [only] on the training set
        pocket_sample=pocket_sample,
        pocket_store_path=pocket_store_path,
        pocket_native_prob=pocket_native_prob,
    )

    dataloader_valid = setup_dataloader(
        config_data, logger,
        colabel_path=colabel_path_valid, sel_m=valid_m,
        cluster_labels_path=cluster_labels_path,
        label_binarize_threshold=label_binarize_threshold,
        shuffle=False,                  # fixed order
        subset_size=valid_eval_size,    # fixed subset (default 2048); the same batch is fully evaluated every round: comparable and time-saving
        subset_seed=valid_subset_seed,
        sample_weight=valid_weight,     # if provided, build a fixed "weighted" subset with the same distribution as the training weighted sampling
        mask_dict_path=mask_path_valid,
        # the validation set does [not] do single-pocket sampling: labels are always union, so valid loss / AUC
        # are directly comparable with historical runs, and ckpt selection (by valid AUC) is not polluted by sampling noise.
    )

    # write the final Dataset state into train.log, and runtime-check the CLI -> Dataset wiring.
    train_pocket_active = bool(getattr(dataloader_train.dataset, "pocket_sample", False))
    valid_dataset = dataloader_valid.dataset
    while isinstance(valid_dataset, Subset):
        valid_dataset = valid_dataset.dataset
    valid_pocket_active = bool(getattr(valid_dataset, "pocket_sample", False))
    if train_pocket_active != bool(pocket_sample) or valid_pocket_active:
        raise RuntimeError(
            "pocket switch wiring error: "
            f"requested_train={bool(pocket_sample)}, actual_train={train_pocket_active}, "
            f"actual_valid={valid_pocket_active}"
        )
    logger.print(
        f"> dataset pocket state: train={'ON' if train_pocket_active else 'OFF'}, "
        f"valid={'ON' if valid_pocket_active else 'OFF'}"
    )

    mapping_train, mapping_valid = None, None
    # mapping is only for structure/sequence/embedding alignment, not a supervision label. native-only training still needs it.
    default_mapping_path = config_data.get('colabel_path', {}).get('chain_mapping')
    cm_train = (
        colabel_path_train.get('chain_mapping')
        if colabel_path_train else default_mapping_path
    )
    cm_valid = (
        colabel_path_valid.get('chain_mapping')
        if colabel_path_valid else default_mapping_path
    )
    if cm_train:
        with open(cm_train, 'rb') as t1:
            mapping_train = pickle.load(t1)
    if cm_valid:
        with open(cm_valid, 'rb') as t2:
            mapping_valid = pickle.load(t2)
    # debug print
    logger.print(f"> training data size: {len(dataloader_train)}")
    logger.print(f"> valid data size: {len(dataloader_valid)}")

    # debug print
    logger.print(">>> Starting training")

    # send model to device
    model = model.to(device)

    criterion = pt.nn.BCEWithLogitsLoss(reduction="none")
    if cluster_labels_path:
        logger.print(
            f"> classification labels from {cluster_labels_path}, "
            f"binarize threshold={label_binarize_threshold}"
        )

    # define optimizer
    optimizer = pt.optim.Adam(model.parameters(), lr=config_runtime["learning_rate"])

    # restart timer
    logger.restart_timer()

    # summary writer
    writer = SummaryWriter(os.path.join(output_path, 'tb'))

    # min loss initial value
    # min_loss = 1e9  # [redundant] model saving is now decided by valid AUC, no longer best-by-loss
    # max valid auc initial value (best model selected by per-column mean AUC on the validation set)
    max_auc = -1.0
    # per-interface historical best AUC (column mean -> split per column): when interface i beats its own historical best -> save model_i.pt
    max_auc_per = [-1.0] * n_types
    # on resume, restore each best AUC from historical test logs; otherwise the first eval would overwrite better old checkpoints with the -1 baseline
    if reloaded:
        for _line in open(logger.log_lst_filepath, 'r'):
            try:
                _d = json.loads(_line)
            except (ValueError, json.JSONDecodeError):
                continue
            if _d.get('step_type') != 'test':
                continue
            _per = {i: _d.get(f'{i}/auc') for i in active_tasks}
            for i, _v in _per.items():
                if _v is not None and np.isfinite(_v) and _v > max_auc_per[i]:
                    max_auc_per[i] = _v
            if all(_v is not None and np.isfinite(_v) for _v in _per.values()):
                _m = float(np.mean(list(_per.values())))
                if _m > max_auc:
                    max_auc = _m
        logger.print(f"> restored best active-task AUC from log: mean={max_auc:.4f}, "
                     f"per-type={['%.4f'%a for a in max_auc_per]}")

    mask_train_name="mask_dict_train.pkl"
    mask_valid_name="mask_dict_valid.pkl"
    mask_path_train=os.path.join(out_dir,mask_train_name)
    mask_path_valid=os.path.join(out_dir,mask_valid_name)
    # only pre-built pairwise masks are supported (on-the-fly rebuilding of the old three-way MSA mask was removed together with the old-format path).
    if not (os.path.exists(mask_path_train) and os.path.exists(mask_path_valid)):
        raise FileNotFoundError(
            f"missing pairwise mask: {mask_path_train} / {mask_path_valid}; "
            "run build_mask_dict_pairwise.py first to generate them, "
            "or symlink the pkl files under new_mask_pairwise/ into out_dir as in ablation/run_*.sh"
        )
    with open(mask_path_train,"rb")as t:
        mask_train=pickle.load(t)
    with open(mask_path_valid,"rb")as t:
        mask_valid=pickle.load(t)
    print("mask loaded from existing paths")


    plm_type=config_data.get("plm",None)

    plm_pack=None   # must be defined even when plm is off, otherwise eval_step(plm_pack=plm_pack) below raises NameError
    if plm_type is not None:
        plm_pack=get_plm_model(device,plm_type)

    # precomputed embeddings: open the HDF5 only once for the whole run, avoiding repeated opens of a huge file per batch
    h5_train_f = open_plm_embedding_h5(h5_train) if h5_train else None
    if h5_valid and h5_train and os.path.abspath(h5_valid) == os.path.abspath(h5_train):
        h5_valid_f = h5_train_f
    else:
        h5_valid_f = open_plm_embedding_h5(h5_valid) if h5_valid else None
    if h5_train_f is not None or h5_valid_f is not None:
        logger.print(
            f"> PLM embedding HDF5: train={'opened' if h5_train_f else '—'}, "
            f"valid={'same as train' if h5_valid_f is h5_train_f and h5_valid_f else ('opened' if h5_valid_f else '—')}"
        )

    try:
        for epoch in range(config_runtime['num_epochs']):
            # train mode
            model = model.train()

            # train model
            train_results = []
            for batch_train_data in tqdm(dataloader_train):
                # batch=batch_train_data[0]
                
                # print(batch.pos.shape,batch.x.shape,batch.ids_topk.shape,batch.batch.shape,batch.M.shape)
                # break
            
                # global step
                global_step += 1

                # set gradient to zero
                optimizer.zero_grad()
                # forward propagation
                losses, y, p = eval_step(model, device, batch_train_data, criterion, pos_ratios, config_runtime['pos_weight_factor'], global_step,ref_db=ref_db,mapping=mapping_train,mask=mask_train,plm_type=plm_type,plm_fusion=plm_fusion,plm_pack=plm_pack,h5_path=h5_train,h5_file=h5_train_f,active_tasks=active_tasks)
                if losses=='failed':
                    continue
                # backward propagation
                loss = pt.sum(losses)
                loss.backward()
    
                total_norm = 0.0
                for params in model.parameters():
                    if params.grad is not None:
                        param_norm = params.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                writer.add_scalar("train/grad_norm", total_norm, global_step)
    
    
                pt.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # optimization step
                optimizer.step()
    
                # store evaluation results
                train_results.append([losses.detach().cpu(), y.cpu(), p.cpu()])
    
                # log step
                if (global_step+1) % config_runtime["log_step"] == 0:
                    # process evaluation results
                    with pt.no_grad():
                        # scores evaluation results and reset buffer
                        scores = scoring(train_results, device=device, active_tasks=active_tasks)
                        train_results = []
    
                        # logging
                        logging(logger, writer, scores, global_step, pos_ratios, "train", active_tasks=active_tasks)
    
                        # save model checkpoint
                        model_filepath = os.path.join(output_path, f'model_ckpt_{tag}.pt')
                        pt.save(model.state_dict(), model_filepath)
    
                # evaluation step
                if (global_step+1) % config_runtime["eval_step"] == 0:
                    # evaluation mode
                    model = model.eval()
    
                    with pt.no_grad():
                        # evaluate model
                        test_results = []
                        for step_te, batch_test_data in enumerate(dataloader_valid):
                            # forward propagation (is_train=False: no stats update, plain BCE)

                            losses, y, p = eval_step(model, device, batch_test_data, criterion, pos_ratios, config_runtime['pos_weight_factor'], global_step,ref_db=ref_db,mapping=mapping_valid,mask=mask_valid,plm_type=plm_type,plm_pack=plm_pack,plm_fusion=plm_fusion,h5_path=h5_valid,h5_file=h5_valid_f,is_train=False,active_tasks=active_tasks)
                            if losses=='failed':
                                print("failed")
                                continue
                            # store evaluation results
                            test_results.append([losses.detach().cpu(), y.cpu(), p.cpu()])

                        # scores evaluation results
                        scores = scoring(test_results, device=device, active_tasks=active_tasks)

                        # logging
                        logging(logger, writer, scores, global_step, pos_ratios, "test", active_tasks=active_tasks)

                        # [redundant] old best-by-loss saving, now replaced by best-by-AUC selection
                        # if min_loss >= scores['loss']:
                        #     min_loss = scores['loss']
                        #     model_filepath = os.path.join(output_path, 'model.pt')
                        #     logger.print("> saving model at {}".format(model_filepath))
                        #     pt.save(model.state_dict(), model_filepath)

                        # joint: save by mean AUC over the four active tasks; single-task: save by the target column's AUC.
                        # inactive random outputs take no part in checkpoint selection at all.
                        type_aucs = [scores.get(f'{i}/auc') for i in active_tasks]
                        if all(a is not None and np.isfinite(a) for a in type_aucs):
                            valid_auc = float(np.mean(type_aucs))
                            writer.add_scalar('test/mean_auc', valid_auc, global_step)
                            if valid_auc >= max_auc:
                                max_auc = valid_auc
                                model_filepath = os.path.join(output_path, 'model.pt')
                                logger.print("> saving best mean-auc model at {} (auc={:.4f})".format(model_filepath, valid_auc))
                                pt.save(model.state_dict(), model_filepath)
                        else:
                            logger.print("> [{}] mean active-task valid AUC unavailable; model.pt not saved".format(global_step))

                        # save model_i.pt by [each interface's] own historical best
                        # interface i's AUC = scores[f'{i}/auc']; overwrite only if finite and >= that interface's historical best
                        for i in active_tasks:
                            key = f'{i}/auc'
                            if key not in scores or not np.isfinite(scores[key]):
                                logger.print("> [{}] interface {} AUC unavailable; model_{}.pt not saved".format(global_step, i, i))
                                continue
                            auc_i = float(scores[key])
                            writer.add_scalar('test/auc_{}'.format(i), auc_i, global_step)
                            if auc_i >= max_auc_per[i]:
                                max_auc_per[i] = auc_i
                                model_i_filepath = os.path.join(output_path, 'model_{}.pt'.format(i))
                                logger.print("> saving best-auc model for interface {} at {} (auc={:.4f})".format(i, model_i_filepath, auc_i))
                                pt.save(model.state_dict(), model_i_filepath)

                    # back in train mode
                    model = model.train()

    finally:
        close_plm_embedding_handles(h5_train_f, h5_valid_f)


if __name__ == '__main__':
    # train model
    time_str = time.strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--model_name', type=str, default="model.pt")
    parser.add_argument('--config', type=str, default="config.py")
    parser.add_argument('--label_threshold', type=float, default=0.5,
                        help='Binarize cluster labels: >= threshold -> 1, else 0')
    parser.add_argument('--cluster_labels_path', type=str,
                        default='../colabel/cluster_labels_summary_merged.pkl',
                        help='Merged cluster labels PKL (cluster_name, label)')
    parser.add_argument('--train_m', type=str,
                        default='../processing/refine_dataset/refine_90_train_m.pkl')
    parser.add_argument('--valid_m', type=str,
                        default='../processing/refine_dataset/refine_90_test_m.pkl')
    parser.add_argument('--train_weight', type=str, default=None,
                        help='WeightedRandomSampler weights PKL; omit for uniform shuffle sampling')
    parser.add_argument('--valid_weight', type=str, default=None,
                        help='Weights PKL for the weighted fixed validation subset (generated by create_sampler_benchclean_val.py); '
                             'if provided, draw a cluster-balanced validation subset matching the training weighted sampling distribution; omit for a uniform random subset')
    parser.add_argument('--valid_eval_size', type=int, default=2048,
                        help='Fixed valid subset size (deterministic, evaluated in full each round)')
    parser.add_argument('--valid_subset_seed', type=int, default=42,
                        help='Seed for the fixed valid subset selection')
    parser.add_argument('--active_tasks', type=str, default='0,1,2,3',
                        help='Output dimensions participating in training loss/pos_ratio/checkpoint, comma-separated; '
                             '0=protein, 1=DNA/RNA, 2=ion, 3=ligand. Default: jointly train all four dimensions')
    parser.add_argument('--use_colabel', action='store_true',
                        help='Use the data_handler colabel branch (raw projected labels colabel[cluster], no binarization/noise); '
                             'when off (default), use the cluster_label_store branch (--cluster_labels_path + --label_threshold)')
    parser.add_argument('--train_native_only', action='store_true',
                        help='Disable all colabel/cluster projected labels for the training set only, using only native contacts; '
                             'the validation set still uses the original label branch selected by --use_colabel')
    parser.add_argument('--pocket_sample', action='store_true',
                        help='Single-pocket label sampling + ignore unsampled positives. When enabled, each [training set] draw lets '
                             'the 4 task heads independently pick 1 candidate at random from the colabel pockets of that cluster + native '
                             'as the positive for this round; other real pockets are ignored (excluded from loss), and residues in no pocket '
                             'stay negative. Validation is unaffected (always union). Requires --pocket_store and pairwise mask + CA_BB. Off by default -> identical to baseline')
    parser.add_argument('--pocket_store', type=str, default='../colabel/pocket_store.pkl',
                        help='Path to pocket_store.pkl (generated by colabel/step3_build_pocket_store.py). Read only when --pocket_sample is on')
    parser.add_argument('--pocket_native_prob', type=float, default=0.2,
                        help="Probability that native (the structure's own contacts) is picked as a candidate alongside the colabel pockets; the remaining probability is distributed uniformly across the pockets")
    args = parser.parse_args()
    args.cluster_labels_path = resolve_project_path(args.cluster_labels_path)
    args.train_m = resolve_project_path(args.train_m)
    args.valid_m = resolve_project_path(args.valid_m)
    if args.train_weight is not None:
        args.train_weight = resolve_project_path(args.train_weight)
    if args.valid_weight is not None:
        args.valid_weight = resolve_project_path(args.valid_weight)

    config_path = os.path.abspath(args.config)
    module_name = "custom_config"
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = config_module
    spec.loader.exec_module(config_module)
    
    # assumes your config.py defines these three variables
    config_data = config_module.config_data
    config_model = config_module.config_model
    config_runtime = config_module.config_runtime
    try:
        args.active_tasks = _normalize_active_tasks(
            [part.strip() for part in args.active_tasks.split(',') if part.strip()],
            len(config_data['r_types']),
        )
    except ValueError as exc:
        parser.error(str(exc))


    os.makedirs(args.out_dir, exist_ok=True)
    out_dir=args.out_dir
    model_name=args.model_name
    # archive the [actually used] config (args.config), not the fixed ./config.py —
    # otherwise every --config variant would store the base config, breaking reproducibility.
    shutil.copyfile(config_path, os.path.join(out_dir, time_str+"_"+os.path.basename(config_path)))
    # variant configs usually do `from config import ...` and override; store the base too so the full parameters can be restored
    if os.path.basename(config_path) != "config.py" and os.path.isfile("./config.py"):
        shutil.copyfile("./config.py", os.path.join(out_dir, time_str+"_base_config.py"))
    with open('./all_seq_dict.pkl','rb')as d:
        ref_db=pickle.load(d)

    # branch switch:
    #   --use_colabel on -> colabel branch (raw projected labels colabel[cluster], no binarization/noise):
    #       cluster_labels_path=None makes cluster_label_store None;
    #       colabel_path carries both 'colabel' and 'chain_mapping' keys -> __init__ takes the elif self.colabel_path branch and loads self.colabel
    #   default (off) -> cluster_label_store branch: uses --cluster_labels_path + --label_threshold
    if args.use_colabel:
        colabel_path_train = None if args.train_native_only else config_data['colabel_path']
        colabel_path_valid = config_data['colabel_path']
        cluster_labels_path = None
        if args.train_native_only:
            print("[branch] train=native-only; valid=colabel (raw projected labels)")
        else:
            print("[branch] using colabel branch (raw projected labels)")
    else:
        colabel_path_train = (
            None if args.train_native_only
            else {'chain_mapping': '../colabel/colabel_mask_all.pkl'}
        )
        colabel_path_valid = {'chain_mapping': '../colabel/colabel_mask_all.pkl'}
        cluster_labels_path = args.cluster_labels_path
        if args.train_native_only:
            print("[branch] train=native-only; valid=cluster_label_store")
        else:
            print("[branch] using cluster_label_store branch (binarize/noise via cluster labels)")

    if args.train_native_only and args.pocket_sample:
        parser.error("--train_native_only and --pocket_sample cannot be enabled together: pocket sampling depends on colabel pockets")

    train(config_data, config_model, config_runtime, out_dir, ref_db, model_name=model_name,
          train_m=args.train_m,
          valid_m=args.valid_m,
          colabel_path_train=colabel_path_train,
          colabel_path_valid=colabel_path_valid,
          cluster_labels_path=cluster_labels_path,
          label_binarize_threshold=args.label_threshold,
          train_weight=args.train_weight,
          valid_weight=args.valid_weight,
          valid_eval_size=args.valid_eval_size,
          valid_subset_seed=args.valid_subset_seed,
          pocket_sample=args.pocket_sample,
          pocket_store_path=args.pocket_store,
          pocket_native_prob=args.pocket_native_prob,
          train_native_only=args.train_native_only,
          active_tasks=args.active_tasks,
          h5_train=None,
          h5_valid=None)

    
