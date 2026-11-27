"""Resolve one paper-model configuration. Training arguments cannot override it."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / 'model'

def read_json(path):
    return json.loads(Path(path).read_text())

def merge_checked(base, changes, prefix=''):
    result = deepcopy(base)
    for key, value in changes.items():
        if key not in base:
            raise ValueError(f'Unknown experiment parameter: {prefix}{key}')
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f'{prefix}{key} must be an object')
            result[key] = merge_checked(base[key], value, prefix+key+'.')
        else:
            if type(value) is not type(base[key]):
                # JSON integer values are acceptable for real-valued hyperparameters.
                if not (isinstance(base[key], float) and type(value) in (float, int)):
                    raise ValueError(f'Wrong type for {prefix}{key}')
            result[key] = deepcopy(value)
    return result

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def combine_common(common,variable):
    result=deepcopy(common)
    for key,value in variable.items():
        if isinstance(value,dict) and isinstance(result.get(key),dict):
            result[key]=combine_common(result[key],value)
        else:result[key]=deepcopy(value)
    return result

def resolve(config='GLAURI', paths_file=None):
    path = Path(config)
    if not path.is_file():
        registry = read_json(MODEL/'model_registry.json') + read_json(MODEL/'optional_models.json')
        matches = [r for r in registry if config in [r['id'], r['paper_name'], *r['aliases']]]
        if len(matches)>1:raise ValueError(f'Ambiguous model name: {config}')
        item = matches[0] if matches else None
        if item is None:
            raise ValueError(f'Unknown paper model: {config}')
        path = REPO/item['config']
    descriptor = read_json(path)
    if set(descriptor) != {'id', 'paper_name', 'changes'}:
        raise ValueError('Variant config must contain id, paper_name and changes only')
    fixed = read_json(MODEL/'config_fixed.json')
    if fixed['features']['flags']!=[False,True,False] or fixed['features']['preprocess']!='CA_BB' or fixed['features']['protocol']!='ca_ncac_sidechain_v1' or fixed['features']['plm']!='esm3':
        raise ValueError('This paper implementation requires CA_BB, residue scalars, ESM3 and the corrected CA protocol')
    if fixed['metrics']['checkpoint_selection']!='mean_task_auc' or fixed['metrics']['tie_policy']!='latest':
        raise ValueError('Unsupported checkpoint-selection rule; fixed paper rule is mean AUC/latest tie')
    if fixed['metrics']['prediction_threshold']!=0.5 or fixed['metrics']['label_threshold']!=0.5 or fixed['metrics']['auc_eligibility']!='all_active_tasks':
        raise ValueError('Unsupported fixed paper metric definition')
    if fixed['label_invariants']!={'non_ligand_columns':'binary_union','native_combination':'maximum','validation_profile':'match_training'}:
        raise ValueError('Unsupported label invariants')
    defaults=combine_common(fixed['common'],read_json(MODEL/'config_defaults.json'))
    exp = merge_checked(defaults, descriptor['changes'])
    policy = {k: exp['inputs'][k] for k in ('geometry_mode', 'edge_mode', 'ablation_protocol')}
    if policy['ablation_protocol'] != 'geometry_mask_v2':
        raise ValueError('This training package requires geometry_mask_v2')
    if policy['geometry_mode'] == 'zero':
        if policy['edge_mode'] not in ('sequence', 'none') or exp['training']['coordinate_noise'] != 0 or exp['inputs']['zero_structure']:
            raise ValueError('Zero geometry requires sequence/none edges, noise=0 and legacy zero_structure=false')
    elif policy['geometry_mode'] != 'full' or policy['edge_mode'] != 'spatial':
        raise ValueError('Full geometry requires spatial edges')
    if fixed['ligand_labels'] != {'ligand_mode':'soft','support_threshold':0.5,'soft_floor':0.25}:
        raise ValueError('The released model has a fixed ligand label definition')
    if exp['inputs']['zero_structure']:
        raise ValueError('Legacy zero_structure is not a new-training protocol; use geometry_mode=zero')
    if exp['loss']['task_weighting'] not in ('pos_ratio','uniform'):
        raise ValueError('task_weighting must be pos_ratio or uniform')
    if len(exp['loss']['positive_factors']) != 4 or any(f <= 0 for f in exp['loss']['positive_factors']):
        raise ValueError('Four positive class factors are required')
    if not exp['active_tasks'] or len(set(exp['active_tasks'])) != len(exp['active_tasks']) or not set(exp['active_tasks']) <= set(range(4)):
        raise ValueError('active_tasks must be unique task indices in 0..3')
    if exp['training']['batch_size'] != 1:
        raise ValueError('Paper graph/frame implementation supports batch_size=1')
    for key in ['epochs','evaluate_every_steps','log_every_steps']:
        if exp['training'][key] <= 0: raise ValueError(key+' must be positive')
    if exp['validation']['size'] <= 0 or exp['training']['workers'] < 0:
        raise ValueError('Invalid validation size or worker count')
    if exp['embeddings']['attention_backend'] not in ('auto','math') or exp['embeddings']['memory_size'] < 0:
        raise ValueError('Invalid embedding backend or memory size')
    model = deepcopy(fixed['model'])
    model['noise'] = exp['training']['coordinate_noise']
    flags = model['params']['gcp']['module_cfg']
    flags.update(zero_seq_input=exp['inputs']['zero_plm'],
                 zero_struct_input=exp['inputs']['zero_structure'],
                 zero_rsa_input=exp['inputs']['zero_rsa'])
    model['feature_protocol'] = fixed['features']['protocol']
    model['input_policy'] = policy
    raw_paths = read_json(paths_file or MODEL/'paths.example.json')
    expected_paths = set(read_json(MODEL/'paths.example.json'))
    if set(raw_paths) != expected_paths:
        raise ValueError(f'Paths keys differ: {set(raw_paths)^expected_paths}')
    data_root = str(Path(raw_paths['data_root'].format(repo=str(REPO))).expanduser().resolve())
    paths = {}
    for key, value in raw_paths.items():
        paths[key] = str(Path(value.format(repo=str(REPO),data_root=data_root)).expanduser().resolve()) if value is not None else None
    from src.data_encoding import categ_to_resnames
    data = {
        'dataset_filepath': paths['dataset'], 'rsa_filepath': paths['rsa'],
        'features_flags': fixed['features']['flags'], 'preprocess': fixed['features']['preprocess'],
        'feature_protocol': fixed['features']['protocol'], 'max_num_nn': model['params']['gcp']['module_cfg']['max_nn'],
        'plm': fixed['features']['plm'], 'plm_fusion': 'concat',
        'l_types': categ_to_resnames['protein'],
        'r_types': [categ_to_resnames['protein'], categ_to_resnames['dna']+categ_to_resnames['rna'], categ_to_resnames['ion'], categ_to_resnames['ligand']],
    }
    output = dict(id=descriptor['id'], paper_name=descriptor['paper_name'], config_source=str(path.resolve()),
                  experiment=exp, fixed=fixed, config_model=model, config_data=data, paths=paths)
    output['training_fingerprint'] = fingerprint({k:output[k] for k in ('id','experiment','fixed','config_model')})
    return output

def parameter_decisions(resolved):
    """Every experiment leaf has an explicit runtime consumer; no unregistered knobs."""
    consumers = {
        'labels.native_only':'PaperDataset train/valid -> native contacts vs maximum(native, projected)',
        'sampling.cluster_balanced':'build_loaders -> WeightedRandomSampler vs shuffled chains',
        'loss.task_weighting':'weighted_bce -> normalized positive-ratio vs equal task factors',
        'loss.positive_factors':'weighted_bce -> factor*(1-positive_ratio)/positive_ratio',
        'loss.normalize_positive_weights':'weighted_bce -> optional within-column normalization',
        'inputs.zero_plm':'attach_embeddings + GCPNetModel.apply_input_ablation -> zero raw ESM features before projection; projection bias retained',
        'inputs.zero_structure':'GCPNetModel.apply_input_ablation -> zero structural scalars/edge scalar/vector',
        'inputs.zero_rsa':'GCPNetModel.apply_input_ablation -> zero RSA scalar only',
        'inputs.geometry_mode':'GCPNetModel.forward -> full geometry or geometry_mask_v2 zero tensors',
        'inputs.edge_mode':'input_policy -> spatial kNN, sequence kNN or empty [2,0] adjacency',
        'inputs.ablation_protocol':'resolver + GCPNetModel -> versioned input policy validation',
        'embeddings.attention_backend':'EmbeddingProvider -> explicit training SDPA backend (auto preserves historical training behavior)',
        'embeddings.memory_size':'EmbeddingProvider -> bounded sequence tensor memory',
        'active_tasks':'weighted_bce + validation + checkpoint selection; inactive task factors are zero',
        'training.epochs':'train loop range', 'training.seed':'Python/NumPy/Torch/CUDA/sampler/worker RNG',
        'training.batch_size':'DataLoader (paper implementation requires 1)',
        'training.learning_rate':'torch.optim.Adam', 'training.coordinate_noise':'GCPNetModel.augment_eps during train forward',
        'training.gradient_clip':'clip_grad_norm_', 'training.log_every_steps':'training metrics log cadence',
        'training.evaluate_every_steps':'validation + best-AUC selection cadence',
        'training.workers':'DataLoader num_workers',
        'training.device':'training and PLM execution device',
        'validation.size':'fixed validation subset construction', 'validation.seed':'fixed validation subset generator',
    }
    def leaves(obj,prefix=''):
        out={}
        for k,v in obj.items():
            key=prefix+k
            if isinstance(v,dict):out.update(leaves(v,key+'.'))
            else:out[key]=v
        return out
    values=leaves(resolved['experiment'])
    if set(values)!=set(consumers):raise ValueError('Config leaf/consumer mismatch')
    return [{'parameter':key,'value':value,'consumer':consumers[key]} for key,value in values.items()]
