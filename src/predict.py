"""Shared structure/graph prediction API used by CLI and evaluation callers."""
import torch
from src.embeddings import EmbeddingProvider
from src.utils_eval import process_structure, process_seq_batch, inference


def predict_graphs(loaded, graphs, sequence, provider=None, embedding=None):
    device = next(loaded.model.parameters()).device
    if embedding is None and loaded.config_data.get('plm'):
        provider = provider or EmbeddingProvider(device, backend='math')
        embedding = provider.get(sequence)
    return inference(loaded.model, device, graphs, loaded.config_model.get('name', 'gcp'),
                     plm_embedding=embedding, plm_fusion=loaded.config_data.get('plm_fusion', 'concat'))


def predict_structure(loaded, path, chain, provider=None):
    data = loaded.config_data
    structure = process_structure(str(path), chainid=f'{chain}:0')
    if not isinstance(structure, dict) or not structure:
        raise ValueError(f'Cannot read chain {chain!r} from {path}')
    graphs, sequence = process_seq_batch(structure, data['features_flags'], data.get('preprocess'),
        loaded.config_model.get('name', 'gcp'), data.get('max_num_nn', 64),
        with_rsa=data.get('rsa_filepath') is not None, protocol=loaded.metadata['feature_protocol'])
    positions = graphs[0].pos.detach().cpu().numpy().copy()
    return predict_graphs(loaded, graphs, sequence, provider), sequence, positions
