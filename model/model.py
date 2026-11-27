"""GCP + ESM3 interface model used by GLAURI and versioned input ablations."""
import torch as pt
from src.model_operations import build_edge_index_and_features, orientations
import proteinworkshop.models.graph_encoders.layers.gcp as gcp
from functools import partial
from src.transform import build_b_cfg_from_a
import torch
import torch.nn as nn
import torch.nn.functional as F
from proteinworkshop.models.graph_encoders.components.wrappers import ScalarVector
from proteinworkshop.models.utils import centralize, localize
from proteinworkshop.types import EncoderOutput
from model.input_policy import PROTOCOL, ZeroRBF, apply_geometry_policy

def _zero_scalar_columns(t: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """把 t 的列区间 [lo, hi) 置零, 返回新张量(不做 in-place, 保证 autograd 安全)。"""
    lo = max(0, int(lo))
    hi = min(int(t.shape[1]), int(hi))
    if hi <= lo:
        return t
    parts = []
    if lo > 0:
        parts.append(t[:, :lo])
    parts.append(pt.zeros_like(t[:, lo:hi]))
    if hi < t.shape[1]:
        parts.append(t[:, hi:])
    return pt.cat(parts, dim=1) if len(parts) > 1 else parts[0]

def apply_input_ablation(h, e, xi, esm_offset: int, rsa_dim: int, zero_seq: bool, zero_struct: bool, zero_rsa: bool, seq_dim: int=0):
    """Zero selected ESM3, structural or RSA features while preserving tensor dimensions."""
    struct_dim = int(esm_offset) - int(rsa_dim)
    if zero_seq:
        seq_end = int(esm_offset) + int(seq_dim) if seq_dim else h.shape[1]
        h = _zero_scalar_columns(h, esm_offset, min(seq_end, h.shape[1]))
    if zero_struct:
        h = _zero_scalar_columns(h, 0, struct_dim)
        e = pt.zeros_like(e)
        xi = pt.zeros_like(xi)
    if zero_rsa:
        h = _zero_scalar_columns(h, struct_dim, esm_offset)
    return (h, e, xi)

class GCPNetModel(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        cfg = build_b_cfg_from_a(config['params'])
        module_cfg = cfg['module_cfg']
        model_cfg = cfg['model_cfg']
        layer_cfg = cfg['layer_cfg']
        self.max_nn = module_cfg['max_nn']
        self.feature_protocol=config.get('feature_protocol','legacy_nodes_angles_sidechain_v1_20261102')
        self.predict_node_rep = module_cfg.predict_node_rep
        self.orientations = module_cfg.get('orientations', False)
        self.seq_in = model_cfg.get('seq_in', 1536)
        self.sc_vector = module_cfg.get('sidechain_vector', False)
        self._esm_scalar_offset = max(0, model_cfg.h_input_dim - self.seq_in)
        self.esm_proj_dim = module_cfg.get('esm_proj_dim', None)
        self.esm_proj_drop = module_cfg.get('esm_proj_dropout', 0.0)
        if self.esm_proj_dim:
            self.esm_proj = pt.nn.Sequential(pt.nn.Linear(self.seq_in, self.esm_proj_dim), pt.nn.ELU())
        else:
            self.esm_proj = None
        self.rsa_scalar_dim = int(module_cfg.get('rsa_scalar_dim', 0))
        self.zero_seq_input = module_cfg.get('zero_seq_input', False)
        self.zero_struct_input = module_cfg.get('zero_struct_input', False)
        self.zero_rsa_input = module_cfg.get('zero_rsa_input', False)
        policy = config.get('input_policy', {})
        if policy and (set(policy)!={'geometry_mode','edge_mode','ablation_protocol'} or policy['ablation_protocol'] not in ('legacy',PROTOCOL)):
            raise ValueError('Unsupported or incomplete input policy')
        self.geometry_mode = policy.get('geometry_mode', 'full')
        self.edge_mode = policy.get('edge_mode', 'spatial')
        self.ablation_protocol = policy.get('ablation_protocol', 'legacy')
        if self.geometry_mode not in ('full', 'zero'):
            raise ValueError('Unknown geometry_mode')
        if self.geometry_mode == 'zero':
            if self.ablation_protocol != PROTOCOL or self.edge_mode not in ('sequence', 'none'):
                raise ValueError('Zero geometry requires geometry_mask_v2 and sequence/none edges')
            if config['noise'] != 0 or self.zero_struct_input:
                raise ValueError('Zero geometry forbids coordinate noise and legacy zero_struct_input')
            if not self.orientations or not self.sc_vector:
                raise ValueError('Preserve architecture channels; zero vectors explicitly, do not disable their channel flags')
            if self._esm_scalar_offset != 36 or self.seq_in != 1536 or self.rsa_scalar_dim != 1:
                raise ValueError('Unexpected scalar layout for geometry_mask_v2')
        elif self.edge_mode != 'spatial':
            raise ValueError('Full geometry requires spatial edges')
        if self.rsa_scalar_dim < 0 or self.rsa_scalar_dim > self._esm_scalar_offset:
            raise ValueError(f'rsa_scalar_dim={self.rsa_scalar_dim} 越界: 结构标量段只有 {self._esm_scalar_offset} 列(h_input_dim - seq_in)')
        if self.zero_rsa_input and self.rsa_scalar_dim <= 0:
            raise ValueError("zero_rsa_input=True 需要 rsa_scalar_dim>0; 请在 config 里设置 module_cfg['rsa_scalar_dim'](带 RSA 的 cat36 配置应为 1)")
        edge_input_dims = ScalarVector(model_cfg.e_input_dim, model_cfg.xi_input_dim)
        node_vec_in = model_cfg.chi_input_dim + (1 if self.sc_vector else 0)
        self.node_vector_channels = node_vec_in
        _node_scalar_in = self._esm_scalar_offset + self.esm_proj_dim if self.esm_proj_dim else model_cfg.h_input_dim
        node_input_dims = ScalarVector(_node_scalar_in, node_vec_in)
        self.edge_dims = ScalarVector(model_cfg.e_hidden_dim, model_cfg.xi_hidden_dim)
        self.node_dims = ScalarVector(model_cfg.h_hidden_dim, model_cfg.chi_hidden_dim)
        self.augment_eps = config['noise']
        self.centralize = partial(centralize, key='pos')
        self.localize = partial(localize, norm_pos_diff=module_cfg.norm_pos_diff)
        self.gcp_embedding = gcp.GCPEmbedding(edge_input_dims, node_input_dims, self.edge_dims, self.node_dims, cfg=module_cfg)
        if self.geometry_mode == 'zero':
            self.gcp_embedding.radial_embedding = ZeroRBF(module_cfg.num_rbf)
        self.interaction_layers = nn.ModuleList((gcp.GCPInteractions(self.node_dims, self.edge_dims, cfg=module_cfg, layer_cfg=layer_cfg, dropout=model_cfg.dropout) for _ in range(model_cfg.num_layers)))
        if self.predict_node_rep:
            self.invariant_node_projection = nn.ModuleList([gcp.GCPLayerNorm(self.node_dims), gcp.GCP(self.node_dims, (self.node_dims.scalar, 0), nonlinearities=tuple(module_cfg.nonlinearities), scalar_gate=module_cfg.scalar_gate, vector_gate=module_cfg.vector_gate, enable_e3_equivariance=module_cfg.enable_e3_equivariance, node_inputs=True)])
        self.dm = pt.nn.Sequential(pt.nn.Linear(model_cfg.h_hidden_dim, int(model_cfg.h_hidden_dim / 2)), pt.nn.ELU(), pt.nn.Linear(int(model_cfg.h_hidden_dim / 2), int(model_cfg.h_hidden_dim / 8)), pt.nn.ELU(), pt.nn.Linear(int(model_cfg.h_hidden_dim / 8), int(model_cfg.output_dim)))

    def required_batch_attributes(self):
        return ['edge_index', 'pos', 'x', 'batch']

    def forward(self, batch):
        protocol=getattr(batch,'feature_protocol',None)
        actual=set(protocol if isinstance(protocol,list) else [protocol])
        if self.feature_protocol=='ca_ncac_sidechain_v1' and actual!={self.feature_protocol}:
            raise ValueError(f'CA checkpoint requires CA-protocol graphs, received {actual}')
        if self.geometry_mode == 'zero':
            apply_geometry_policy(batch, self.edge_mode, self.max_nn, self.node_vector_channels, self.zero_rsa_input)
        else:
            self._full_geometry_inputs(batch)
        if self.zero_seq_input or self.zero_struct_input or self.zero_rsa_input:
            batch.h, batch.e, batch.xi = apply_input_ablation(batch.h, batch.e, batch.xi, esm_offset=self._esm_scalar_offset, rsa_dim=self.rsa_scalar_dim, seq_dim=0, zero_seq=self.zero_seq_input, zero_struct=self.zero_struct_input, zero_rsa=self.zero_rsa_input)
        if self.esm_proj is not None:
            off = self._esm_scalar_offset
            end = off + self.seq_in
            esm_part = self.esm_proj(batch.h[:, off:end])
            if self.esm_proj_drop > 0.0:
                esm_part = F.dropout(esm_part, p=self.esm_proj_drop, training=self.training)
            batch.h = pt.cat([batch.h[:, :off], esm_part, batch.h[:, end:]], dim=1)
        (h, chi), (e, xi) = self.gcp_embedding(batch)
        for layer in self.interaction_layers:
            (h, chi), batch.pos = layer((h, chi), (e, xi), batch.edge_index, batch.f_ij, node_pos=batch.pos)
        batch.h, batch.chi, batch.e, batch.xi = (h, chi, e, xi)
        encoder_outputs = {}
        out = h
        if self.predict_node_rep:
            out = self.invariant_node_projection[0](ScalarVector(h, chi))
            out = self.invariant_node_projection[1](out, batch.edge_index, batch.f_ij, node_inputs=True)
        assert batch.M.shape[0] == out.shape[0], f'M rows {batch.M.shape[0]} != nodes {out.shape[0]}'
        out = self.dm(out)
        encoder_outputs['node_embedding'] = out
        return EncoderOutput(encoder_outputs)

    def _full_geometry_inputs(self, batch):
        # Preserve the legacy/full computation order and RNG consumption.
        if self.training and self.augment_eps > 0:
            batch.pos = batch.pos + self.augment_eps * pt.randn_like(batch.pos)
        pos_centroid, batch.pos = self.centralize(batch, batch_index=batch.batch)
        if self.orientations:
            p = orientations(batch.pos, pt.Tensor([0, len(batch.pos)]))
        else:
            p = pt.zeros((batch.x.shape[0], batch.pos.shape[1]), device=batch.pos.device).unsqueeze(1)
        if self.sc_vector:
            sc = getattr(batch, 'sc', None)
            if sc is None:
                raise ValueError('sidechain_vector=True requires a real batch.sc tensor')
            sc = sc.to(batch.pos.device, dtype=p.dtype)
            p = pt.cat([p, sc.unsqueeze(1)], dim=1)
        batch.x_vector_attr = p
        batch.edge_index, D_nn, R_nn = build_edge_index_and_features(batch.pos, self.max_nn)
        r_flat = R_nn
        d_flat = D_nn
        batch.edge_attr = torch.cat([d_flat, batch.edge_index[0].unsqueeze(1).float(), batch.edge_index[1].unsqueeze(1).float()], dim=1)
        batch.edge_vector_attr = r_flat.unsqueeze(1)
        batch.h, batch.chi, batch.e, batch.xi = (batch.x, batch.x_vector_attr, batch.edge_attr, batch.edge_vector_attr)
        batch.f_ij = self.localize(batch.pos, batch.edge_index)
