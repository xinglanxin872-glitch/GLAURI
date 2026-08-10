import torch as pt

from src.model_operations import build_edge_index_and_features, orientations
import proteinworkshop.models.graph_encoders.layers.gcp as gcp

from functools import partial
from src.model_operations import build_b_cfg_from_a
import torch
import torch.nn as nn
import torch.nn.functional as F

from proteinworkshop.models.graph_encoders.components.wrappers import (
    ScalarVector,
)
from proteinworkshop.models.utils import (
    centralize,
    decentralize,
    get_aggregation,
    localize,
)
from proteinworkshop.types import EncoderOutput


def _zero_scalar_columns(t: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Zero out the column range [lo, hi) of t and return a new tensor (not in-place, autograd-safe)."""
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


def apply_input_ablation(
    h, e, xi,
    esm_offset: int,
    rsa_dim: int,
    zero_seq: bool,
    zero_struct: bool,
    zero_rsa: bool,
):
    """Input feature ablation (always active, consistent between training and validation).

    Column layout of the node scalars h (concat mode):
        [0, struct_dim)      structure scalars: residue one-hot + dihedral encoding
        [struct_dim, esm_offset)  per-residue RSA (rsa_dim columns; under cat35 rsa_dim=0 and this segment is empty)
        [esm_offset, D)      PLM (ESM3) embedding
    where struct_dim = esm_offset - rsa_dim, esm_offset = h_input_dim - seq_in.
    In series mode esm_offset = 0 and all of h is PLM.

    - zero_seq:    zero out the PLM segment; structure scalars / RSA / edge features untouched.
    - zero_struct: zero out the structure scalar segment (one-hot + dihedrals) plus the edge scalar e and edge vector xi;
                   RSA and PLM segments untouched. The edge_index topology and f_ij local frames are kept (required by GCP).
    - zero_rsa:    zero out only the RSA segment; everything else untouched.
    The three switches are independent and can be combined.
    """
    struct_dim = int(esm_offset) - int(rsa_dim)
    if zero_seq:
        h = _zero_scalar_columns(h, esm_offset, h.shape[1])
    if zero_struct:
        h = _zero_scalar_columns(h, 0, struct_dim)
        e = pt.zeros_like(e)
        xi = pt.zeros_like(xi)
    if zero_rsa:
        h = _zero_scalar_columns(h, struct_dim, esm_offset)
    return h, e, xi


class GCPNetModel(torch.nn.Module):
    def __init__(
        self,config,
        num_layers: int = 5,
        node_s_emb_dim: int = 128,
        node_v_emb_dim: int = 16,
        edge_s_emb_dim: int = 32,
        edge_v_emb_dim: int = 4,
        r_max: float = 10.0,
        num_rbf: int = 8,
        activation: str = "silu",
        pool: str = "sum",
        # Note: Each of the arguments above are stored in the corresponding `kwargs` configs below
        # They are simply listed here to highlight key available arguments
        **kwargs,
    ):

        super().__init__()
        cfg=build_b_cfg_from_a(config["params"])
        # assert all(
        #     [cfg in kwargs for cfg in ["module_cfg", "model_cfg", "layer_cfg"]]
        # ), "All required GCPNet `DictConfig`s must be provided."
        module_cfg = cfg["module_cfg"]
        model_cfg = cfg["model_cfg"]
        layer_cfg = cfg["layer_cfg"]

        self.max_nn = module_cfg["max_nn"]
        self.predict_node_pos = module_cfg.predict_node_positions
        self.predict_node_rep = module_cfg.predict_node_rep
        self.orientations = module_cfg.get("orientations",False)

        self.seq_in=model_cfg.get("seq_in",1280)
        # Side-chain orientation vector switch (for ablation): when enabled, append 1 channel to the node vector features
        # (unit vector = normalize(side-chain heavy-atom centroid - CA); zero vector for GLY/missing)
        self.sc_vector = module_cfg.get("sidechain_vector", False)
        # offset = h_input_dim - seq_in: series (1280-1280=0), concat (36+1536-1536=36).
        # i.e. the start of the PLM segment within the node scalars; the preceding offset columns are native structure scalars (last rsa_scalar_dim columns are RSA).
        self._esm_scalar_offset = max(0, model_cfg.h_input_dim - self.seq_in)
        # ==== Pre-ESM compression (esm_proj): before gcp_embedding, first compress the ESM segment of batch.h
        #      from seq_in -> esm_proj_dim (Linear+ELU), optional dropout, then concat back with the structure scalars in the preceding offset columns.
        #      Default None = disabled, keeping existing behavior (ESM goes into gcp_embedding as-is, which compresses it to h_hidden_dim).
        #      A nonlinearity is required: otherwise Linear(seq_in->esm_proj_dim) followed by gcp_embedding's Linear is still a linear
        #      composition, rank-equivalent to a direct concat (both output h_hidden_dim), and the ablation would measure nothing.
        self.esm_proj_dim = module_cfg.get("esm_proj_dim", None)
        self.esm_proj_drop = module_cfg.get("esm_proj_dropout", 0.0)
        if self.esm_proj_dim:
            self.esm_proj = pt.nn.Sequential(
                pt.nn.Linear(self.seq_in, self.esm_proj_dim),
                pt.nn.ELU(),
            )
        else:
            self.esm_proj = None
        # ==== Input ablation switches (always active, consistent between training and validation) ====
        # Node scalar column layout: [structure one-hot+dihedrals | RSA (rsa_scalar_dim columns) | PLM (seq_in columns)]
        #   offset = h_input_dim - seq_in is the start of the PLM segment; RSA is cat'ed by data_handler at the end
        #   of the structure scalars, so the RSA segment = [offset - rsa_scalar_dim, offset). cat35 has no RSA -> rsa_scalar_dim=0.
        # zero_seq_input:    zero out the PLM segment; structure scalars/RSA/edge features untouched -> "no sequence features" ablation.
        # zero_struct_input: zero out the structure scalar segment (one-hot + dihedrals) + edge scalar batch.e + edge vector batch.xi,
        #                    RSA and PLM segments kept; edge_index topology and f_ij local frames kept (required by GCP).
        # zero_rsa_input:    zero out only the RSA segment -> "no RSA" ablation (parameter count/initialization identical to baseline).
        self.rsa_scalar_dim = int(module_cfg.get("rsa_scalar_dim", 0))
        self.zero_seq_input = module_cfg.get("zero_seq_input", False)
        self.zero_struct_input = module_cfg.get("zero_struct_input", False)
        self.zero_rsa_input = module_cfg.get("zero_rsa_input", False)
        if self.rsa_scalar_dim < 0 or self.rsa_scalar_dim > self._esm_scalar_offset:
            raise ValueError(
                f"rsa_scalar_dim={self.rsa_scalar_dim} 越界: 结构标量段只有 "
                f"{self._esm_scalar_offset} 列(h_input_dim - seq_in)"
            )
        if self.zero_rsa_input and self.rsa_scalar_dim <= 0:
            # Without this hard error, zero_rsa would silently zero nothing, producing a bogus "RSA has no contribution" ablation result.
            raise ValueError(
                "zero_rsa_input=True 需要 rsa_scalar_dim>0; "
                "请在 config 里设置 module_cfg['rsa_scalar_dim'](带 RSA 的 cat36 配置应为 1)"
            )
        # Feature dimensionalities
        edge_input_dims = ScalarVector(model_cfg.e_input_dim, model_cfg.xi_input_dim)
        # chi_input_dim describes the base number of vector channels (corresponding to orientations); +1 more when sidechain is enabled
        node_vec_in = model_cfg.chi_input_dim + (1 if self.sc_vector else 0)
        # With esm_proj enabled, the scalar width into gcp_embedding = structure scalars (offset) + compressed ESM (esm_proj_dim)
        _node_scalar_in = (
            self._esm_scalar_offset + self.esm_proj_dim
            if self.esm_proj_dim
            else model_cfg.h_input_dim
        )
        node_input_dims = ScalarVector(_node_scalar_in, node_vec_in)
        self.edge_dims = ScalarVector(model_cfg.e_hidden_dim, model_cfg.xi_hidden_dim)
        self.node_dims = ScalarVector(model_cfg.h_hidden_dim, model_cfg.chi_hidden_dim)
        self.augment_eps = config['noise']
        # Position-wise operations
        self.centralize = partial(centralize, key="pos")
        self.localize = partial(localize, norm_pos_diff=module_cfg.norm_pos_diff)
        self.decentralize = partial(decentralize, key="pos")


        # Input embeddings
        self.gcp_embedding = gcp.GCPEmbedding(
            edge_input_dims,
            node_input_dims,
            self.edge_dims,
            self.node_dims,
            cfg=module_cfg,
        )

        # Message-passing layers
        self.interaction_layers = nn.ModuleList(
            gcp.GCPInteractions(
                self.node_dims,
                self.edge_dims,
                cfg=module_cfg,
                layer_cfg=layer_cfg,
                dropout=model_cfg.dropout,
            )
            for _ in range(model_cfg.num_layers)
        )

        if self.predict_node_rep:
            # Predictions
            self.invariant_node_projection = nn.ModuleList(
                [
                    gcp.GCPLayerNorm(self.node_dims),
                    gcp.GCP(
                        # Note: `GCPNet` defaults to providing SE(3) equivariance
                        # It is possible to provide E(3) equivariance by instead setting `module_cfg.enable_e3_equivariance=true`
                        self.node_dims,
                        (self.node_dims.scalar, 0),
                        nonlinearities=tuple(module_cfg.nonlinearities),
                        scalar_gate=module_cfg.scalar_gate,
                        vector_gate=module_cfg.vector_gate,
                        enable_e3_equivariance=module_cfg.enable_e3_equivariance,
                        node_inputs=True,
                    ),
                ]
            )
        # Global pooling/readout function
        self.readout = get_aggregation(
            module_cfg.pool
        ) #128 64 32 4     96  48 12
        self.dm = pt.nn.Sequential(
            pt.nn.Linear(model_cfg.h_hidden_dim, int(model_cfg.h_hidden_dim/2)),
            pt.nn.ELU(),
            pt.nn.Linear(int(model_cfg.h_hidden_dim/2), int(model_cfg.h_hidden_dim/8)),#8
            pt.nn.ELU(),
            pt.nn.Linear(int(model_cfg.h_hidden_dim/8), int(model_cfg.output_dim)),#8
        )
    def required_batch_attributes(self):
        return ["edge_index", "pos", "x", "batch"]

    def forward(self, batch) :
        if self.training and self.augment_eps > 0:
            batch.pos = batch.pos + self.augment_eps * pt.randn_like(batch.pos)

        pos_centroid, batch.pos = self.centralize(batch, batch_index=batch.batch)

        if self.orientations:
            p=orientations(batch.pos,pt.Tensor([0,len(batch.pos)]))

        else:
            p=pt.zeros((batch.x.shape[0],batch.pos.shape[1]), device=batch.pos.device).unsqueeze(1)

        if self.sc_vector:
            # Append the side-chain orientation unit-vector channel (N,3) -> (N,1,3); when missing it should be the zero vector filled in by the dataset
            sc = getattr(batch, "sc", None)
            if sc is None:
                sc = pt.zeros((batch.x.shape[0], batch.pos.shape[1]), device=batch.pos.device)
            sc = sc.to(batch.pos.device, dtype=p.dtype)
            p = pt.cat([p, sc.unsqueeze(1)], dim=1)   # (N, base+1, 3)


        batch.x_vector_attr=p

        batch.edge_index, D_nn, R_nn = build_edge_index_and_features(batch.pos,self.max_nn)
        r_flat = R_nn
        d_flat = D_nn

        batch.edge_attr = torch.cat([
                            d_flat,               # [E, 1] your scalar feature
                            batch.edge_index[0].unsqueeze(1).float(),  # [E, 1] source node index
                            batch.edge_index[1].unsqueeze(1).float()   # [E, 1] target node index
                        ], dim=1)  # [E, 3]
        batch.edge_vector_attr=r_flat.unsqueeze(1)


        batch.h, batch.chi, batch.e, batch.xi = (
            batch.x,
            batch.x_vector_attr,
            batch.edge_attr,
            batch.edge_vector_attr,
        )


        # Craft complete local frames corresponding to each edge
        batch.f_ij = self.localize(batch.pos, batch.edge_index)
        # Input ablation: zero out entire blocks before the ESM proj, so downstream branches receive the ablated input.
        if self.zero_seq_input or self.zero_struct_input or self.zero_rsa_input:
            batch.h, batch.e, batch.xi = apply_input_ablation(
                batch.h, batch.e, batch.xi,
                esm_offset=self._esm_scalar_offset,
                rsa_dim=self.rsa_scalar_dim,
                zero_seq=self.zero_seq_input,
                zero_struct=self.zero_struct_input,
                zero_rsa=self.zero_rsa_input,
            )
        # Pre-ESM compression: seq_in -> esm_proj_dim (+ELU), optional dropout, keeping the first offset columns of structure scalars unchanged
        if self.esm_proj is not None:
            off = self._esm_scalar_offset
            esm_part = self.esm_proj(batch.h[:, off:])
            if self.esm_proj_drop > 0.0:
                esm_part = F.dropout(esm_part, p=self.esm_proj_drop, training=self.training)
            batch.h = pt.cat([batch.h[:, :off], esm_part], dim=1) if off > 0 else esm_part
        # Embed node and edge input features
        (h, chi), (e, xi) = self.gcp_embedding(batch)



        # Update graph features using a series of geometric message-passing layers
        for layer in self.interaction_layers:
            (h, chi), batch.pos = layer(
                (h, chi),
                (e, xi),
                batch.edge_index,
                batch.f_ij,
                node_pos=batch.pos,
            )

        # Record final version of each feature in `Batch` object
        batch.h, batch.chi, batch.e, batch.xi = h, chi, e, xi

        # initialize encoder outputs
        encoder_outputs = {}

        out = h



        if self.predict_node_rep:
            out = self.invariant_node_projection[0](
                ScalarVector(h, chi)
            )  # e.g., GCPLayerNorm()
            out = self.invariant_node_projection[1](
                out, batch.edge_index, batch.f_ij, node_inputs=True
            )  # e.g., GCP((h, chi)) -> h'
        assert batch.M.shape[0] == out.shape[0], \
    f"M rows {batch.M.shape[0]} != nodes {out.shape[0]}"


        encoder_outputs["graph_embedding"] = self.readout(
            out, batch.batch
        )
        #out = batch.M.T @ out
        out=self.dm(out)

        encoder_outputs["node_embedding"] = out
        return EncoderOutput(encoder_outputs)
