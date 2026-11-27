"""Geometry operations for the GLAURI GCP encoder."""
import torch as pt
import torch.nn.functional as F
from torch_cluster import knn_graph

def orientations(X, coords_slice_index=None):
    batch_num_nodes = X.shape[0]
    if batch_num_nodes == 0:
        return pt.zeros((0, 2, X.shape[-1]), device=X.device, dtype=X.dtype)
    if coords_slice_index is None:
        coords_slice_index = [0, batch_num_nodes]
    slice_index = coords_slice_index[1:] - 1
    if len(slice_index) == 1:
        last_node_index = pt.tensor([slice_index[0].item()], dtype=pt.long)
        first_node_index = pt.tensor([0])
    else:
        last_node_index = slice_index[:-1]
        first_node_index = slice_index[:-1] + 1
    slice_mask = pt.zeros(batch_num_nodes - 1, dtype=pt.bool)
    last_node_forward_slice_mask = slice_mask.clone()
    first_node_backward_slice_mask = slice_mask.clone()
    forward_slice = X[1:] - X[:-1]
    backward_slice = X[:-1] - X[1:]
    if len(last_node_index) > 1:
        last_node_forward_slice_mask[last_node_index] = True
    if len(first_node_index) > 0:
        backward_indices = first_node_index - 1
        valid_backward_mask = backward_indices >= 0
        if valid_backward_mask.any():
            first_node_backward_slice_mask[backward_indices[valid_backward_mask]] = True
    forward_slice[last_node_forward_slice_mask] = 0.0
    backward_slice[first_node_backward_slice_mask] = 0.0
    forward = pt.nan_to_num(pt.div(forward_slice, pt.norm(forward_slice, dim=-1, keepdim=True)))
    backward = pt.nan_to_num(pt.div(backward_slice, pt.norm(backward_slice, dim=-1, keepdim=True)))
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    orientations = pt.cat((forward.unsqueeze(-2), backward.unsqueeze(-2)), dim=-2)
    return orientations

def build_edge_index_and_features(pos, max_nn, batch=None):
    """
    基于坐标构建k-NN图并计算空间特征
    
    参数:
    - pos: 节点坐标 [N, 3]
    - max_nn: 每个节点的最大邻居数 (k)
    - batch: 批处理索引 [N], 确保邻居搜索不会跨越不同的样本
    
    返回:
    - edge_index: 边索引 [2, E]
    - D_nn: 邻居间的欧氏距离 [E, 1]
    - R_nn: 邻居间的相对位移向量 [E, 3]
    """
    min_threshold = 0.01
    edge_index = knn_graph(pos, k=max_nn, batch=batch, loop=False)
    row, col = edge_index
    R_vec = pos[col] - pos[row]
    D_nn = pt.norm(R_vec, p=2, dim=-1, keepdim=True)
    mask = D_nn.view(-1) > min_threshold
    if not mask.all():
        edge_index = edge_index[:, mask]
        R_vec = R_vec[mask]
        D_nn = D_nn[mask]
    R_nn = R_vec / D_nn.clamp(min=1e-08)
    return (edge_index, D_nn, R_nn)
