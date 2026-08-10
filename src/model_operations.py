import torch as pt
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from omegaconf import OmegaConf


def build_b_cfg_from_a(a_cfg: dict):
    """将 A 的 python config 转成 B (proteinworkshop GCP layers) 期望的 DictConfig.

    Moved here from the old src/transform.py (2026-08 release cleanup) — that file's only
    other content (TRAIN_DICT/register_transforms/IdentityTransform/gcp_adaptor/get_transform)
    was dead/broken code with no callers anywhere in the tree; this was the one function
    model.py actually imports.
    """
    b_cfg_dict = {
        "model_cfg": a_cfg["gcp"]["model_cfg"],
        "layer_cfg": a_cfg["gcp"]["layer_cfg"],
        "module_cfg": a_cfg["gcp"]["module_cfg"],
    }
    return OmegaConf.create(b_cfg_dict)


# >> UTILS
def unpack_state_features(X, ids_topk, q):
    # compute displacement vectors
    R_nn = X[ids_topk-1] - X.unsqueeze(1)#[N_atom,64,3]
    # compute distance matrix
    D_nn = pt.norm(R_nn, dim=2)#[N_atom,64]计算距离
    # mask distances
    D_nn = D_nn + pt.max(D_nn)*(D_nn < 1e-2).float()#加入惩罚值
    # normalize displacement vectors
    R_nn = R_nn / D_nn.unsqueeze(2)#[N_atom,64,3]方向向量

    # prepare sink
    q = pt.cat([pt.zeros((1, q.shape[1]), device=q.device), q], dim=0)#[N_atom+1,32]
    ids_topk = pt.cat([pt.zeros((1, ids_topk.shape[1]), dtype=pt.long, device=ids_topk.device), ids_topk], dim=0)#[N_atom+1,64]
    D_nn = pt.cat([pt.zeros((1, D_nn.shape[1]), device=D_nn.device), D_nn], dim=0)#[N_atom+1,64]
    R_nn = pt.cat([pt.zeros((1, R_nn.shape[1], R_nn.shape[2]), device=R_nn.device), R_nn], dim=0)#[N_atom+1,64,3]

    return q, ids_topk, D_nn, R_nn
from torch_cluster import knn_graph


def orientations(
    X, 
    coords_slice_index=None
):
    batch_num_nodes = X.shape[0]
    if batch_num_nodes == 0:
        return pt.zeros((0, 2, X.shape[-1]), device=X.device, dtype=X.dtype)

    # 处理单个子图的情况
    if coords_slice_index is None:
        # 单个子图：边界是第一个和最后一个节点
        coords_slice_index = [0, batch_num_nodes]
    
    # 原有逻辑保持不变
    slice_index = coords_slice_index[1:] - 1  # 最后一个节点的索引
    
    # 处理边界情况：当只有一个子图时，last_node_index 和 first_node_index 应该包含边界
    if len(slice_index) == 1:
        # 只有一个子图的情况
        last_node_index = pt.tensor([slice_index[0].item()], dtype=pt.long)  # 最后一个节点
        first_node_index = pt.tensor([0])  # 第一个节点
    else:
        # 多个子图的情况
        last_node_index = slice_index[:-1]  # 除了最后一个子图外的所有子图的最后一个节点
        first_node_index = slice_index[:-1] + 1  # 除了第一个子图外的所有子图的第一个节点
    
    # 创建掩码
    slice_mask = pt.zeros(batch_num_nodes - 1, dtype=pt.bool)
    last_node_forward_slice_mask = slice_mask.clone()
    first_node_backward_slice_mask = slice_mask.clone()
    
    # 计算方向向量
    forward_slice = X[1:] - X[:-1]
    backward_slice = X[:-1] - X[1:]
    
    # 设置边界节点掩码
    if len(last_node_index) > 1:
        last_node_forward_slice_mask[last_node_index] = True
    if len(first_node_index) > 0:
        # 注意：first_node_index 是节点索引，但掩码是对应于向量的索引
        # 第一个子图的第一个节点（索引0）没有向后的向量
        backward_indices = first_node_index - 1
        # 确保索引在有效范围内
        valid_backward_mask = backward_indices >= 0
        if valid_backward_mask.any():
            first_node_backward_slice_mask[backward_indices[valid_backward_mask]] = True
    
    # 边界节点方向向量设为0
    forward_slice[last_node_forward_slice_mask] = 0.0
    backward_slice[first_node_backward_slice_mask] = 0.0
    
    # 归一化和填充
    forward = pt.nan_to_num(
        pt.div(forward_slice, pt.norm(forward_slice, dim=-1, keepdim=True))
    )
    backward = pt.nan_to_num(
        pt.div(backward_slice, pt.norm(backward_slice, dim=-1, keepdim=True))
    )

    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    orientations = pt.cat((forward.unsqueeze(-2), backward.unsqueeze(-2)), dim=-2)
    
    return orientations# NOTE: this handles all but the first node in the first subgraph

def orientations_forward(X, coords_slice_index=None):
    batch_num_nodes = X.shape[0]
    
    # 处理单个子图的情况
    if coords_slice_index is None:
        # 单个子图：边界是第一个和最后一个节点
        coords_slice_index = [0, batch_num_nodes]
    
    # 原有逻辑保持不变
    slice_index = [x-1 for x in coords_slice_index[1:]] # 最后一个节点的索引
    
    # 处理边界情况：当只有一个子图时，last_node_index 和 first_node_index 应该包含边界
    if len(slice_index) == 1:
        # 只有一个子图的情况
        last_node_index = pt.tensor([slice_index[0]], dtype=pt.long)  # 最后一个节点
        first_node_index = pt.tensor([0])  # 第一个节点
    else:
        # 多个子图的情况
        last_node_index = slice_index[:-1]  # 除了最后一个子图外的所有子图的最后一个节点
        first_node_index = slice_index[:-1] + 1  # 除了第一个子图外的所有子图的第一个节点
    
    # 创建掩码
    slice_mask = pt.zeros(batch_num_nodes - 1, dtype=pt.bool)
    last_node_forward_slice_mask = slice_mask.clone()
    first_node_backward_slice_mask = slice_mask.clone()
    
    # 计算方向向量
    forward_slice = X[1:] - X[:-1]
    backward_slice = X[:-1] - X[1:]
    
    # 设置边界节点掩码
    if len(last_node_index) > 1:
        last_node_forward_slice_mask[last_node_index] = True
    if len(first_node_index) > 0:
        # 注意：first_node_index 是节点索引，但掩码是对应于向量的索引
        # 第一个子图的第一个节点（索引0）没有向后的向量
        backward_indices = first_node_index - 1
        # 确保索引在有效范围内
        valid_backward_mask = backward_indices >= 0
        if valid_backward_mask.any():
            first_node_backward_slice_mask[backward_indices[valid_backward_mask]] = True
    
    # 边界节点方向向量设为0
    forward_slice[last_node_forward_slice_mask] = 0.0
    backward_slice[first_node_backward_slice_mask] = 0.0
    
    # 归一化和填充
    forward = pt.nan_to_num(
        pt.div(forward_slice, pt.norm(forward_slice, dim=-1, keepdim=True))
    )
    backward = pt.nan_to_num(
        pt.div(backward_slice, pt.norm(backward_slice, dim=-1, keepdim=True))
    )

    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    #orientations = pt.cat((forward.unsqueeze(-2), backward.unsqueeze(-2)), dim=-2)
    
    return forward# NOTE: this handles all but the first node in the first subgraph

def build_edge_index_and_features_from_idx(X, ids_topk, q):
    """
    X: [N_atom, 3] 节点坐标
    ids_topk: [N_atom, nn] 每个节点的邻居索引 (1-based)
    q: [N_atom, feature_dim] 节点特征

    返回:
        edge_index: [2, E]，边索引
        D_nn: [E]，每条边的距离
        R_nn: [E, 3]，每条边的单位方向向量
    """
    N_atom = X.shape[0]
    nn = ids_topk.shape[1]
    device = X.device

    # 1️⃣ ids_topk 合法性裁剪：不超过节点范围，最小1 (1-based)
    ids_topk = ids_topk.clamp(1, N_atom)

    # 2️⃣ 转换为 0-based 索引
    ids_topk_0b = ids_topk - 1  # [N_atom, nn]

    # 3️⃣ 生成 edge_index
    # src = 每个节点重复 nn 次
    src = pt.arange(N_atom, device=device).unsqueeze(1).repeat(1, nn)  # [N_atom, nn]
    dst = ids_topk_0b                                                   # [N_atom, nn]
    edge_index = pt.stack([src.flatten(), dst.flatten()], dim=0)     # [2, E], E = N_atom*nn

    # 4️⃣ 根据 edge_index 计算 R_nn 和 D_nn
    pos_src = X[edge_index[0]]  # [E, 3]
    pos_dst = X[edge_index[1]]  # [E, 3]

    r_vec = pos_dst - pos_src          # [E, 3]
    d_nn = pt.norm(r_vec, dim=1)    # [E]
    # 避免除零
    d_nn = d_nn + (d_nn < 1e-8).float() * 1e-8
    r_nn = r_vec / d_nn.unsqueeze(1)   # [E, 3] 单位方向向量

    return edge_index, d_nn, r_nn
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
    
    # 1. 计算 edge_index: [2, E]
    # target -> source (j -> i)，flow="source_to_target"
    # edge_index = knn_graph(pos, k=max_nn, batch=batch, loop=False)
    
    # # 2. 计算相对位移向量 R_nn
    # # i 是目标节点，j 是邻居节点
    # row, col = edge_index
    # # R_ij = p_j - p_i
    # R_vec = pos[col] - pos[row]
    
    # # 3. 计算欧氏距离 D_nn
    # # 使用 norm(dim=-1) 并保持维度以方便后续处理
    # D_nn = pt.norm(R_vec, p=2, dim=-1, keepdim=True)
    # R_nn = R_vec / D_nn
    # #print("edge_shape",edge_index.shape, D_nn.shape, R_nn.shape)
    # return edge_index, D_nn, R_nn
    #def build_edge_index_and_features(pos, max_nn, batch=None):
        # 1. 构图
    # knn_graph 返回 [j, i]，即 [neighbor, target]
    # edge_index = knn_graph(pos, k=max_nn, batch=batch, loop=False)
    
    # # 2. 计算向量 (注意方向需与你模型之前的逻辑一致)
    # row, col = edge_index
    # R_vec = pos[col] - pos[row]
    
    # # 3. 计算距离
    # D_nn = pt.norm(R_vec, p=2, dim=-1, keepdim=True)
    
    # # 4. 数值保护：不改变形状，只改变极小值
    # D_nn_stable = D_nn.clamp(min=1e-8)
    
    # # 5. 归一化方向向量
    # R_nn = R_vec / D_nn_stable
    min_threshold=1e-2
    # return edge_index, D_nn, R_nn # D_nn 依然可以用原始的，但 R_nn 必须用 stable 的
    edge_index = knn_graph(pos, k=max_nn, batch=batch, loop=False)
    
    # 2. 计算距离
    row, col = edge_index
    R_vec = pos[col] - pos[row]
    D_nn = pt.norm(R_vec, p=2, dim=-1, keepdim=True)
    
    # 3. 【核心修正】过滤掉距离小于阈值的边
    # 这能完美模拟你旧代码中“不选过近点”的逻辑
    mask = D_nn.view(-1) > min_threshold
    
    if not mask.all():
        edge_index = edge_index[:, mask]
        R_vec = R_vec[mask]
        D_nn = D_nn[mask]
        # print(f"Filtered {(~mask).sum()} edges because they were too close.")

    # 4. 归一化
    R_nn = R_vec / D_nn.clamp(min=1e-8)
    
    return edge_index, D_nn, R_nn

def compute_edge_index(ids_topk):
    """
    ids_topk: [N_atom+1, nn]，已经包含 sink row
    返回 edge_index: [2, E]，与 ids_topk 顺序完全对齐
    """
    N, nn = ids_topk.shape
    device = ids_topk.device

    # 每个节点 i 生成 nn 条边
    src = pt.arange(N, device=device).unsqueeze(1).repeat(1, nn)  # [N, nn]
    dst = ids_topk                                                   # [N, nn]

    # flatten
    src = src.flatten()
    dst = dst.flatten()

    edge_index = pt.stack([src, dst], dim=0)  # [2, N*nn]
    return edge_index
class StateUpdate(pt.nn.Module):
    def __init__(self, Ns, Nh, Nk):
        super(StateUpdate, self).__init__()
        # operation parameters
        self.Ns = Ns
        self.Nh = Nh
        self.Nk = Nk#number of kernels / channels

        # node query model
        self.nqm = pt.nn.Sequential(
            pt.nn.Linear(2*Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, 2*Nk*Nh),
        )

        # edges scalar keys model
        self.eqkm = pt.nn.Sequential(
            pt.nn.Linear(6*Ns+1, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Nk),
        )

        # edges vector keys model
        self.epkm = pt.nn.Sequential(
            pt.nn.Linear(6*Ns+1, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, 3*Nk),
        )

        # edges value model
        self.evm = pt.nn.Sequential(
            pt.nn.Linear(6*Ns+1, 2*Ns),
            pt.nn.ELU(),
            pt.nn.Linear(2*Ns, 2*Ns),
            pt.nn.ELU(),
            pt.nn.Linear(2*Ns, 2*Ns),
        )

        # scalar projection model
        self.qpm = pt.nn.Sequential(
            pt.nn.Linear(Nh*Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
        )

        # vector projection model
        self.ppm = pt.nn.Sequential(
            pt.nn.Linear(Nh*Ns, Ns, bias=False),
        )

        # scaling factor for attention
        self.sdk = pt.nn.Parameter(pt.sqrt(pt.tensor(Nk).float()), requires_grad=False)

    def forward(self, q, p, q_nn, p_nn, d_nn, r_nn):
        # q: [N, S]
        # p: [N, 3, S]
        # q_nn: [N, n, S]
        # p_nn: [N, n, 3, S]
        # d_nn: [N, n]
        # r_nn: [N, n, 3]
        # N: number of nodes
        # n: number of nearest neighbors
        # S: state dimensions
        # H: number of attention heads

        #q[N_atom+1,32
        #p[N_atom+1,3,32] zeros
        #ids_topk[N_atom+1,64]
        #D_topk[N_atom+1,64]
        #R_topk[N_atom+1,64,3]


        # get dimensions
        N, n, S = q_nn.shape#[N_atom+1,64,]

        # node inputs packing
        X_n = pt.cat([
            q,
            pt.norm(p, dim=1),
        ], dim=1)  # [N, 2*S]

        # edge inputs packing
        X_e = pt.cat([
            d_nn.unsqueeze(2),                                  # distance
            X_n.unsqueeze(1).repeat(1,n,1),                     # centered state
            q_nn,                                               # neighbors states
            pt.norm(p_nn, dim=2),                               # neighbors vector states norms
            pt.sum(p.unsqueeze(1) * r_nn.unsqueeze(3), dim=2),  # centered vector state projections
            pt.sum(p_nn * r_nn.unsqueeze(3), dim=2),            # neighbors vector states projections
        ], dim=2)  # [N, n, 6*S+1]

        # node queries
        Q = self.nqm.forward(X_n).view(N, 2, self.Nh, self.Nk)  # [N, 2*S] -> [N, 2, Nh, Nk]

        # scalar edges keys while keeping interaction order inveriance
        Kq = self.eqkm.forward(X_e).view(N, n, self.Nk).transpose(1,2)  # [N, n, 6*S+1] -> [N, Nk, n]

        # vector edges keys while keeping bond order inveriance
        Kp = pt.cat(pt.split(self.epkm.forward(X_e), self.Nk, dim=2), dim=1).transpose(1,2)# [N, Nk, n*3]

        # edges values while keeping interaction order inveriance
        V = self.evm.forward(X_e).view(N, n, 2, S).transpose(1,2)  # [N, n, 6*S+1] -> [N, 2, n, S]

        # vectorial inputs packing
        Vp = pt.cat([
            V[:,1].unsqueeze(2) * r_nn.unsqueeze(3),
            p.unsqueeze(1).repeat(1,n,1,1),
            p_nn,
            #pt.cross(p.unsqueeze(1).repeat(1,n,1,1), r_nn.unsqueeze(3).repeat(1,1,1,S), dim=2),
        ], dim=1).transpose(1,2)  # [N, 3, 3*n, S]

        # queries and keys collapse
        Mq = pt.nn.functional.softmax(pt.matmul(Q[:,0], Kq) / self.sdk, dim=2)  # [N, Nh, n]
        Mp = pt.nn.functional.softmax(pt.matmul(Q[:,1], Kp) / self.sdk, dim=2)  # [N, Nh, 3*n]

        # scalar state attention mask and values collapse
        Zq = pt.matmul(Mq, V[:,0]).view(N, self.Nh*self.Ns)  # [N, Nh*S]
        Zp = pt.matmul(Mp.unsqueeze(1), Vp).view(N, 3, self.Nh*self.Ns)  # [N, 3, Nh*S]

        # decode outputs
        qh = self.qpm.forward(Zq)
        ph = self.ppm.forward(Zp)

        # update state with residual
        qz = q + qh
        pz = p + ph

        return qz, pz


def state_max_pool(q, p, M):
    # get norm of state vector
    s = pt.norm(p, dim=2)  # [N, S]

    # perform mask pool on mask
    q_max, _ = pt.max(M.unsqueeze(2) * q.unsqueeze(1), dim=0)  # [n, S]
    _, s_ids = pt.max(M.unsqueeze(2) * s.unsqueeze(1), dim=0)  # [n, S]

    # get maximum state vector
    p_max = pt.gather(p, 0, s_ids.unsqueeze(2).repeat((1,1,p.shape[2])))

    return q_max, p_max


class StatePoolLayer(pt.nn.Module):
    def __init__(self, N0, N1, Nh):
        super(StatePoolLayer, self).__init__()
        # state attention model
        self.sam = pt.nn.Sequential(
            pt.nn.Linear(2*N0, N0),
            pt.nn.ELU(),
            pt.nn.Linear(N0, N0),
            pt.nn.ELU(),
            pt.nn.Linear(N0, 2*Nh),
        )

        # attention heads decoding
        self.zdm = pt.nn.Sequential(
            pt.nn.Linear(Nh * N0, N0),
            pt.nn.ELU(),
            pt.nn.Linear(N0, N0),
            pt.nn.ELU(),
            pt.nn.Linear(N0, N1),
        )

        # vector attention heads decoding
        self.zdm_vec = pt.nn.Sequential(
            pt.nn.Linear(Nh * N0, N1, bias=False)
        )

    def forward(self, q, p, M):
        # create filter for softmax
        F = (1.0 - M + 1e-6) / (M - 1e-6)

        # pack features
        z = pt.cat([q, pt.norm(p, dim=1)], dim=1)

        # multiple attention pool on state
        Ms = pt.nn.functional.softmax(self.sam.forward(z).unsqueeze(1) + F.unsqueeze(2), dim=0).view(M.shape[0], M.shape[1], -1, 2)
        qh = pt.matmul(pt.transpose(q,0,1), pt.transpose(Ms[:,:,:,0],0,1))
        ph = pt.matmul(pt.transpose(pt.transpose(p,0,2),0,1), pt.transpose(Ms[:,:,:,1],0,1).unsqueeze(1))

        # attention heads decoding
        qr = self.zdm.forward(qh.view(Ms.shape[1], -1))
        pr = self.zdm_vec.forward(ph.view(Ms.shape[1], p.shape[1], -1))

        return qr, pr


# >>> LAYERS
class StateUpdateLayer(pt.nn.Module):
    def __init__(self, layer_params):
        super(StateUpdateLayer, self).__init__()
        # define operation
        self.su = StateUpdate(*[layer_params[k] for k in ['Ns', 'Nh', 'Nk']])
        # store number of nearest neighbors
        self.m_nn = pt.nn.Parameter(pt.arange(layer_params['nn'], dtype=pt.int64), requires_grad=False)
        #[0,1,2...7]逐渐增加，从8到64
    def forward(self, Z):
        # unpack input
        q, p, ids_topk, D_topk, R_topk = Z
#q[N_atom+1,32
#p[N_atom+1,3,32] zeros
#ids_topk[N_atom+1,64]
#D_topk[N_atom+1,64]
#R_topk[N_atom+1,64,3]

        # update q, p
        ids_nn = ids_topk[:,self.m_nn]#ids_topk[N_atom+1,nn]
        # q, p = self.su.forward(q, p, q[ids_nn], p[ids_nn], D_topk[:,self.m_nn], R_topk[:,self.m_nn])

        # with checkpoint
        q = q.requires_grad_()
        p = p.requires_grad_()
        q, p = checkpoint(self.su.forward, q, p, q[ids_nn], p[ids_nn], D_topk[:,self.m_nn], R_topk[:,self.m_nn])
        #q, p = self.su.forward(q, p, q[ids_nn], p[ids_nn], D_topk[:,self.m_nn], R_topk[:,self.m_nn])
        # sink
        q[0] = q[0] * 0.0
        p[0] = p[0] * 0.0

        return q, p, ids_topk, D_topk, R_topk


class CrossStateUpdateLayer(pt.nn.Module):
    def __init__(self, layer_params):
        super(CrossStateUpdateLayer, self).__init__()
        # get cross-states update layer parameters
        Ns = layer_params['Ns']
        self.Nh = layer_params['cNh']
        self.Nk = layer_params['cNk']

        # atomic level state update layers
        self.sul = StateUpdateLayer(layer_params)

        # queries model
        self.cqm = pt.nn.Sequential(
            pt.nn.Linear(2*Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, self.Nk*self.Nh),
        )

        # keys model
        self.ckm = pt.nn.Sequential(
            pt.nn.Linear(2*Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, self.Nk),
        )

        # values model
        self.cvm = pt.nn.Sequential(
            pt.nn.Linear(2*Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
        )

        # projection heads
        self.cpm = pt.nn.Sequential(
            pt.nn.Linear((self.Nh+1)*Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
        )

        # scaling factor for attention
        self.sdk = pt.nn.Parameter(pt.sqrt(pt.tensor(self.Nk).float()), requires_grad=False)

    def forward(self, Z):
        # unpack input
        q0, p0, ids0_topk, D0_nn, R0_nn = Z[0]
        q1, p1, ids1_topk, D1_nn, R1_nn = Z[1]

        # forward independently
        qa0, pz0, _, _, _ = self.sul.forward((q0, p0, ids0_topk, D0_nn, R0_nn))
        qa1, pz1, _, _, _ = self.sul.forward((q1, p1, ids1_topk, D1_nn, R1_nn))

        # pack states
        s0 = pt.cat([qa0, pt.norm(pz0, dim=1)], dim=1)
        s1 = pt.cat([qa1, pt.norm(pz1, dim=1)], dim=1)

        # compute queries
        Q0 = self.cqm.forward(s0).reshape(s0.shape[0], self.Nh, self.Nk)
        Q1 = self.cqm.forward(s1).reshape(s1.shape[0], self.Nh, self.Nk)

        # compute keys
        K0 = self.ckm.forward(s0).transpose(0,1)
        K1 = self.ckm.forward(s1).transpose(0,1)

        # compute values
        V0 = self.cvm.forward(s0)
        V1 = self.cvm.forward(s1)

        # transform 1 -> 0
        M10 = pt.nn.functional.softmax(pt.matmul(Q0,K1 / self.sdk), dim=2)
        qh0 = pt.matmul(M10, V1).view(Q0.shape[0], -1)

        # transform 0 -> 1
        M01 = pt.nn.functional.softmax(pt.matmul(Q1,K0 / self.sdk), dim=2)
        qh1 = pt.matmul(M01, V0).view(Q1.shape[0], -1)

        # projections and residual
        #qz0 = qa0 + self.cpm.forward(qh0)
        #qz1 = qa1 + self.cpm.forward(qh1)
        qz0 = self.cpm.forward(pt.cat([qa0,qh0], dim=1))
        qz1 = self.cpm.forward(pt.cat([qa1,qh1], dim=1))

        return ((qz0, pz0, ids0_topk, D0_nn, R0_nn), (qz1, pz1, ids1_topk, D1_nn, R1_nn))
