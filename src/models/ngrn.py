# -*- coding: utf-8 -*-
"""
邻居感知的群组推荐网络 (NGRN)。
- 群组子图加权 GCN + 均值池化 -> 局部表示 h_c
- 群组外邻居注意力融合 -> h_c^nbr
- [h_c || h_c^nbr] -> W3 -> 全局表示 h_tilde_c
- 打分 q(i,c) = v_i^T h_tilde_c
- 训练: L_rank + λ L_tag；采用反向传播(BP)更新参数，训练阶段引入 Dropout 正则化以防过拟合、提高泛化能力。
"""

import numpy as np
import torch
import torch.nn as torch_nn
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path


def resolve_sparse_group_adj_for_train(data_dir, sparse_flag: bool, dense_flag: bool) -> bool:
    """
    是否对群内 GCN 使用稀疏归一化邻接。
    默认：data_dir 末级目录名为 ml10m 时为 True（避免超大群 T×T 稠密矩阵 OOM），否则 False（稠密通常更快）。
    """
    if sparse_flag and dense_flag:
        raise ValueError("不能同时指定 --sparse_group_adj 与 --dense_group_adj")
    if sparse_flag:
        return True
    if dense_flag:
        return False
    return Path(str(data_dir)).resolve().name.lower() == "ml10m"


def _to_tensor(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    return torch.tensor(x, device=device)


def build_group_subgraph_data(data, device="cpu"):
    """
    从 build_ngrn_data 的返回中整理每个群组的子图与邻居信息，便于模型前向。
    返回:
      group_nodes: list of list of user_id (每个群组的成员)
      group_edges: list of list of (u_idx_in_Vc, v_idx_in_Vc, weight)，u_idx 为在 V_c 中的局部下标
      group_neighbors: list of (neighbor_global_ids, s_c_u_weights)
      node_to_local: 每个群组内 global_id -> local_idx 的映射 list
    """
    adj = data["adj"]
    edge_weight = data["edge_weight"]
    candidates = data["candidates"]

    group_nodes = []
    group_edges = []
    group_neighbors = []
    node_to_local_list = []

    for V_c in candidates:
        local_idx = {u: i for i, u in enumerate(V_c)}
        V_c_set = set(V_c)
        # 群组内边（仅保留两端都在 V_c 的），权从 edge_weight 取；无向边只存一次或按需
        E_c = []
        for u in V_c:
            for v, w in adj.get(u, []):
                if v in V_c_set:
                    i, j = local_idx[u], local_idx[v]
                    if i <= j:
                        E_c.append((i, j, w))
        # 外部邻居 N(c) 及 s(c,u)
        nbr_set = set()
        for v in V_c:
            for u, w in adj.get(v, []):
                if u not in V_c_set:
                    nbr_set.add(u)
        s_c_u = []
        nbr_list = sorted(nbr_set)
        for u in nbr_list:
            total = 0
            count = 0
            for v in V_c:
                w = edge_weight.get((u, v)) or edge_weight.get((v, u))
                if w is not None:
                    total += w
                    count += 1
            s_c_u.append(total / len(V_c) if V_c else 0)
        group_nodes.append(V_c)
        group_edges.append(E_c)
        group_neighbors.append((nbr_list, s_c_u))
        node_to_local_list.append(local_idx)

    return {
        "group_nodes": group_nodes,
        "group_edges": group_edges,
        "group_neighbors": group_neighbors,
        "node_to_local_list": node_to_local_list,
    }


class WeightedGCNLayer(torch_nn.Module):
    """单层加权 GCN: H' = σ(D^{-1/2} A_tilde D^{-1/2} H Θ)。"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = torch_nn.Linear(in_dim, out_dim)

    def forward(self, H, A_norm):
        # H: (T, in_dim), A_norm: (T, T) dense 或 sparse COO 已归一化
        if getattr(A_norm, "is_sparse", False):
            H = torch.sparse.mm(A_norm, H)
        else:
            H = A_norm @ H
        return F.relu(self.linear(H))


class NeighborFusion(torch_nn.Module):
    """邻居感知融合: z = p^T tanh(W1 h_c + W2 u + b), alpha = softmax(z + γ*s), h_c^nbr = sum alpha * u."""

    def __init__(self, d, use_gamma_bias=True):
        super().__init__()
        self.d = d
        self.W1 = torch_nn.Linear(d, d)
        self.W2 = torch_nn.Linear(d, d)
        self.b = torch_nn.Parameter(torch.zeros(d))
        self.p = torch_nn.Parameter(torch.randn(d) * 0.01)
        self.gamma = torch_nn.Parameter(torch.tensor(1.0)) if use_gamma_bias else 1.0

    def forward(self, h_c, neighbor_embs, s_c_u):
        """
        h_c: (d,), neighbor_embs: (N_nbr, d), s_c_u: (N_nbr,)
        返回 h_c_nbr: (d,)
        """
        if neighbor_embs is None or neighbor_embs.shape[0] == 0:
            return torch.zeros_like(h_c)
        # z_{c,u} = p^T tanh(W1 h_c + W2 u + b)
        h_c_b = h_c.unsqueeze(0)
        u_b = neighbor_embs
        z = torch.tanh(self.W1(h_c_b) + self.W2(u_b) + self.b)
        z = z @ self.p
        gamma = self.gamma if isinstance(self.gamma, torch.Tensor) else torch.tensor(self.gamma, device=z.device)
        s = _to_tensor(s_c_u, z.device).float()
        logits = z.squeeze(-1) + gamma * s
        alpha = F.softmax(logits, dim=0)
        h_c_nbr = (alpha.unsqueeze(1) * neighbor_embs).sum(0)
        return h_c_nbr


class NGRN(torch_nn.Module):
    """
    Neighborhood-aware Group Recommendation Network.
    输入: 数据字典 + 预计算的 group_subgraph_data。
    前向: 对每个群组计算 h_tilde_c；对 (item_id, group_idx) 计算 q(i,c)。
    训练时采用 BP 更新参数，并在 GCN 输出与融合表示上使用 Dropout 正则化。
    """

    def __init__(
        self,
        n_users,
        n_items,
        n_tags,
        d=64,
        gcn_layers=2,
        dropout=0.1,
        max_neighbors=500,
        device="cpu",
        edge_weight_mode="learned",
        use_neighbor_fusion=True,
        use_gcn=True,
        use_skip_connection=True,
        use_sparse_group_adj=False,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_tags = n_tags
        self.d = d
        self.gcn_layers = gcn_layers
        self.max_neighbors = max_neighbors
        self.device = device
        # 消融: learned=σ(α·w_jaccard+β); jaccard=原始重叠权; uniform=群内边权恒为1
        self.edge_weight_mode = edge_weight_mode
        self.use_neighbor_fusion = use_neighbor_fusion
        self.use_gcn = use_gcn
        self.use_skip_connection = use_skip_connection
        # 大群组 T×T 稠密邻接易 OOM（如 ml10m）；稀疏版用 torch.sparse.mm，小数据集用稠密更快
        self.use_sparse_group_adj = use_sparse_group_adj

        self.user_emb = torch_nn.Embedding(n_users, d)
        self.item_emb = torch_nn.Embedding(n_items, d)
        torch_nn.init.xavier_uniform_(self.user_emb.weight)
        torch_nn.init.xavier_uniform_(self.item_emb.weight)

        self.gcn = torch_nn.ModuleList([
            WeightedGCNLayer(d, d) for _ in range(gcn_layers)
        ])
        self.neighbor_fusion = NeighborFusion(d)
        self.W3 = torch_nn.Linear(2 * d, d)
        # 成员均值 skip：h_tilde_c 含 W_skip(mean(X_c))，图噪声大时仍可依赖直接成员信息，便于超越纯嵌入基线
        self.W_skip = torch_nn.Linear(d, d)
        torch_nn.init.xavier_uniform_(self.W_skip.weight, gain=0.5)
        torch_nn.init.zeros_(self.W_skip.bias)
        self.skip_scale = torch_nn.Parameter(torch.tensor(0.3))

        # 标签辅助头: z_c = W_r h_tilde_c + b_r, 输出 n_tags 维
        self.W_r = torch_nn.Linear(d, n_tags)
        self.b_r = torch_nn.Parameter(torch.zeros(n_tags))

        self.dropout = torch_nn.Dropout(dropout)

        # Learnable re-weighting of social edge strengths (on top of Jaccard):
        # weight(u,v) = sigmoid(alpha * w_jaccard + beta)
        self.edge_alpha = torch_nn.Parameter(torch.tensor(1.0))
        self.edge_beta = torch_nn.Parameter(torch.tensor(0.0))

    def _norm_adj(self, V_c, E_c):
        """从 V_c 和 E_c (local_idx, local_idx, w_jaccard) 构建对称 D^{-1/2} A_tilde D^{-1/2}。

        edge_weight_mode:
          learned: sigmoid(alpha * w_jaccard + beta)
          jaccard: 使用数据侧 Jaccard 权 w
          cosine: 使用数据侧 Cosine 权 w
          uniform: 有边则权为 1
        """
        T = len(V_c)
        if T == 0:
            if self.use_sparse_group_adj:
                return torch.sparse_coo_tensor(
                    torch.zeros((2, 0), dtype=torch.long, device=self.device),
                    torch.zeros((0,), dtype=torch.float32, device=self.device),
                    (0, 0),
                    device=self.device,
                )
            return torch.zeros(0, 0, device=self.device)

        if not self.use_sparse_group_adj:
            A = torch.zeros((T, T), device=self.device)
            for i, j, w in E_c:
                base = torch.tensor(float(w), device=self.device)
                if self.edge_weight_mode == "uniform":
                    w_ij = torch.tensor(1.0, device=self.device)
                elif self.edge_weight_mode in ("jaccard", "cosine"):
                    w_ij = base
                else:
                    w_ij = torch.sigmoid(self.edge_alpha * base + self.edge_beta)
                if i == j:
                    A[i, j] = w_ij
                else:
                    A[i, j] = w_ij
                    A[j, i] = w_ij
            A = A + torch.eye(T, device=self.device, dtype=A.dtype)
            D = A.sum(dim=1)
            d_inv_sqrt = torch.pow(D + 1e-8, -0.5)
            return d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)

        rows, cols, vals = [], [], []
        for i, j, w in E_c:
            base = torch.tensor(float(w), device=self.device)
            if self.edge_weight_mode == "uniform":
                w_ij = torch.tensor(1.0, device=self.device)
            elif self.edge_weight_mode in ("jaccard", "cosine"):
                w_ij = base
            else:
                w_ij = torch.sigmoid(self.edge_alpha * base + self.edge_beta)
            if i == j:
                rows.append(i)
                cols.append(j)
                vals.append(w_ij)
            else:
                rows.extend([i, j])
                cols.extend([j, i])
                vals.extend([w_ij, w_ij])

        one = torch.tensor(1.0, device=self.device)
        for i in range(T):
            rows.append(i)
            cols.append(i)
            vals.append(one)

        row_idx = torch.tensor(rows, dtype=torch.long, device=self.device)
        col_idx = torch.tensor(cols, dtype=torch.long, device=self.device)
        val = torch.stack(vals).float()

        deg = torch.zeros(T, device=self.device)
        deg.index_add_(0, row_idx, val)
        d_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        norm_val = val * d_inv_sqrt[row_idx] * d_inv_sqrt[col_idx]

        idx = torch.stack([row_idx, col_idx], dim=0)
        return torch.sparse_coo_tensor(idx, norm_val, (T, T), device=self.device).coalesce()

    def _group_representation(self, group_idx, subgraph_data):
        """
        计算单个群组的全局表示 h_tilde_c。
        group_nodes[group_idx], group_edges[group_idx], group_neighbors[group_idx].
        """
        V_c = subgraph_data["group_nodes"][group_idx]
        E_c = subgraph_data["group_edges"][group_idx]
        nbr_list, s_c_u = subgraph_data["group_neighbors"][group_idx]

        T = len(V_c)
        node_ids = _to_tensor(np.array(V_c), self.device).long()
        X_c = self.user_emb(node_ids)

        A_norm = self._norm_adj(V_c, E_c)

        if self.use_gcn and self.gcn_layers > 0:
            H = X_c
            for layer in self.gcn:
                H = layer(H, A_norm)
                H = self.dropout(H)
            h_c = H.mean(dim=0)
        else:
            h_c = X_c.mean(dim=0)

        if self.use_neighbor_fusion and nbr_list:
            # 仅保留在 user_emb 范围内的邻居，避免 IndexError（数据中边可能含评分外用户）
            n_users = self.user_emb.num_embeddings
            valid = [(u, s) for u, s in zip(nbr_list, s_c_u) if 0 <= u < n_users]
            nbr_list = [u for u, _ in valid]
            s_c_u = [s for _, s in valid]
            if nbr_list:
                s_arr = np.array(s_c_u, dtype=np.float32)
                if len(nbr_list) > self.max_neighbors:
                    top_idx = np.argsort(s_arr)[-self.max_neighbors:]
                    nbr_list = [nbr_list[i] for i in top_idx]
                    s_c_u = s_arr[top_idx].tolist()
                nbr_ids = _to_tensor(np.array(nbr_list), self.device).long()
                neighbor_embs = self.user_emb(nbr_ids)
                s_c_u = np.array(s_c_u, dtype=np.float32)
                h_c_nbr = self.neighbor_fusion(h_c, neighbor_embs, s_c_u)
            else:
                h_c_nbr = torch.zeros_like(h_c)
        else:
            h_c_nbr = torch.zeros_like(h_c)

        h_cat = torch.cat([h_c, h_c_nbr], dim=-1)
        h_cat = self.dropout(h_cat)
        h_struct = self.W3(h_cat)
        mean_member = X_c.mean(dim=0)
        if self.use_skip_connection:
            h_tilde_c = h_struct + torch.sigmoid(self.skip_scale) * self.W_skip(mean_member)
        else:
            h_tilde_c = h_struct
        return h_tilde_c

    def forward_group_representations(self, subgraph_data, group_indices=None):
        """计算一批群组的 h_tilde_c。group_indices 为 None 时计算全部群组。"""
        if group_indices is None:
            group_indices = range(len(subgraph_data["group_nodes"]))
        out = []
        for c_idx in group_indices:
            h = self._group_representation(c_idx, subgraph_data)
            out.append(h)
        return torch.stack(out, dim=0)

    def score_item_group(self, item_id, h_tilde_c):
        """q(i,c) = v_i^T h_tilde_c。item_id 为标量或 1-d tensor。"""
        v_i = self.item_emb(item_id)
        if v_i.dim() == 1:
            v_i = v_i.unsqueeze(0)
        return (v_i * h_tilde_c).sum(dim=-1)

    def forward_scores(self, item_ids, group_indices, subgraph_data):
        """对 (item_ids, group_indices) 计算 q(i,c)。"""
        h_all = self.forward_group_representations(subgraph_data, group_indices)
        item_ids = _to_tensor(item_ids, self.device).long()
        v = self.item_emb(item_ids)
        if v.dim() == 1:
            v = v.unsqueeze(0)
        # h_all: (n_groups_selected, d), v: (batch,) or (batch, d)
        scores = torch.matmul(v, h_all.t())
        return scores

    def predict_group_labels(self, h_tilde_c):
        """z_c = W_r h_tilde_c + b_r, hat_y_c = σ(z_c)。"""
        return torch.sigmoid(self.W_r(h_tilde_c) + self.b_r)

    def recommend_groups_for_item(self, item_idx, subgraph_data, top_k=10, group_indices=None):
        """
        给定项目 item_idx（0-based），对候选群组按 q(i,c) 排序，返回 Top-K 群组索引。
        """
        if group_indices is None:
            group_indices = list(range(len(subgraph_data["group_nodes"])))
        h_all = self.forward_group_representations(subgraph_data, group_indices)
        q = self.score_item_group(torch.tensor([item_idx], device=self.device), h_all).squeeze()
        _, idx = torch.topk(q, min(top_k, q.shape[0]))
        return idx.cpu().tolist()


def sample_rank_triples(y_ic, n_samples, rng):
    """从 y_ic 中采样 (i, c+, c-)：对同一 item i，c+ 相关(y=1)，c- 不相关(y=0)，且 c+ != c-。负样本必须针对同一 i。"""
    # 每个 item 的正群组、负群组列表，便于按 item 采样困难负样本
    item_to_pos = defaultdict(list)
    item_to_neg = defaultdict(list)
    for (i, c), y in y_ic.items():
        if y == 1:
            item_to_pos[i].append(c)
        else:
            item_to_neg[i].append(c)
    pos_pairs = [(i, c) for i, cs in item_to_pos.items() for c in cs]
    if not pos_pairs:
        return []
    triples = []
    for _ in range(n_samples):
        i, c_plus = pos_pairs[rng.integers(0, len(pos_pairs))]
        negs = item_to_neg.get(i)
        if not negs:
            continue
        c_minus = negs[rng.integers(0, len(negs))]
        if c_plus == c_minus:
            continue
        triples.append((i, c_plus, c_minus))
    return triples


def sample_rank_triples_group_to_items(y_ic, n_samples, rng):
    """群体决策：对群组推荐项目。采样 (c, i+, i-)：对同一群组 c，i+ 相关(y=1)，i- 不相关(y=0)。"""
    group_to_pos = defaultdict(list)
    group_to_neg = defaultdict(list)
    for (i, c), y in y_ic.items():
        if y == 1:
            group_to_pos[c].append(i)
        else:
            group_to_neg[c].append(i)
    pos_pairs = [(c, i) for c, is_ in group_to_pos.items() for i in is_]
    if not pos_pairs:
        return []
    triples = []
    for _ in range(n_samples):
        c, i_plus = pos_pairs[rng.integers(0, len(pos_pairs))]
        negs = group_to_neg.get(c)
        if not negs:
            continue
        i_minus = negs[rng.integers(0, len(negs))]
        if i_plus == i_minus:
            continue
        triples.append((c, i_plus, i_minus))
    return triples


def compute_rank_loss_group_to_items(model, subgraph_data, sample_triples, device):
    """群体决策：L_rank = - mean log σ(q(c,i+)-q(c,i-))，q(c,i)=v_i^T h_c。"""
    if not sample_triples:
        return torch.tensor(0.0, device=device)
    groups_needed = set()
    items_needed = set()
    for c, i_plus, i_minus in sample_triples:
        groups_needed.add(c)
        items_needed.add(i_plus)
        items_needed.add(i_minus)
    group_list = sorted(groups_needed)
    h_dict = {}
    if group_list:
        h_batch = model.forward_group_representations(subgraph_data, group_list)
        h_dict = {c: h_batch[idx] for idx, c in enumerate(group_list)}
    item_list = sorted(items_needed)
    v_dict = {}
    if item_list:
        v_batch = model.item_emb(torch.tensor(item_list, device=device))
        v_dict = {i: v_batch[idx] for idx, i in enumerate(item_list)}
    losses = []
    for c, i_plus, i_minus in sample_triples:
        h_c = h_dict[c]
        v_plus = v_dict[i_plus]
        v_minus = v_dict[i_minus]
        q_plus = (v_plus * h_c).sum()
        q_minus = (v_minus * h_c).sum()
        losses.append(-F.logsigmoid(q_plus - q_minus))
    return torch.stack(losses).mean()


def compute_rank_loss(model, subgraph_data, sample_triples, device):
    """L_rank = - mean log σ(q(i,c+)-q(i,c-))。批量计算以加速。"""
    if not sample_triples:
        return torch.tensor(0.0, device=device)
    # 收集需要计算的群组索引
    groups_needed = set()
    items_needed = set()
    for i, c_plus, c_minus in sample_triples:
        groups_needed.add(c_plus)
        groups_needed.add(c_minus)
        items_needed.add(i)
    # 批量计算所有需要的群组表示
    group_list = sorted(groups_needed)
    h_dict = {}
    if group_list:
        h_batch = model.forward_group_representations(subgraph_data, group_list)
        h_dict = {c: h_batch[idx] for idx, c in enumerate(group_list)}
    # 批量计算所有需要的项目嵌入
    item_list = sorted(items_needed)
    v_dict = {}
    if item_list:
        v_batch = model.item_emb(torch.tensor(item_list, device=device))
        v_dict = {i: v_batch[idx] for idx, i in enumerate(item_list)}
    # 计算每个三元组的损失
    losses = []
    for i, c_plus, c_minus in sample_triples:
        h_plus = h_dict[c_plus]
        h_minus = h_dict[c_minus]
        v_i = v_dict[i]
        q_plus = (v_i * h_plus).sum()
        q_minus = (v_i * h_minus).sum()
        losses.append(-F.logsigmoid(q_plus - q_minus))
    return torch.stack(losses).mean()


def compute_tag_loss(model, group_r_tensor, subgraph_data, group_indices, device):
    """L_tag = BCE(hat_y_c, r_c)。"""
    h_all = model.forward_group_representations(subgraph_data, group_indices)
    hat_y = model.predict_group_labels(h_all)
    target = group_r_tensor.to(device)
    return F.binary_cross_entropy(hat_y, target)
