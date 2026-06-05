# -*- coding: utf-8 -*-
"""
NGRN 数据预处理与候选群组生成。
- 偏好信号驱动的边重加权（Jaccard on L(u)）
- k-核候选群组生成
- 群组标签 R(c) 与多热向量 r_c
"""

import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None


def _load_ratings(ratings_path):
    """加载评分：(user_id, item_id, rating) 列表，user/item 使用 0-based 连续 ID。"""
    rows = []
    users, items = set(), set()
    with open(ratings_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u, i, r = int(row["user_id"]), int(row["movie_id"]), float(row["rating"])
            rows.append((u, i, r))
            users.add(u)
            items.add(i)
    return rows, sorted(users), sorted(items)


def _load_edges(edges_path):
    """加载社交边：(u, v) 列表。"""
    edges = []
    with open(edges_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u, v = int(row["user_id"]), int(row["neighbor_id"])
            edges.append((u, v))
    return edges


def _load_item_labels(movies_path):
    """加载项目标签 L(i)：movie_id -> set of genre tags。"""
    item_labels = {}
    with open(movies_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = int(row["movie_id"])
            genres = row.get("genres", "").strip()
            if genres and genres != "(no genres listed)":
                item_labels[mid] = set(g for g in genres.split("|") if g.strip())
            else:
                item_labels[mid] = set()
    return item_labels


def _user_label_sets(ratings, item_labels):
    """L(u) = union of L(i) for i in I(u)。返回 user_id -> set of labels。"""
    user_items = defaultdict(set)
    for u, i, _ in ratings:
        user_items[u].add(i)
    user_labels = {}
    for u, items_u in user_items.items():
        L_u = set()
        for i in items_u:
            L_u |= item_labels.get(i, set())
        user_labels[u] = L_u
    return user_labels


def _user_item_sets(ratings):
    """返回 user_id -> set(item_id)，用于 cosine 相似度边权。"""
    user_items = defaultdict(set)
    for u, i, _ in ratings:
        user_items[u].add(i)
    return user_items


def edge_weight_jaccard(u, v, user_labels):
    """weight(u,v) = |L(u)∩L(v)| / |L(u)∪L(v)|。"""
    Lu = user_labels.get(u, set())
    Lv = user_labels.get(v, set())
    if not Lu and not Lv:
        return 1.0
    if not Lu or not Lv:
        return 0.0
    inter = len(Lu & Lv)
    union = len(Lu | Lv)
    return inter / union if union else 0.0


def edge_weight_cosine(u, v, user_items):
    """weight(u,v)=|I(u)∩I(v)| / sqrt(|I(u)|·|I(v)|)。"""
    Iu = user_items.get(u, set())
    Iv = user_items.get(v, set())
    if not Iu or not Iv:
        return 0.0
    inter = len(Iu & Iv)
    den = float(np.sqrt(len(Iu) * len(Iv)))
    return (inter / den) if den > 0 else 0.0


def build_weighted_graph(edges, user_labels, user_items, base_weight_mode="jaccard"):
    """构建加权图：邻接表 + 边权。返回 (adj: u->[(v,w)], weight_dict: (u,v)->w)。"""
    adj = defaultdict(list)
    weight_dict = {}
    for u, v in edges:
        if base_weight_mode == "cosine":
            w = edge_weight_cosine(u, v, user_items)
        else:
            w = edge_weight_jaccard(u, v, user_labels)
        adj[u].append((v, w))
        weight_dict[(u, v)] = w
    return dict(adj), weight_dict


def k_core_components(edges, k):
    """
    在无向图上计算 k-核，并返回每个连通分量作为候选群组。
    返回 list of list of node_id，每个内层 list 为一个候选群组的成员 ID。
    """
    if nx is None:
        raise RuntimeError("需要安装 networkx: pip install networkx")
    G = nx.Graph()
    for u, v in edges:
        G.add_edge(u, v)
    try:
        Gk = nx.k_core(G, k=k)
    except Exception:
        Gk = G
    comps = list(nx.connected_components(Gk))
    return [sorted(c) for c in comps if len(c) >= 2]


def k_core_candidates_filtered(edges, k, min_group_size=2, max_group_size=200):
    """
    与 build_ngrn_data 中 k-core 分支一致：先按规模过滤；若过滤后为空则回退为 k_core_components 原始输出。
    """
    candidates_raw = k_core_components(edges, k)
    candidates = [c for c in candidates_raw if min_group_size <= len(c) <= max_group_size]
    if not candidates:
        candidates = candidates_raw
    return [sorted(c) for c in candidates]


def sample_kcore_subgroups(
    edges,
    k,
    min_group_size=2,
    max_group_size=200,
    max_groups=0,
    seed=42,
):
    """
    在 k-core 图中采样多个“子群组”，并保证每个子群组内仍满足 k-core 约束（最小度 >= k）。
    用于 k-core 连通分量过少（如仅 1 个大分量）时补充候选群组。
    """
    if max_groups <= 0:
        return []
    if nx is None:
        raise RuntimeError("需要安装 networkx: pip install networkx")

    G = nx.Graph()
    for u, v in edges:
        G.add_edge(u, v)
    try:
        Gk = nx.k_core(G, k=k)
    except Exception:
        Gk = G
    if Gk.number_of_nodes() == 0:
        return []

    rng = np.random.default_rng(seed)
    nodes = list(Gk.nodes())
    degrees = dict(Gk.degree())
    # 先按度排序，再用随机打散同度节点，兼顾稳定性与多样性
    rng.shuffle(nodes)
    nodes.sort(key=lambda u: (-degrees.get(u, 0), u))

    sampled = []
    seen = set()
    for center in nodes:
        if len(sampled) >= max_groups:
            break
        # 优先在 1-hop 邻域内找局部群组，不够大再扩展到 2-hop
        n1 = set(Gk.neighbors(center))
        local_nodes = n1 | {center}
        if len(local_nodes) < min_group_size:
            n2 = set()
            for v in local_nodes:
                n2.update(Gk.neighbors(v))
            local_nodes |= n2
        if len(local_nodes) < min_group_size:
            continue

        H = Gk.subgraph(local_nodes).copy()
        try:
            Hk = nx.k_core(H, k=k)
        except Exception:
            Hk = H
        if Hk.number_of_nodes() < min_group_size:
            continue

        for comp in nx.connected_components(Hk):
            if len(sampled) >= max_groups:
                break
            g = sorted(comp)
            if len(g) < min_group_size:
                continue
            if len(g) > max_group_size:
                # 过大时取中心邻域内高连接子集，再做一次 k-core 保证约束
                g_sorted = sorted(g, key=lambda u: (-degrees.get(u, 0), u))
                g = sorted(g_sorted[:max_group_size])
                H2 = Gk.subgraph(g).copy()
                try:
                    H2k = nx.k_core(H2, k=k)
                except Exception:
                    H2k = H2
                g = sorted(H2k.nodes())
                if len(g) < min_group_size or len(g) > max_group_size:
                    continue

            key = tuple(g)
            if key in seen:
                continue
            seen.add(key)
            sampled.append(g)

    return sampled


def group_label_multihot(candidates, user_labels, tag_vocab):
    """
    对每个候选群组 c 计算 R(c) 并编码为多热向量 r_c。
    Count_c(t) = sum_{u in V_c} I(t in L(u))
    M = ceil( (1/|V_c|) * sum_{u in V_c} |L(u)| )
    top-M 标签作为 R(c)，编码为 r_c (len(tag_vocab))。
    返回: list of np.ndarray (multi-hot), tag_list (list of tag names).
    """
    tag_list = list(tag_vocab) if not isinstance(tag_vocab, list) else tag_vocab
    tag2idx = {t: i for i, t in enumerate(tag_list)}
    r_c_list = []
    for V_c in candidates:
        count_t = defaultdict(int)
        total_labels = 0
        for u in V_c:
            Lu = user_labels.get(u, set())
            total_labels += len(Lu)
            for t in Lu:
                if t in tag2idx:
                    count_t[t] += 1
        M = max(1, int(np.ceil(total_labels / len(V_c))))
        top_tags = sorted(count_t.keys(), key=lambda t: -count_t[t])[:M]
        r_c = np.zeros(len(tag_list), dtype=np.float32)
        for t in top_tags:
            r_c[tag2idx[t]] = 1.0
        r_c_list.append(r_c)
    return r_c_list, tag_list


def _load_candidates_from_file(candidates_path, user_ids_set, min_group_size, max_group_size):
    """
    从文件加载候选群组：
      - JSON: list[list[user_id]] 或 {"candidates": list[list[user_id]]}
      - CSV : 两列 user_id,group_id
    与 k-core 分支一致：先按 [min,max] 规模过滤；若过滤后为空则回退为「仅满足最小规模」的原始列表。
    """
    path = Path(candidates_path)
    if not path.is_file():
        raise FileNotFoundError(f"候选群组文件不存在: {path}")

    raw_groups = []
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            raw = obj.get("candidates", [])
        else:
            raw = obj
        for g in raw:
            if not isinstance(g, list):
                continue
            members = sorted({int(u) for u in g if int(u) in user_ids_set})
            if len(members) >= min_group_size:
                raw_groups.append(members)
    else:
        gid_to_users = defaultdict(set)
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    u = int(row["user_id"])
                    gid = int(row["group_id"])
                except (KeyError, ValueError):
                    continue
                if u in user_ids_set:
                    gid_to_users[gid].add(u)
        for gu in gid_to_users.values():
            members = sorted(gu)
            if len(members) >= min_group_size:
                raw_groups.append(members)

    dedup = []
    seen = set()
    for g in raw_groups:
        key = tuple(g)
        if key not in seen:
            seen.add(key)
            dedup.append(g)

    filtered = [g for g in dedup if min_group_size <= len(g) <= max_group_size]
    if filtered:
        return filtered
    return dedup


def build_ngrn_data(
    data_dir,
    k_core=2,
    rating_threshold=3.5,
    min_group_size=2,
    max_group_size=200,
    min_raters_in_group=1,
    only_k_core=False,
    max_fallback_groups=5000,
    use_raw_groups_only=False,
    group_candidates_path="",
    sampled_kcore_groups=0,
    edge_weight_mode="learned",
    seed=42,
    ratings_override=None,
    edges_override=None,
):
    """
    从 data_dir 下的 joint_*.csv 构建 NGRN 所需全部结构。
    ratings_override / edges_override: 可选，传入已加载的 (ratings, user_ids, item_ids) 或 edges 列表以做扰动实验。
    返回一个 dict，包含：
      - user_labels, item_labels, tag_vocab
      - adj, edge_weight, candidates, group_r
      - ratings, user_ids, item_ids
      - y_ic: dict (i, c_idx) -> 0/1
      - group_item_ratings, n_users, n_items, n_groups, n_tags
    max_fallback_groups: fallback 时从 joint_social_groups 补充的群组数量上限，避免候选暴增。
    use_raw_groups_only: 若 True，候选群组仅来自 joint_social_groups.csv（不用 k-core），用于“基线(原始群组)”对比。
    sampled_kcore_groups: >0 时，若 k-core 连通分量过少，则先在 k-core 图内采样最多该数量的“仍满足 k-core”的子群组。
    """
    data_dir = Path(data_dir)
    ratings_path = data_dir / "joint_ratings.csv"
    edges_path = data_dir / "joint_social_edges.csv"
    movies_path = data_dir / "joint_movies.csv"

    if ratings_override is not None:
        ratings, user_ids, item_ids = ratings_override
    else:
        ratings, user_ids, item_ids = _load_ratings(ratings_path)
    if edges_override is not None:
        edges = list(edges_override)
    else:
        edges = _load_edges(edges_path)
    item_labels = _load_item_labels(movies_path)

    # 只保留两端均在评分用户集合内的边，避免图中出现 user_emb 范围外的节点导致 IndexError
    user_ids_set = set(user_ids)
    edges = [(u, v) for u, v in edges if u in user_ids_set and v in user_ids_set]

    user_labels = _user_label_sets(ratings, item_labels)
    tag_vocab = sorted(set().union(*(item_labels.get(i, set()) for i in item_ids)))
    if not tag_vocab:
        tag_vocab = ["Unknown"]

    user_items = _user_item_sets(ratings)
    base_mode = "cosine" if edge_weight_mode == "cosine" else "jaccard"
    adj, edge_weight = build_weighted_graph(edges, user_labels, user_items, base_weight_mode=base_mode)

    # 候选群组：优先 group_candidates_path；否则按既有逻辑
    used_social_groups_fallback = False
    used_kcore_sampling = False
    n_sampled_kcore = 0
    n_kcore = 0
    if group_candidates_path:
        candidates = _load_candidates_from_file(
            group_candidates_path,
            user_ids_set=user_ids_set,
            min_group_size=min_group_size,
            max_group_size=max_group_size,
        )
    elif use_raw_groups_only:
        candidates = []
        groups_path = data_dir / "joint_social_groups.csv"
        if groups_path.exists():
            gid_to_users = defaultdict(set)
            with open(groups_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    u, g = int(row["user_id"]), int(row["group_id"])
                    gid_to_users[g].add(u)
            extra = [sorted(gu) for gu in gid_to_users.values() if min_group_size <= len(gu) <= max_group_size]
            if extra:
                if len(extra) > max_fallback_groups:
                    rating_count = defaultdict(int)
                    for u, i, r in ratings:
                        rating_count[u] += 1
                    scored = [(sum(rating_count[u] for u in gu), tuple(gu)) for gu in extra]
                    scored.sort(key=lambda x: (-x[0], x[1]))
                    extra = [list(gu) for _, gu in scored[:max_fallback_groups]]
                candidates = extra
    else:
        candidates_raw = k_core_components(edges, k_core)
        n_kcore = len(candidates_raw)
        candidates = k_core_candidates_filtered(
            edges, k_core, min_group_size=min_group_size, max_group_size=max_group_size
        )
        if len(candidates) <= 1 and int(sampled_kcore_groups or 0) > 0:
            sampled = sample_kcore_subgroups(
                edges=edges,
                k=k_core,
                min_group_size=min_group_size,
                max_group_size=max_group_size,
                max_groups=int(sampled_kcore_groups),
                seed=seed,
            )
            if sampled:
                # 用采样结果替换单一大分量（或空分量）以提升候选多样性
                candidates = sampled
                used_kcore_sampling = True
                n_sampled_kcore = len(sampled)
        if not only_k_core and len(candidates) <= 1:
            groups_path = data_dir / "joint_social_groups.csv"
            if groups_path.exists():
                gid_to_users = defaultdict(set)
                with open(groups_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        u, g = int(row["user_id"]), int(row["group_id"])
                        gid_to_users[g].add(u)
                extra = [sorted(gu) for gu in gid_to_users.values() if min_group_size <= len(gu) <= max_group_size]
                if extra:
                    if len(extra) > max_fallback_groups:
                        rating_count = defaultdict(int)
                        for u, i, r in ratings:
                            rating_count[u] += 1
                        scored = [(sum(rating_count[u] for u in gu), tuple(gu)) for gu in extra]
                        scored.sort(key=lambda x: (-x[0], x[1]))
                        extra = [list(gu) for _, gu in scored[:max_fallback_groups]]
                    used_social_groups_fallback = True
                    candidates = extra if len(candidates) == 0 else (candidates + extra)

    group_r, tag_list = group_label_multihot(candidates, user_labels, tag_vocab)

    user_to_items = defaultdict(dict)
    for u, i, r in ratings:
        user_to_items[u][i] = r

    y_ic = {}
    group_item_ratings = defaultdict(dict)
    for c_idx, V_c in enumerate(candidates):
        for i in item_ids:
            omega = [u for u in V_c if i in user_to_items[u]]
            if len(omega) < min_raters_in_group:
                continue
            r_bar = np.mean([user_to_items[u][i] for u in omega])
            group_item_ratings[c_idx][i] = r_bar
            y_ic[(i, c_idx)] = 1 if r_bar >= rating_threshold else 0

    item_id_to_idx = {mid: idx for idx, mid in enumerate(item_ids)}
    np.random.seed(seed)
    return {
        "user_labels": user_labels,
        "item_labels": item_labels,
        "tag_vocab": tag_list,
        "adj": adj,
        "edge_weight": edge_weight,
        "candidates": candidates,
        "group_r": group_r,
        "ratings": ratings,
        "user_ids": user_ids,
        "item_ids": item_ids,
        "item_id_to_idx": item_id_to_idx,
        "y_ic": y_ic,
        "group_item_ratings": dict(group_item_ratings),
        "rating_threshold": rating_threshold,
        "n_users": len(user_ids),
        "n_items": len(item_ids),
        "n_groups": len(candidates),
        "n_tags": len(tag_list),
        "k_core": k_core,
        "used_social_groups_fallback": used_social_groups_fallback,
        "used_kcore_sampling": used_kcore_sampling,
        "n_sampled_kcore_groups": n_sampled_kcore,
        "n_kcore_components": n_kcore,
    }


def split_y_ic(y_ic, test_ratio=0.2, seed=42):
    """
    将 y_ic 划分为训练用与测试评估用，便于和论文的“测试集指标”公平对比。
    - 只对正样本 (y_ic=1) 做划分；负样本全部用于训练。
    - 返回的 train_y_ic 中，测试集正样本被置为 0（不参与训练）；eval 时仅用 test_item_to_relevant。
    """
    if test_ratio <= 0 or test_ratio >= 1:
        return dict(y_ic), defaultdict(set)

    pos_pairs = [(i, c) for (i, c), v in y_ic.items() if v == 1]
    if len(pos_pairs) < 10:
        return dict(y_ic), defaultdict(set)

    rng = np.random.default_rng(seed)
    rng.shuffle(pos_pairs)
    n_test = max(1, int(len(pos_pairs) * test_ratio))
    test_pairs = set(pos_pairs[:n_test])
    train_pos_set = set(pos_pairs[n_test:])

    train_y_ic = {}
    for (i, c), v in y_ic.items():
        if (i, c) in test_pairs:
            train_y_ic[(i, c)] = 0  # 测试集正样本在训练时视为 0，不参与正样本采样
        else:
            train_y_ic[(i, c)] = v

    test_item_to_relevant = defaultdict(set)
    for (i, c) in test_pairs:
        test_item_to_relevant[i].add(c)

    return train_y_ic, dict(test_item_to_relevant)
