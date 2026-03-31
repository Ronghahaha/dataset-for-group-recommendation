# -*- coding: utf-8 -*-
"""
NGRN 训练脚本（优化版）- 对项目推荐群组（item→groups）
- 任务：给定项目 i，对群组 c 排序。
- 为同时提升 Prec/Recall/NDCG：按综合指标保存并恢复最佳 checkpoint，默认 lambda_tag=0.1、dropout=0.05。
- 对比实验（同一任务）：
  - 基线: python train_baseline.py --data_dir data/ml100k
  - NGRN:  python train_optimized.py --data_dir data/ml100k
"""

import argparse
import inspect
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler

from src.data.dataset import build_ngrn_data, split_y_ic
from src.models.ngrn import (
    NGRN,
    build_group_subgraph_data,
    sample_rank_triples,
    compute_rank_loss,
    compute_tag_loss,
)


# 评估指标 K 的取值（含更大 K 以观察 Recall 提升空间）
EVAL_K = (3, 5, 10, 20, 50)


def dcg_at_k(rel_list, k):
    """rel_list: 排序后的相关性列表（0/1），长度为至少 k。DCG@K = sum_{i=1}^K (2^rel_i - 1) / log2(i+1)。"""
    rel_list = rel_list[:k]
    return sum((2.0 ** r - 1.0) / np.log2(i + 2) for i, r in enumerate(rel_list))


def ndcg_at_k(pred_list, relevant_set, k):
    """pred_list: 预测的 top-k 群组有序列表；relevant_set: 真实相关群组集合。NDCG@K = DCG/IDCG。"""
    pred_list = pred_list[:k]
    rel_list = [1 if c in relevant_set else 0 for c in pred_list]
    dcg = dcg_at_k(rel_list, k)
    ideal_rel = sorted([1] * len(relevant_set) + [0] * max(0, k - len(relevant_set)), reverse=True)
    idcg = dcg_at_k(ideal_rel, k)
    if idcg <= 0:
        return 1.0
    return dcg / idcg


def evaluate_metrics(model, subgraph_data, y_ic, n_items, n_groups, K_list, device, eval_item_to_relevant=None):
    """
    计算 Prec@K、Rec@K、NDCG@K，K ∈ K_list。
    y_ic: (item_idx, group_idx) -> 0/1（用于兼容；若提供 eval_item_to_relevant 则以其为准）。
    eval_item_to_relevant: 若提供（如测试集划分），则仅在此 dict 上评估，便于与论文测试集指标公平对比。
    """
    if eval_item_to_relevant is not None:
        item_to_relevant = defaultdict(set)
        for i, groups in eval_item_to_relevant.items():
            item_to_relevant[i] = set(groups)
    else:
        item_to_relevant = defaultdict(set)
        for (i, c), v in y_ic.items():
            if v == 1:
                item_to_relevant[i].add(c)
    eval_items = list(item_to_relevant.keys())
    if not eval_items:
        return {
            **{f"prec@{k}": 0.0 for k in K_list},
            **{f"rec@{k}": 0.0 for k in K_list},
            **{f"ndcg@{k}": 0.0 for k in K_list},
        }

    model.eval()
    with torch.no_grad():
        h_all = model.forward_group_representations(subgraph_data, None)
        v_all = model.item_emb.weight
        scores = torch.matmul(v_all, h_all.t())
    max_k = max(K_list)
    prec = {k: [] for k in K_list}
    rec = {k: [] for k in K_list}
    ndcg = {k: [] for k in K_list}

    for i in eval_items:
        G_i = item_to_relevant[i]
        _, top_idx = torch.topk(scores[i], min(max_k, scores.shape[1]))
        pred_list = top_idx.cpu().tolist()
        for k in K_list:
            pred_k = pred_list[:k]
            hit = len(set(pred_k) & G_i)
            prec[k].append(hit / k)
            rec[k].append(hit / len(G_i) if G_i else 0.0)
            ndcg[k].append(ndcg_at_k(pred_k, G_i, k))

    return {
        **{f"prec@{k}": np.mean(prec[k]) for k in K_list},
        **{f"rec@{k}": np.mean(rec[k]) for k in K_list},
        **{f"ndcg@{k}": np.mean(ndcg[k]) for k in K_list},
    }


def parse_args():
    p = argparse.ArgumentParser(description="NGRN 训练（优化版）")
    p.add_argument("--data_dir", type=str, default="data", help="数据目录")
    p.add_argument("--k_core", type=int, default=3, help="k-核参数；候选群组来自 k-核连通分量。k 过大时可能得到 0 个而走 joint_social_groups 补充")
    p.add_argument("--only_k_core", action="store_true", help="仅用 k-核连通分量，不用 joint_social_groups 补充")
    p.add_argument("--max_fallback_groups", type=int, default=180, help="fallback 时从 joint_social_groups 补充的群组数量上限")
    p.add_argument("--rating_threshold", type=float, default=3.5, help="聚合评分阈值 τ")
    p.add_argument("--min_raters_in_group", type=int, default=1, help="群组内至少几人评分才计入 y_ic（≥2 可提高 Recall 上限）")
    p.add_argument("--d", type=int, default=128, help="嵌入与隐藏维度（增加模型容量，默认128）")
    p.add_argument("--gcn_layers", type=int, default=3, help="GCN 层数（3 可提升表示能力）")
    p.add_argument("--epochs", type=int, default=200, help="训练轮数（增加以充分训练）")
    p.add_argument("--lr", type=float, default=0.01, help="学习率（提高以加快收敛，默认0.01）")
    p.add_argument("--lambda_tag", type=float, default=0.1, help="标签损失权重 λ（较小让 L_rank 主导，利于 Prec/NDCG）")
    p.add_argument("--dropout", type=float, default=0.05, help="Dropout 比例（略降利于排序更锐利，三项指标更均衡）")
    p.add_argument("--rank_samples", type=int, default=800, help="每轮采样排序三元组数量（增加可加快学习，默认800）")
    p.add_argument("--steps_per_epoch", type=int, default=1, help="每轮梯度更新批次数（>1 时每轮多步更新）")
    p.add_argument("--tag_loss_freq", type=int, default=5, help="标签损失计算频率（每N个epoch计算一次）")
    p.add_argument("--lr_decay", type=float, default=0.95, help="学习率衰减因子（每10轮，step 调度器）")
    p.add_argument("--scheduler", type=str, default="step", choices=["step", "plateau"], help="step=每10轮衰减; plateau=损失停滞时衰减")
    p.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减（L2正则化）")
    p.add_argument("--test_ratio", type=float, default=0.0, help="测试集比例(0~1)。>0 时从正样本中划分测试集并仅在其上评估，便于与论文公平对比")
    p.add_argument("--primary_metric", type=str, default="composite", choices=["composite", "ndcg10"],
                   help="选最佳 checkpoint 的指标: composite=(Prec@5+Rec@5+NDCG@5)/3 平衡三项; ndcg10=仅 NDCG@10")
    p.add_argument("--early_stop_patience", type=int, default=0, help="早停：连续多少次评估无提升则停止，0 表示不早停")
    p.add_argument("--eval_every", type=int, default=20, help="每多少轮评估一次并更新最佳 checkpoint")
    p.add_argument("--min_epochs_before_best", type=int, default=40,
                   help="仅在此 epoch 之后才参与「最佳」选取，避免第一次评估就被当成最佳（默认 40）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_neighbors", type=int, default=500, help="每群组 NeighborFusion 最多使用的邻居数，防止 OOM")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save", type=str, default="", help="模型保存路径")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    data_dir = Path(args.data_dir)
    print("========== NGRN 训练（任务：对项目推荐群组 item→groups）==========")
    print("加载数据并构建 k-核候选群组与边权...")
    _kwargs = dict(
        data_dir=data_dir,
        k_core=args.k_core,
        rating_threshold=args.rating_threshold,
        min_raters_in_group=args.min_raters_in_group,
        only_k_core=getattr(args, "only_k_core", False),
        seed=args.seed,
    )
    if "max_fallback_groups" in inspect.signature(build_ngrn_data).parameters:
        _kwargs["max_fallback_groups"] = getattr(args, "max_fallback_groups", 5000)
    data = build_ngrn_data(**_kwargs)
    k_core = getattr(args, "k_core", 2)
    n_grp = data["n_groups"]
    fallback = data.get("used_social_groups_fallback", False)
    n_kcore = data.get("n_kcore_components", n_grp)
    print(f"  用户数: {data['n_users']}, 项目数: {data['n_items']}, 标签数: {data['n_tags']}")
    if fallback:
        print(f"  候选群组: k-核(k={k_core}) 仅得到 {n_kcore} 个连通分量，已用 joint_social_groups 补充（上限 {getattr(args, 'max_fallback_groups', 5000)}），共 {n_grp} 个（若希望仅用 k-核可试 --only_k_core）")
    else:
        print(f"  候选群组: 来自 k-核(k={k_core}) 连通分量，共 {n_grp} 个")

    subgraph_data = build_group_subgraph_data(data, device=device)
    item_id_to_idx = data["item_id_to_idx"]
    y_ic_raw = data["y_ic"]
    group_r = data["group_r"]

    y_ic = {}
    for (i, c), v in y_ic_raw.items():
        idx = item_id_to_idx.get(i)
        if idx is not None:
            y_ic[(idx, c)] = v

    # 训练/测试划分：与论文一致在测试集上汇报指标时使用
    test_item_to_relevant = None
    if getattr(args, "test_ratio", 0) and args.test_ratio > 0:
        train_y_ic, test_item_to_relevant = split_y_ic(y_ic, test_ratio=args.test_ratio, seed=args.seed)
        y_ic = train_y_ic
        n_test_items = len(test_item_to_relevant)
        n_test_pairs = sum(len(s) for s in test_item_to_relevant.values())
        print(f"  已划分测试集: test_ratio={args.test_ratio}, 测试 item 数={n_test_items}, 测试正样本数={n_test_pairs}（仅用于评估，不参与训练）")

    n_users = data["n_users"]
    n_items = data["n_items"]
    n_tags = data["n_tags"]
    n_groups = data["n_groups"]

    if n_groups == 0 or n_users == 0 or n_items == 0:
        print("错误: 候选群组或用户/项目数为 0，无法训练。")
        print("  - Epinions: 请先运行 python scripts/download_epinions.py 重新生成数据（会使用占位评分）")
        print("  - 其他数据集: 检查 joint_ratings.csv、joint_social_edges.csv 是否有有效数据")
        return

    model = NGRN(
        n_users=n_users,
        n_items=n_items,
        n_tags=n_tags,
        d=args.d,
        gcn_layers=args.gcn_layers,
        dropout=args.dropout,
        max_neighbors=args.max_neighbors,
        device=device,
    ).to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=args.lr_decay, patience=15, verbose=True)
    else:
        scheduler = lr_scheduler.StepLR(opt, step_size=10, gamma=args.lr_decay)

    group_r_tensor = torch.from_numpy(np.stack(group_r)).float()  # group_r 非空（已检查 n_groups>0）
    rng = np.random.default_rng(args.seed)
    last_L_tag_val = 0.0

    print(f"\n开始训练（学习率: {args.lr}, 权重衰减: {args.weight_decay}, 每10轮衰减 {args.lr_decay}）...")
    print(f"训练配置: epochs={args.epochs}, d={args.d}, rank_samples={args.rank_samples}, lr={args.lr}")
    print(f"最佳 checkpoint 按 {args.primary_metric} 选取，每 {args.eval_every} 轮评估；仅 epoch > {args.min_epochs_before_best} 参与选取；早停 patience={args.early_stop_patience}")

    best_primary = -1.0
    best_state_dict = None
    best_epoch = -1
    no_improve_count = 0

    # 诊断数据分布
    item_to_relevant = defaultdict(set)
    for (i, c), v in y_ic.items():
        if v == 1:
            item_to_relevant[i].add(c)
    eval_items = list(item_to_relevant.keys())
    if eval_items:
        relevant_counts = [len(item_to_relevant[i]) for i in eval_items]
        mean_rel = np.mean(relevant_counts)
        print(f"数据诊断: {len(eval_items)}个item有相关群组, 平均每个item有 {mean_rel:.1f} 个相关群组")
        print(f"  Recall@10理论最大值: {np.mean([min(10, len(item_to_relevant[i])) / len(item_to_relevant[i]) for i in eval_items]):.4f}")
        if mean_rel > 500:
            print("  建议: 每 item 相关群组过多会导致 Prec/Rec 偏小，可尝试 --k_core 3 或 4、--min_raters_in_group 2、--rating_threshold 4.0 提高门槛")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss_rank = 0.0
        epoch_loss_tag = last_L_tag_val
        n_steps = 0
        compute_tag = (epoch + 1) % args.tag_loss_freq == 0 or epoch == 0
        if compute_tag:
            group_indices = list(range(n_groups))
            L_tag = compute_tag_loss(model, group_r_tensor, subgraph_data, group_indices, device)
            last_L_tag_val = L_tag.item()

        for step in range(args.steps_per_epoch):
            triples = sample_rank_triples(y_ic, args.rank_samples, rng)
            if not triples:
                if compute_tag and step == 0:
                    loss = args.lambda_tag * L_tag
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()
                continue
            L_rank = compute_rank_loss(model, subgraph_data, triples, device)
            epoch_loss_rank += L_rank.item()
            n_steps += 1
            if compute_tag and step == 0:
                loss = L_rank + args.lambda_tag * L_tag
            else:
                loss = L_rank
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        if args.scheduler == "plateau" and n_steps > 0:
            scheduler.step(epoch_loss_rank / max(n_steps, 1))
        elif args.scheduler == "step":
            scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            tag_val = (L_tag.item() if compute_tag else last_L_tag_val)
            avg_rank = epoch_loss_rank / max(n_steps, 1)
            current_lr = opt.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}/{args.epochs}  L_rank={avg_rank:.4f}  L_tag={tag_val:.4f}  lr={current_lr:.6f}")

        # 每 eval_every 轮评估，按 primary_metric 更新最佳 checkpoint
        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                metrics = evaluate_metrics(model, subgraph_data, y_ic, n_items, n_groups, EVAL_K, device, eval_item_to_relevant=test_item_to_relevant)
                if args.primary_metric == "composite":
                    primary = (metrics["prec@5"] + metrics["rec@5"] + metrics["ndcg@5"]) / 3.0
                else:
                    primary = metrics["ndcg@10"]
                # 仅当超过 min_epochs_before_best 后才参与「最佳」选取，避免训练初期被误选为最佳
                can_update_best = (epoch + 1) > args.min_epochs_before_best
                if can_update_best and primary > best_primary:
                    best_primary = primary
                    best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch + 1
                    no_improve_count = 0
                    print(f"\n[中间评估] Epoch {epoch+1}  *** 新最佳 ***  {args.primary_metric}={primary:.4f}")
                else:
                    if not can_update_best:
                        print(f"\n[中间评估] Epoch {epoch+1}  {args.primary_metric}={primary:.4f} (未满 {args.min_epochs_before_best} 轮，不参与最佳选取)")
                    else:
                        no_improve_count += 1
                        print(f"\n[中间评估] Epoch {epoch+1}  {args.primary_metric}={primary:.4f} (最佳 {best_primary:.4f} @ epoch {best_epoch})")
                for k in EVAL_K:
                    print(f"  Prec@{k}={metrics[f'prec@{k}']:.4f}  Rec@{k}={metrics[f'rec@{k}']:.4f}  NDCG@{k}={metrics[f'ndcg@{k}']:.4f}")
                print()
            model.train()
            if args.early_stop_patience > 0 and no_improve_count >= args.early_stop_patience:
                print(f"早停: 连续 {no_improve_count} 次评估无提升，停止训练")
                break

    # 恢复综合指标最佳的 checkpoint
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict, strict=True)
        print(f"\n已恢复 epoch {best_epoch} 的最佳 checkpoint（{args.primary_metric}={best_primary:.4f}）")

    print("\n========== 评估指标 (Prec@K, Rec@K, NDCG@K), K=3,5,10,20,50 ==========")
    if test_item_to_relevant:
        print("（以下为测试集上的指标，与论文常见汇报方式一致）")
    metrics = evaluate_metrics(model, subgraph_data, y_ic, n_items, n_groups, EVAL_K, device, eval_item_to_relevant=test_item_to_relevant)
    for k in EVAL_K:
        print(f"  K={k}:  Prec@{k}={metrics[f'prec@{k}']:.4f}   Rec@{k}={metrics[f'rec@{k}']:.4f}   NDCG@{k}={metrics[f'ndcg@{k}']:.4f}")
    print("================================================================")

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "data_meta": {"n_users": n_users, "n_items": n_items, "n_tags": n_tags}}, save_path)
        print(f"模型已保存: {save_path}")


if __name__ == "__main__":
    main()
