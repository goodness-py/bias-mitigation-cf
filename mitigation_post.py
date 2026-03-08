"""
mitigation_post.py — Layer 4
Post-processing mitigation: re-ranks Top-N recommendation lists to enforce
demographic parity by interleaving items from M and F rated item pools.

Measures:
  - Bias reduction (demographic parity gap before vs after)
  - NDCG loss (accuracy cost of re-ranking)
  
"""

import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_all
from baseline_cf import train_baseline, get_top_n
from stage_extraction import extract_stage5_predictions, extract_stage6_ranking
from fairness_metrics import demographic_parity_gap
from evaluation_metrics import ndcg_at_k

RESULTS_DIR = "results"


def build_item_gender_affinity(ratings_df, gender_map, threshold=0.6):
    """
    Label each item as 'M', 'F', or 'neutral' based on whether ≥threshold
    of its raters belong to one gender group.
    Returns dict {item_id: 'M'|'F'|'neutral'}.
    """
    ratings_df = ratings_df.copy()
    ratings_df["gender"] = ratings_df["user_id"].map(gender_map)

    item_gender = {}
    for item_id, grp in ratings_df.groupby("item_id"):
        gender_counts = grp["gender"].value_counts()
        total = len(grp)
        if total == 0:
            item_gender[item_id] = "neutral"
            continue
        m_frac = gender_counts.get("M", 0) / total
        f_frac = gender_counts.get("F", 0) / total
        if m_frac >= threshold:
            item_gender[item_id] = "M"
        elif f_frac >= threshold:
            item_gender[item_id] = "F"
        else:
            item_gender[item_id] = "neutral"

    return item_gender


def rerank_for_parity(top_n, item_gender_affinity, target_f_fraction=0.5, n=10):
    """
    Re-rank each user's recommendation list to enforce approximate demographic
    parity of item exposure (M-affinity vs F-affinity items).

    Strategy: greedy interleaving — alternate M and F affinity items while
    filling remaining slots with neutral items.

    Returns dict {user_id: [(item_id, est_rating), ...]} of length n.
    """
    reranked = {}
    for uid, recs in top_n.items():
        m_items = [(iid, est) for iid, est in recs if item_gender_affinity.get(iid) == "M"]
        f_items = [(iid, est) for iid, est in recs if item_gender_affinity.get(iid) == "F"]
        neutral = [(iid, est) for iid, est in recs if item_gender_affinity.get(iid) == "neutral"]

        target_f = round(n * target_f_fraction)
        target_m = n - target_f

        new_list = []
        m_ptr, f_ptr, neu_ptr = 0, 0, 0

        # Fill target_f slots with F items, fallback to neutral
        for _ in range(target_f):
            if f_ptr < len(f_items):
                new_list.append(f_items[f_ptr]); f_ptr += 1
            elif neu_ptr < len(neutral):
                new_list.append(neutral[neu_ptr]); neu_ptr += 1

        # Fill target_m slots with M items, fallback to neutral
        for _ in range(target_m):
            if m_ptr < len(m_items):
                new_list.append(m_items[m_ptr]); m_ptr += 1
            elif neu_ptr < len(neutral):
                new_list.append(neutral[neu_ptr]); neu_ptr += 1

        # Fill any remaining slots from whatever's left
        remaining = (
            m_items[m_ptr:] + f_items[f_ptr:] + neutral[neu_ptr:]
        )
        new_list += remaining[: n - len(new_list)]

        reranked[uid] = new_list[:n]

    return reranked


def run(ratings=None, users=None, gender_map=None, predictions=None, top_n_original=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if ratings is None:
        print("\n[1/4] Loading data and training CF...")
        ratings, users, items, gender_map, _ = load_all()
        algo, trainset, testset, predictions = train_baseline(ratings)
        top_n_original = get_top_n(predictions, n=10)

    pred_df = extract_stage5_predictions(predictions)

    print("[2/4] Building item gender affinity map...")
    item_gender = build_item_gender_affinity(ratings, gender_map)
    n_m = sum(1 for v in item_gender.values() if v == "M")
    n_f = sum(1 for v in item_gender.values() if v == "F")
    n_neu = sum(1 for v in item_gender.values() if v == "neutral")
    print(f"  Items: M-affinity={n_m}, F-affinity={n_f}, neutral={n_neu}")

    print("[3/4] Computing pre-mitigation metrics...")
    pre_dp_gap = demographic_parity_gap(top_n_original, gender_map)
    pre_ndcg = ndcg_at_k(top_n_original, pred_df, k=10)
    print(f"  Pre-mitigation  DP gap={pre_dp_gap:.4f}  NDCG@10={pre_ndcg:.4f}")

    print("[4/4] Re-ranking for demographic parity...")
    top_n_reranked = rerank_for_parity(top_n_original, item_gender, n=10)
    post_dp_gap = demographic_parity_gap(top_n_reranked, gender_map)
    post_ndcg = ndcg_at_k(top_n_reranked, pred_df, k=10)
    print(f"  Post-mitigation DP gap={post_dp_gap:.4f}  NDCG@10={post_ndcg:.4f}")

    bias_reduction = pre_dp_gap - post_dp_gap
    ndcg_loss = pre_ndcg - post_ndcg
    print(f"\n  Bias reduction : {bias_reduction:+.4f}")
    print(f"  NDCG loss      : {ndcg_loss:+.4f}")

    result = {
        "strategy": "post_processing_reranking",
        "pre_dp_gap": float(pre_dp_gap),
        "post_dp_gap": float(post_dp_gap),
        "bias_reduction": float(bias_reduction),
        "pre_ndcg": float(pre_ndcg),
        "post_ndcg": float(post_ndcg),
        "ndcg_loss": float(ndcg_loss),
    }

    json_path = os.path.join(RESULTS_DIR, "mitigation_post_results.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {json_path}")

    return result, top_n_reranked


if __name__ == "__main__":
    run()
