"""
mitigation_neighbourhood.py — Layer 4
In-processing mitigation: constrained neighbourhood selection.
Enforces a minimum representation threshold for the minority gender group (F)
in each user's k-neighbourhood, then recomputes predictions and fairness metrics.

"""

import os
import json
import math
import numpy as np
import pandas as pd
from collections import defaultdict

from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split

from data_loader import load_all
from baseline_cf import train_baseline, build_surprise_dataset
from stage_extraction import extract_stage5_predictions
from fairness_metrics import (
    neighbourhood_composition_gap,
    prediction_rmse_gap,
    demographic_parity_gap,
)
from evaluation_metrics import ndcg_at_k, rmse

RESULTS_DIR = "results"
MIN_F_FRACTION = 0.30  # minimum fraction of F users in each neighbourhood


def constrained_neighbourhood(algo, trainset, gender_map, k=50, min_f_frac=MIN_F_FRACTION):
    """
    For each user, select k neighbours while ensuring at least min_f_frac
    fraction are female (minority group).

    Strategy: first fill with available F neighbours, then fill remainder
    with M/neutral neighbours, sorted by similarity descending.

    Returns dict {raw_user_id: [raw_neighbour_ids]}.
    """
    sim = algo.compute_similarities()
    n_users = trainset.n_users
    neighbourhood = {}

    target_f = max(1, round(k * min_f_frac))
    target_m = k - target_f

    for inner_uid in range(n_users):
        raw_uid = trainset.to_raw_uid(inner_uid)
        sims_row = sim[inner_uid].copy()
        sims_row[inner_uid] = -1  # exclude self

        # Sort all other users by similarity descending
        sorted_inner = np.argsort(sims_row)[::-1]

        f_nbrs = []
        m_nbrs = []
        for inner_nbr in sorted_inner:
            raw_nbr = trainset.to_raw_uid(inner_nbr)
            g = gender_map.get(raw_nbr)
            if g == "F" and len(f_nbrs) < target_f:
                f_nbrs.append(raw_nbr)
            elif g != "F" and len(m_nbrs) < target_m:
                m_nbrs.append(raw_nbr)
            if len(f_nbrs) >= target_f and len(m_nbrs) >= target_m:
                break

        # If not enough F available, fill remainder with M
        combined = f_nbrs + m_nbrs
        if len(combined) < k:
            for inner_nbr in sorted_inner:
                raw_nbr = trainset.to_raw_uid(inner_nbr)
                if raw_nbr not in combined:
                    combined.append(raw_nbr)
                if len(combined) >= k:
                    break

        neighbourhood[raw_uid] = combined[:k]

    return neighbourhood


def predict_with_constrained_neighbourhood(algo, trainset, testset, constrained_nbrs, gender_map):
    """
    Generate predictions for testset using constrained neighbourhoods.
    Falls back to baseline prediction if constrained neighbours give no overlap.
    Returns list of (uid, iid, true_r, est_r) tuples.
    """
    sim = algo.compute_similarities()

    # Build inner_id lookup
    raw_to_inner = {trainset.to_raw_uid(i): i for i in range(trainset.n_users)}

    # Build user-item rating lookup from trainset
    user_items = defaultdict(dict)
    for uid, iid, rating in trainset.all_ratings():
        raw_uid = trainset.to_raw_uid(uid)
        raw_iid = trainset.to_raw_iid(iid)
        user_items[raw_uid][raw_iid] = rating

    global_mean = trainset.global_mean
    predictions = []

    for raw_uid, raw_iid, true_r in testset:
        # Constrained neighbours for this user
        nbr_raw_ids = constrained_nbrs.get(raw_uid, [])

        # Weighted average of neighbour ratings for this item
        numerator = 0.0
        denominator = 0.0
        inner_uid = raw_to_inner.get(raw_uid)

        for raw_nbr in nbr_raw_ids:
            inner_nbr = raw_to_inner.get(raw_nbr)
            if inner_nbr is None:
                continue
            nbr_rating = user_items[raw_nbr].get(raw_iid)
            if nbr_rating is None:
                continue
            # Similarity weight
            w = sim[inner_uid][inner_nbr] if inner_uid is not None else 0.0
            if w <= 0:
                continue
            numerator += w * nbr_rating
            denominator += w

        if denominator > 0:
            est = numerator / denominator
        else:
            est = global_mean  # fallback

        # Clip to rating scale
        est = max(1.0, min(5.0, est))
        predictions.append((raw_uid, raw_iid, true_r, est))

    return predictions


def run(ratings=None, users=None, gender_map=None, algo=None,
        trainset=None, testset=None, predictions_baseline=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if ratings is None:
        print("\n[1/5] Loading data and training CF...")
        ratings, users, items, gender_map, _ = load_all()
        algo, trainset, testset, predictions_baseline = train_baseline(ratings)

    print("[2/5] Computing baseline neighbourhood and fairness...")
    from stage_extraction import extract_stage4_neighbourhood
    baseline_nbrs = extract_stage4_neighbourhood(algo, trainset)
    baseline_nbr_gap = neighbourhood_composition_gap(baseline_nbrs, gender_map)
    baseline_pred_df = extract_stage5_predictions(predictions_baseline)
    baseline_rmse_gap = prediction_rmse_gap(baseline_pred_df, gender_map)
    baseline_rmse_val = rmse(baseline_pred_df)
    print(f"  Baseline nbr composition gap : {baseline_nbr_gap:.4f}")
    print(f"  Baseline prediction RMSE gap : {baseline_rmse_gap:.4f}")
    print(f"  Baseline overall RMSE        : {baseline_rmse_val:.4f}")

    print(f"[3/5] Building constrained neighbourhoods (min F={MIN_F_FRACTION:.0%})...")
    constrained_nbrs = constrained_neighbourhood(
        algo, trainset, gender_map, k=50, min_f_frac=MIN_F_FRACTION
    )
    constrained_nbr_gap = neighbourhood_composition_gap(constrained_nbrs, gender_map)
    print(f"  Constrained nbr composition gap: {constrained_nbr_gap:.4f}")

    print("[4/5] Generating predictions with constrained neighbourhoods...")
    constrained_preds = predict_with_constrained_neighbourhood(
        algo, trainset, testset, constrained_nbrs, gender_map
    )

    # Convert to DataFrame
    pred_rows = [{"user_id": uid, "item_id": iid, "true_rating": r, "est_rating": est}
                 for uid, iid, r, est in constrained_preds]
    constrained_pred_df = pd.DataFrame(pred_rows)

    constrained_rmse_gap = prediction_rmse_gap(constrained_pred_df, gender_map)
    constrained_rmse_val = rmse(constrained_pred_df)

    print("[5/5] Summarising results...")
    nbr_gap_reduction = baseline_nbr_gap - constrained_nbr_gap
    rmse_change = constrained_rmse_val - baseline_rmse_val
    pred_gap_reduction = baseline_rmse_gap - constrained_rmse_gap

    print(f"\n  === Neighbourhood Mitigation Results ===")
    print(f"  Nbr composition gap: {baseline_nbr_gap:.4f} → {constrained_nbr_gap:.4f}  (Δ={nbr_gap_reduction:+.4f})")
    print(f"  Prediction RMSE gap: {baseline_rmse_gap:.4f} → {constrained_rmse_gap:.4f}  (Δ={pred_gap_reduction:+.4f})")
    print(f"  Overall RMSE       : {baseline_rmse_val:.4f} → {constrained_rmse_val:.4f}  (Δ={rmse_change:+.4f})")

    result = {
        "strategy": "in_processing_neighbourhood_constraint",
        "min_f_fraction": MIN_F_FRACTION,
        "baseline_nbr_composition_gap": float(baseline_nbr_gap),
        "constrained_nbr_composition_gap": float(constrained_nbr_gap),
        "nbr_gap_reduction": float(nbr_gap_reduction),
        "baseline_pred_rmse_gap": float(baseline_rmse_gap),
        "constrained_pred_rmse_gap": float(constrained_rmse_gap),
        "pred_gap_reduction": float(pred_gap_reduction),
        "baseline_overall_rmse": float(baseline_rmse_val),
        "constrained_overall_rmse": float(constrained_rmse_val),
        "rmse_change": float(rmse_change),
    }

    json_path = os.path.join(RESULTS_DIR, "mitigation_neighbourhood_results.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {json_path}")

    return result


if __name__ == "__main__":
    run()
