import math
import numpy as np
import pandas as pd
from collections import defaultdict

def rmse(pred_df):
    """Root Mean Squared Error over all predictions."""
    err = pred_df["est_rating"] - pred_df["true_rating"]
    return math.sqrt((err ** 2).mean())


def mae(pred_df):
    """Mean Absolute Error over all predictions."""
    err = (pred_df["est_rating"] - pred_df["true_rating"]).abs()
    return err.mean()


def rmse_mae_by_group(pred_df, gender_map):
    """Return {gender: {'rmse': float, 'mae': float}} for M and F."""
    pred_df = pred_df.copy()
    pred_df["gender"] = pred_df["user_id"].map(gender_map)
    result = {}
    for g in ("M", "F"):
        sub = pred_df[pred_df["gender"] == g]
        if len(sub):
            err = sub["est_rating"] - sub["true_rating"]
            result[g] = {
                "rmse": math.sqrt((err ** 2).mean()),
                "mae": err.abs().mean(),
                "n": len(sub),
            }
        else:
            result[g] = {"rmse": 0.0, "mae": 0.0, "n": 0}
    return result


def _build_relevant_set(pred_df, threshold=4.0):
    """Return dict {user_id: set of relevant item_ids} from ground truth."""
    relevant = defaultdict(set)
    for _, row in pred_df.iterrows():
        if row["true_rating"] >= threshold:
            relevant[row["user_id"]].add(row["item_id"])
    return dict(relevant)


def precision_at_k(top_n, pred_df, k=10, threshold=4.0):
    """
    Precision@K averaged over users.
    top_n: dict {user_id: [(item_id, est), ...]}
    """
    relevant = _build_relevant_set(pred_df, threshold)
    scores = []
    for uid, recs in top_n.items():
        rec_items = [iid for iid, _ in recs[:k]]
        rel = relevant.get(uid, set())
        hits = sum(1 for iid in rec_items if iid in rel)
        scores.append(hits / k if k else 0.0)
    return np.mean(scores) if scores else 0.0


def recall_at_k(top_n, pred_df, k=10, threshold=4.0):
    """Recall@K averaged over users."""
    relevant = _build_relevant_set(pred_df, threshold)
    scores = []
    for uid, recs in top_n.items():
        rec_items = [iid for iid, _ in recs[:k]]
        rel = relevant.get(uid, set())
        if not rel:
            continue
        hits = sum(1 for iid in rec_items if iid in rel)
        scores.append(hits / len(rel))
    return np.mean(scores) if scores else 0.0


def ndcg_at_k(top_n, pred_df, k=10, threshold=4.0):
    """NDCG@K averaged over users."""
    relevant = _build_relevant_set(pred_df, threshold)

    def dcg(hits):
        return sum(h / math.log2(i + 2) for i, h in enumerate(hits))

    scores = []
    for uid, recs in top_n.items():
        rec_items = [iid for iid, _ in recs[:k]]
        rel = relevant.get(uid, set())
        hits = [1 if iid in rel else 0 for iid in rec_items]
        ideal = sorted(hits, reverse=True)
        ideal_dcg = dcg(ideal)
        if ideal_dcg == 0:
            continue
        scores.append(dcg(hits) / ideal_dcg)
    return np.mean(scores) if scores else 0.0


def ranking_metrics_by_group(top_n, pred_df, gender_map, k=10, threshold=4.0):
    """
    Return {gender: {'precision': float, 'recall': float, 'ndcg': float}}
    computed separately for M and F users.
    """
    result = {}
    for g in ("M", "F"):
        g_users = {uid for uid, gender in gender_map.items() if gender == g}
        g_top_n = {uid: recs for uid, recs in top_n.items() if uid in g_users}
        g_pred = pred_df[pred_df["user_id"].isin(g_users)]
        result[g] = {
            "precision@k": precision_at_k(g_top_n, g_pred, k=k, threshold=threshold),
            "recall@k": recall_at_k(g_top_n, g_pred, k=k, threshold=threshold),
            "ndcg@k": ndcg_at_k(g_top_n, g_pred, k=k, threshold=threshold),
            "n_users": len(g_users),
        }
    return result


def full_evaluation(predictions_df, top_n, gender_map, k=10):
    """
    Run all evaluation metrics.
    Returns a nested dict with overall and per-group results.
    """
    overall_rmse = rmse(predictions_df)
    overall_mae = mae(predictions_df)
    group_accuracy = rmse_mae_by_group(predictions_df, gender_map)
    group_ranking = ranking_metrics_by_group(top_n, predictions_df, gender_map, k=k)

    return {
        "overall": {"rmse": overall_rmse, "mae": overall_mae},
        "by_group_accuracy": group_accuracy,
        "by_group_ranking": group_ranking,
    }

if __name__ == "__main__":
    print("evaluation_metrics.py loaded — call full_evaluation() from main_experiment.")
