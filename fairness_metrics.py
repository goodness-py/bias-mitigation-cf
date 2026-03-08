import math
import numpy as np
import pandas as pd
from collections import defaultdict

def _split_by_gender(user_ids, gender_map):
    """Partition a list of user IDs into {'M': [...], 'F': [...]}."""
    groups = defaultdict(list)
    for uid in user_ids:
        g = gender_map.get(uid)
        if g in ("M", "F"):
            groups[g].append(uid)
    return groups


def input_activity_gap(stage1_df):
    """
    Demographic parity gap at the input stage.
    Measures difference in average number of ratings per user between M and F.
    Returns (M_avg - F_avg) normalised by overall average.
    """
    if "gender" not in stage1_df.columns:
        return 0.0
    avg = stage1_df.groupby("gender")["n_ratings"].mean()
    m = avg.get("M", 0.0)
    f = avg.get("F", 0.0)
    overall = stage1_df["n_ratings"].mean()
    return (m - f) / overall if overall else 0.0


def similarity_bias(sim_df, gender_map):
    """
    Bias in the similarity matrix.
    For each user, compute fraction of their top-50 similar users that share gender.
    Gap = |same-gender fraction_M - same-gender fraction_F|.
    Higher = more homophily / demographic segregation.
    """
    top_k = 50
    same_frac = {"M": [], "F": []}

    for uid in sim_df.index:
        g = gender_map.get(uid)
        if g not in ("M", "F"):
            continue
        row = sim_df.loc[uid].drop(uid)           # exclude self
        top_nbrs = row.nlargest(top_k).index.tolist()
        n_same = sum(1 for n in top_nbrs if gender_map.get(n) == g)
        same_frac[g].append(n_same / top_k)

    m_mean = np.mean(same_frac["M"]) if same_frac["M"] else 0.0
    f_mean = np.mean(same_frac["F"]) if same_frac["F"] else 0.0
    return abs(m_mean - f_mean)


def top_similar_composition_gap(top_similar, gender_map):
    """
    Demographic composition gap in top-50 similar users.
    For minority group F: fraction of their similar-user lists that are M
    minus the population proportion of M.
    Gap > 0 means F users have over-representation of M in their neighbourhood.
    """
    n_m = sum(1 for g in gender_map.values() if g == "M")
    n_f = sum(1 for g in gender_map.values() if g == "F")
    total = n_m + n_f
    pop_m_frac = n_m / total if total else 0.5

    f_users = [uid for uid, g in gender_map.items() if g == "F"]
    if not f_users:
        return 0.0

    cross_fracs = []
    for uid in f_users:
        nbrs = top_similar.get(uid, [])
        if not nbrs:
            continue
        frac_m = sum(1 for n in nbrs if gender_map.get(n) == "M") / len(nbrs)
        cross_fracs.append(frac_m)

    if not cross_fracs:
        return 0.0
    return abs(np.mean(cross_fracs) - pop_m_frac)


def neighbourhood_composition_gap(neighbourhood, gender_map):
    """
    Demographic composition gap in selected k-neighbourhoods.
    For each user group (M/F), measure average fraction of opposite-gender neighbours
    versus population baseline.
    Returns absolute deviation from population proportion.
    """
    n_m = sum(1 for g in gender_map.values() if g == "M")
    n_f = sum(1 for g in gender_map.values() if g == "F")
    total = n_m + n_f
    pop_f_frac = n_f / total if total else 0.5

    f_fracs_per_m_user = []
    for uid, nbrs in neighbourhood.items():
        if not nbrs or gender_map.get(uid) != "M":
            continue
        frac_f = sum(1 for n in nbrs if gender_map.get(n) == "F") / len(nbrs)
        f_fracs_per_m_user.append(frac_f)

    if not f_fracs_per_m_user:
        return 0.0
    actual_f_frac = np.mean(f_fracs_per_m_user)
    return abs(actual_f_frac - pop_f_frac)


def prediction_rmse_gap(pred_df, gender_map):
    """
    Prediction accuracy gap between groups.
    Returns |RMSE_M - RMSE_F|.
    """
    pred_df = pred_df.copy()
    pred_df["gender"] = pred_df["user_id"].map(gender_map)

    def rmse(df):
        err = df["est_rating"] - df["true_rating"]
        return math.sqrt((err ** 2).mean()) if len(df) else 0.0

    m_rmse = rmse(pred_df[pred_df["gender"] == "M"])
    f_rmse = rmse(pred_df[pred_df["gender"] == "F"])
    return abs(m_rmse - f_rmse)


def rmse_by_group(pred_df, gender_map):
    """Return dict {gender: RMSE}."""
    pred_df = pred_df.copy()
    pred_df["gender"] = pred_df["user_id"].map(gender_map)
    result = {}
    for g in ("M", "F"):
        sub = pred_df[pred_df["gender"] == g]
        if len(sub):
            err = sub["est_rating"] - sub["true_rating"]
            result[g] = math.sqrt((err ** 2).mean())
        else:
            result[g] = 0.0
    return result


def demographic_parity_gap(ranking, gender_map):
    """
    Demographic parity gap at ranking stage.
    Measures difference in average number of recommendations received
    between male and female users.
    Returns (avg_recs_M - avg_recs_F) / overall_avg.
    """
    counts = {"M": [], "F": []}
    for uid, recs in ranking.items():
        g = gender_map.get(uid)
        if g in ("M", "F"):
            counts[g].append(len(recs))

    m_avg = np.mean(counts["M"]) if counts["M"] else 0.0
    f_avg = np.mean(counts["F"]) if counts["F"] else 0.0
    overall = np.mean(counts["M"] + counts["F"]) if (counts["M"] or counts["F"]) else 1.0
    return abs(m_avg - f_avg) / overall if overall else 0.0


def equal_opportunity_gap(pred_df, gender_map, threshold=4.0):
    """
    Equal opportunity gap: difference in True Positive Rate (hit rate at threshold)
    between M and F groups.
    """
    pred_df = pred_df.copy()
    pred_df["gender"] = pred_df["user_id"].map(gender_map)
    pred_df["relevant"] = pred_df["true_rating"] >= threshold
    pred_df["predicted_relevant"] = pred_df["est_rating"] >= threshold

    def tpr(df):
        pos = df[df["relevant"]]
        if len(pos) == 0:
            return 0.0
        return (pos["predicted_relevant"].sum()) / len(pos)

    m_tpr = tpr(pred_df[pred_df["gender"] == "M"])
    f_tpr = tpr(pred_df[pred_df["gender"] == "F"])
    return abs(m_tpr - f_tpr)


if __name__ == "__main__":
    print("fairness_metrics.py loaded — use via pipeline_decomposition or main_experiment.")
