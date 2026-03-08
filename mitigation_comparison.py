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
from stage_extraction import extract_stage5_predictions
from fairness_metrics import (
    prediction_rmse_gap,
    demographic_parity_gap,
    neighbourhood_composition_gap,
)
from evaluation_metrics import rmse, ndcg_at_k
import mitigation_post
import mitigation_neighbourhood

RESULTS_DIR = "results"

def oversample_minority(ratings_df, gender_map, oversample_factor=2.0):
    """
    Duplicate ratings from F users by oversample_factor to balance
    the training distribution.
    Returns augmented ratings DataFrame.
    """
    ratings_df = ratings_df.copy()
    ratings_df["gender"] = ratings_df["user_id"].map(gender_map)

    f_ratings = ratings_df[ratings_df["gender"] == "F"]
    n_extra = int(len(f_ratings) * (oversample_factor - 1.0))

    extra = f_ratings.sample(n=n_extra, replace=True, random_state=42)
    augmented = pd.concat([ratings_df, extra], ignore_index=True)
    augmented = augmented.drop(columns=["gender"])
    return augmented

def run_preprocessing_strategy(ratings, gender_map):
    """Train CF on oversampled data and return metrics."""
    print("\n  [Pre-processing] Oversampling F ratings (factor=2)...")
    augmented = oversample_minority(ratings, gender_map)
    algo, trainset, testset, predictions = train_baseline(augmented)
    top_n = get_top_n(predictions, n=10)
    pred_df = extract_stage5_predictions(predictions)

    dp_gap = demographic_parity_gap(top_n, gender_map)
    rmse_gap = prediction_rmse_gap(pred_df, gender_map)
    overall_rmse = rmse(pred_df)
    ndcg = ndcg_at_k(top_n, pred_df, k=10)

    print(f"DP gap={dp_gap:.4f}  RMSE gap={rmse_gap:.4f}  RMSE={overall_rmse:.4f}  NDCG@10={ndcg:.4f}")
    return {
        "strategy": "pre_processing_oversampling",
        "dp_gap": float(dp_gap),
        "rmse_gap": float(rmse_gap),
        "overall_rmse": float(overall_rmse),
        "ndcg": float(ndcg),
    }

def plot_grouped_bar(results_list, baseline, filename="mitigation_comparison_bar.png"):
    """Grouped bar chart: DP gap and RMSE gap per strategy."""
    labels = [r["strategy"].replace("_", "\n") for r in results_list]
    dp_gaps = [r["dp_gap"] for r in results_list]
    rmse_gaps = [r["rmse_gap"] for r in results_list]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, dp_gaps, width, label="DP Gap", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, rmse_gaps, width, label="RMSE Gap", color="#DD8452")

    # Baseline reference lines
    ax.axhline(baseline["dp_gap"], color="#4C72B0", linestyle="--", alpha=0.5,
               label=f"Baseline DP Gap ({baseline['dp_gap']:.4f})")
    ax.axhline(baseline["rmse_gap"], color="#DD8452", linestyle="--", alpha=0.5,
               label=f"Baseline RMSE Gap ({baseline['rmse_gap']:.4f})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Bias Gap")
    ax.set_title("Bias Reduction by Mitigation Strategy")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_pareto_frontier(results_list, baseline, filename="mitigation_pareto.png"):
    """Accuracy (NDCG) vs fairness (DP gap) scatter — Pareto frontier."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # Baseline point
    ax.scatter(baseline["dp_gap"], baseline["ndcg"], marker="*", s=200,
               color="red", zorder=5, label="Baseline")
    ax.annotate("Baseline", (baseline["dp_gap"], baseline["ndcg"]),
                textcoords="offset points", xytext=(5, 5), fontsize=9)

    colors = ["#2ca02c", "#9467bd", "#1f77b4"]
    for i, r in enumerate(results_list):
        label = r["strategy"].replace("_", " ")
        ax.scatter(r["dp_gap"], r["ndcg"], s=100, color=colors[i], zorder=5)
        ax.annotate(label, (r["dp_gap"], r["ndcg"]),
                    textcoords="offset points", xytext=(5, -12), fontsize=8)

    ax.set_xlabel("Demographic Parity Gap (lower = fairer)")
    ax.set_ylabel("NDCG@10 (higher = more accurate)")
    ax.set_title("Accuracy–Fairness Trade-off (Pareto Frontier)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def print_comparison_table(results_list, baseline):
    """Print a formatted comparison table to stdout."""
    header = f"{'Strategy':<40} {'DP Gap':>8} {'RMSE Gap':>10} {'RMSE':>8} {'NDCG@10':>9}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    b = baseline
    print(f"{'Baseline (no mitigation)':<40} {b['dp_gap']:>8.4f} {b['rmse_gap']:>10.4f} {b['overall_rmse']:>8.4f} {b['ndcg']:>9.4f}")
    for r in results_list:
        name = r["strategy"].replace("_", " ")[:40]
        print(f"{name:<40} {r['dp_gap']:>8.4f} {r['rmse_gap']:>10.4f} {r['overall_rmse']:>8.4f} {r['ndcg']:>9.4f}")
    print(sep)


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n=== Mitigation Comparison ===")
    print("[1/5] Loading data and training baseline CF...")
    ratings, users, items, gender_map, _ = load_all()
    algo, trainset, testset, predictions_baseline = train_baseline(ratings)
    top_n_baseline = get_top_n(predictions_baseline, n=10)
    pred_df_baseline = extract_stage5_predictions(predictions_baseline)

    baseline = {
        "strategy": "baseline",
        "dp_gap": float(demographic_parity_gap(top_n_baseline, gender_map)),
        "rmse_gap": float(prediction_rmse_gap(pred_df_baseline, gender_map)),
        "overall_rmse": float(rmse(pred_df_baseline)),
        "ndcg": float(ndcg_at_k(top_n_baseline, pred_df_baseline, k=10)),
    }
    print(f"  Baseline: DP gap={baseline['dp_gap']:.4f}  RMSE={baseline['overall_rmse']:.4f}  NDCG@10={baseline['ndcg']:.4f}")

    print("\n[2/5] Running Pre-processing strategy...")
    pre_result = run_preprocessing_strategy(ratings, gender_map)

    print("\n[3/5] Running In-processing (neighbourhood) strategy...")
    nbr_result = mitigation_neighbourhood.run(
        ratings=ratings, users=users, gender_map=gender_map,
        algo=algo, trainset=trainset, testset=testset,
        predictions_baseline=predictions_baseline
    )
    # Normalise field names for comparison
    nbr_summary = {
        "strategy": "in_processing_neighbourhood",
        "dp_gap": nbr_result["constrained_nbr_composition_gap"],
        "rmse_gap": nbr_result["constrained_pred_rmse_gap"],
        "overall_rmse": nbr_result["constrained_overall_rmse"],
        "ndcg": baseline["ndcg"],
    }

    print("\n[4/5] Running Post-processing (re-ranking) strategy...")
    post_result, _ = mitigation_post.run(
        ratings=ratings, users=users, gender_map=gender_map,
        predictions=predictions_baseline, top_n_original=top_n_baseline
    )
    post_summary = {
        "strategy": "post_processing_reranking",
        "dp_gap": post_result["post_dp_gap"],
        "rmse_gap": baseline["rmse_gap"],
        "overall_rmse": baseline["overall_rmse"],
        "ndcg": post_result["post_ndcg"],
    }

    results_list = [pre_result, nbr_summary, post_summary]

    print("\n[5/5] Generating comparison charts and table...")
    print_comparison_table(results_list, baseline)
    plot_grouped_bar(results_list, baseline)
    plot_pareto_frontier(results_list, baseline)

    combined = {
        "baseline": baseline,
        "strategies": results_list,
    }

    json_path = os.path.join(RESULTS_DIR, "mitigation_comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Saved: {json_path}")

    return combined

if __name__ == "__main__":
    run()
