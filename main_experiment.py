import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_all
from preprocessing import preprocess
from baseline_cf import train_baseline, get_top_n
from stage_extraction import extract_all_stages
from fairness_metrics import (
    input_activity_gap,
    similarity_bias,
    top_similar_composition_gap,
    neighbourhood_composition_gap,
    prediction_rmse_gap,
    demographic_parity_gap,
    equal_opportunity_gap,
    rmse_by_group,
)
from evaluation_metrics import full_evaluation

RESULTS_DIR = "results"

def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)

def _to_serialisable(obj):
    """Recursively convert numpy types to native Python for JSON."""
    if isinstance(obj, dict):
        return {str(k): _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def save_json(data, filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(_to_serialisable(data), f, indent=2)
    print(f"  Saved: {path}")

def plot_rmse_by_group(group_accuracy, filename="baseline_rmse_by_group.png"):
    groups = list(group_accuracy.keys())
    rmses = [group_accuracy[g]["rmse"] for g in groups]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(groups, rmses, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("RMSE")
    ax.set_title("Baseline RMSE by Gender Group")
    ax.set_ylim(0, max(rmses) * 1.2)
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_precision_by_group(group_ranking, filename="baseline_precision_by_group.png"):
    groups = list(group_ranking.keys())
    precisions = [group_ranking[g]["precision@k"] for g in groups]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(groups, precisions, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("Precision@10")
    ax.set_title("Baseline Precision@10 by Gender Group")
    ax.set_ylim(0, max(precisions) * 1.3 if max(precisions) > 0 else 1.0)
    for bar, val in zip(bars, precisions):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    

def run():
    ensure_results_dir()

    # ── 1. Load data
    print("\n[1/6] Loading data...")
    ratings, users, items, gender_map, _ = load_all()
    print(f"{len(ratings):,} ratings | {len(users)} users | {len(items)} items")

    # ── 2. Preprocess
    print("[2/6] Preprocessing...")
    enriched, input_stats = preprocess(ratings, users)
    print(f"M users: {input_stats['group_counts'].get('M', 0)} | "
          f"F users: {input_stats['group_counts'].get('F', 0)}")
    print(f"Activity gap: {input_stats['activity_gap']:.2f} ratings/user")

    # ── 3. Train baseline CF
    print("[3/6] Training baseline CF (user-based KNN, k=50)...")
    algo, trainset, testset, predictions = train_baseline(ratings)
    top_n = get_top_n(predictions, n=10)
    print(f"Train users: {trainset.n_users} | Test samples: {len(testset)}")

    # ── 4. Extract stage artefacts
    print("[4/6] Extracting pipeline stage artefacts...")
    stages = extract_all_stages(algo, trainset, predictions, ratings, gender_map)

    # ── 5. Compute fairness metrics at each stage
    print("[5/6] Computing stage-level fairness metrics...")
    pred_df = stages["stage5_predictions"]

    fairness_summary = {
        "stage1_input_activity_gap": input_activity_gap(stages["stage1_input"]),
        "stage2_similarity_bias": similarity_bias(stages["stage2_similarity"], gender_map),
        "stage3_top_similar_comp_gap": top_similar_composition_gap(stages["stage3_top_similar"], gender_map),
        "stage4_neighbourhood_comp_gap": neighbourhood_composition_gap(stages["stage4_neighbourhood"], gender_map),
        "stage5_prediction_rmse_gap": prediction_rmse_gap(pred_df, gender_map),
        "stage6_demographic_parity_gap": demographic_parity_gap(stages["stage6_ranking"], gender_map),
        "stage6_equal_opportunity_gap": equal_opportunity_gap(pred_df, gender_map),
        "rmse_by_group": rmse_by_group(pred_df, gender_map),
    }

    print("\n  === Stage-Level Fairness Summary ===")
    for metric, value in fairness_summary.items():
        if isinstance(value, dict):
            print(f"  {metric}: { {k: f'{v:.4f}' for k,v in value.items()} }")
        else:
            print(f"  {metric}: {value:.4f}")

    # ── 6. Compute accuracy metrics
    print("\n[6/6] Computing accuracy metrics...")
    eval_results = full_evaluation(pred_df, top_n, gender_map, k=10)

    print(f"\n Overall RMSE : {eval_results['overall']['rmse']:.4f}")
    print(f" Overall MAE  : {eval_results['overall']['mae']:.4f}")
    for g, metrics in eval_results["by_group_accuracy"].items():
        print(f"  [{g}] RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  n={metrics['n']}")
    for g, metrics in eval_results["by_group_ranking"].items():
        print(f"  [{g}] P@10={metrics['precision@k']:.4f}  R@10={metrics['recall@k']:.4f}  NDCG@10={metrics['ndcg@k']:.4f}")

    # ── Save results
    baseline_results = {
        "input_stats": input_stats,
        "fairness_summary": fairness_summary,
        "evaluation": eval_results,
    }
    save_json(baseline_results, "baseline_results.json")

    # ── Plots
    plot_rmse_by_group(eval_results["by_group_accuracy"])
    plot_precision_by_group(eval_results["by_group_ranking"])

    print("\nDone. Results saved to results/")
    return algo, trainset, testset, predictions, stages, gender_map


if __name__ == "__main__":
    run()
