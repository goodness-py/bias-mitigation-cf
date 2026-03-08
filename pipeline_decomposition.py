import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_loader import load_all
from baseline_cf import train_baseline, get_top_n
from stage_extraction import extract_all_stages
from fairness_metrics import (
    input_activity_gap,
    similarity_bias,
    top_similar_composition_gap,
    neighbourhood_composition_gap,
    prediction_rmse_gap,
    demographic_parity_gap,
)

RESULTS_DIR = "results"
STAGE_LABELS = [
    "S1: Input",
    "S2: Similarity",
    "S3: Top-Similar",
    "S4: Neighbourhood",
    "S5: Prediction",
    "S6: Ranking",
]

def measure_stage_bias(stages, gender_map):
    """
    Compute a single bias score at each of the 6 stages.
    All scores are normalised to a comparable scale (absolute gap value).

    Returns list of 6 floats [B1, B2, B3, B4, B5, B6].
    """
    pred_df = stages["stage5_predictions"]

    scores = [
        input_activity_gap(stages["stage1_input"]),
        similarity_bias(stages["stage2_similarity"], gender_map),
        top_similar_composition_gap(stages["stage3_top_similar"], gender_map),
        neighbourhood_composition_gap(stages["stage4_neighbourhood"], gender_map),
        prediction_rmse_gap(pred_df, gender_map),
        demographic_parity_gap(stages["stage6_ranking"], gender_map),
    ]
    return scores


def compute_delta_amplification(bias_scores):
    """
    Compute Δᵢ = Bᵢ - Bᵢ₋₁ for stages 2–6.
    Returns list of 5 deltas.
    """
    return [bias_scores[i] - bias_scores[i - 1] for i in range(1, len(bias_scores))]


def identify_bottleneck(bias_scores):
    """Return (stage_index, stage_label) of the stage with maximum bias score."""
    idx = int(np.argmax(bias_scores))
    return idx, STAGE_LABELS[idx]


def plot_waterfall(bias_scores, delta_amplification, filename="decomposition_waterfall.png"):
    """
    Waterfall chart showing bias accumulation across pipeline stages.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: bias score per stage
    ax = axes[0]
    colors = ["#4C72B0" if s >= 0 else "#DD8452" for s in bias_scores]
    bars = ax.bar(STAGE_LABELS, bias_scores, color=colors, edgecolor="white")
    ax.set_title("Bias Score by Pipeline Stage")
    ax.set_ylabel("Bias Score (gap)")
    ax.set_xticklabels(STAGE_LABELS, rotation=30, ha="right")
    for bar, val in zip(bars, bias_scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(bias_scores) * 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    # Right: delta amplification
    ax2 = axes[1]
    delta_labels = [f"Δ{i+2}" for i in range(len(delta_amplification))]
    colors2 = ["#2ca02c" if d >= 0 else "#d62728" for d in delta_amplification]
    bars2 = ax2.bar(delta_labels, delta_amplification, color=colors2, edgecolor="white")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_title("Delta Amplification Between Stages")
    ax2.set_ylabel("Δ Bias")
    for bar, val in zip(bars2, delta_amplification):
        y = bar.get_height() if val >= 0 else bar.get_height() - abs(val) * 0.15
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (max(delta_amplification, default=0) * 0.02),
                 f"{val:+.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

def run(ratings=None, users=None, gender_map=None, algo=None,
        trainset=None, predictions=None, stages=None):
    """
    Run full pipeline decomposition.
    If model artefacts are provided, skips re-training (used by temporal_analysis).
    Returns result dict.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if ratings is None:
        print("\n[1/3] Loading data and training baseline CF...")
        ratings, users, items, gender_map, _ = load_all()
        algo, trainset, testset, predictions = train_baseline(ratings)

    if stages is None:
        print("[2/3] Extracting pipeline stage artefacts...")
        stages = extract_all_stages(algo, trainset, predictions, ratings, gender_map)

    print("[3/3] Computing stage-level bias and delta amplification...")
    bias_scores = measure_stage_bias(stages, gender_map)
    deltas = compute_delta_amplification(bias_scores)
    bottleneck_idx, bottleneck_label = identify_bottleneck(bias_scores)

    # Print summary
    print("\n  === Pipeline Decomposition Results ===")
    for label, score in zip(STAGE_LABELS, bias_scores):
        marker = " ← BOTTLENECK" if label == bottleneck_label else ""
        print(f"  {label:<22} bias={score:.4f}{marker}")
    print()
    for i, d in enumerate(deltas):
        print(f"Δ{i+2} ({STAGE_LABELS[i]} → {STAGE_LABELS[i+1]}): {d:+.4f}")
    print(f"\n  Bottleneck: {bottleneck_label} (bias={bias_scores[bottleneck_idx]:.4f})")

    result = {
        "stage_labels": STAGE_LABELS,
        "stage_bias_scores": dict(zip(STAGE_LABELS, bias_scores)),
        "delta_amplification": dict(zip(
            [f"delta_{i+2}" for i in range(len(deltas))],
            deltas
        )),
        "bottleneck_stage": bottleneck_label,
        "bottleneck_bias": float(bias_scores[bottleneck_idx]),
    }

    # Plot and save
    plot_waterfall(bias_scores, deltas)

    path = os.path.join(RESULTS_DIR, "decomposition_results.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {path}")

    return result

if __name__ == "__main__":
    run()
