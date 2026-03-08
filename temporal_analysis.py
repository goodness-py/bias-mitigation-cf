"""
temporal_analysis.py — Layer 3
Splits MovieLens-100K by timestamp into 4 chronological windows,
trains CF sequentially on each, runs pipeline decomposition, and measures
whether bias amplifies over time.

"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from data_loader import load_all
from baseline_cf import train_baseline, get_top_n
from stage_extraction import extract_all_stages
from pipeline_decomposition import measure_stage_bias, STAGE_LABELS

RESULTS_DIR = "results"
N_WINDOWS = 4

def split_by_timestamp(ratings_df, n_windows=N_WINDOWS):
    """
    Split ratings into n_windows chronological windows.
    Each window is cumulative (window k includes all ratings up to timestamp k).
    Returns list of DataFrames.
    """
    sorted_df = ratings_df.sort_values("timestamp").reset_index(drop=True)
    n = len(sorted_df)
    window_size = n // n_windows
    windows = []
    for i in range(1, n_windows + 1):
        end = i * window_size if i < n_windows else n
        windows.append(sorted_df.iloc[:end].copy())
    return windows


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n[1/3] Loading data...")
    ratings, users, items, gender_map, _ = load_all()

    print(f"[2/3] Splitting into {N_WINDOWS} temporal windows (cumulative)...")
    windows = split_by_timestamp(ratings, n_windows=N_WINDOWS)
    for i, w in enumerate(windows):
        ts_min = pd.to_datetime(w["timestamp"].min(), unit="s").strftime("%Y-%m")
        ts_max = pd.to_datetime(w["timestamp"].max(), unit="s").strftime("%Y-%m")
        print(f"  Window {i+1}: {len(w):,} ratings  ({ts_min} → {ts_max})")

    print("[3/3] Training CF and measuring bias at each window...")
    temporal_bias = {label: [] for label in STAGE_LABELS}
    window_labels = []

    for i, window_df in enumerate(windows):
        label = f"W{i+1}"
        window_labels.append(label)
        print(f"\n  --- Window {i+1} ---")

        try:
            algo, trainset, testset, predictions = train_baseline(window_df)
            stages = extract_all_stages(algo, trainset, predictions, window_df, gender_map)
            bias_scores = measure_stage_bias(stages, gender_map)

            for j, stage_label in enumerate(STAGE_LABELS):
                temporal_bias[stage_label].append(float(bias_scores[j]))

            print(f"  Bias scores: {[f'{s:.4f}' for s in bias_scores]}")
        except Exception as e:
            print(f"  Warning: window {i+1} failed ({e}), filling with NaN")
            for stage_label in STAGE_LABELS:
                temporal_bias[stage_label].append(float("nan"))

    # ── Compute amplification slopes (linear regression)
    slopes = {}
    x = np.arange(N_WINDOWS)
    for stage_label, values in temporal_bias.items():
        valid = [(xi, v) for xi, v in zip(x, values) if not np.isnan(v)]
        if len(valid) >= 2:
            xs, ys = zip(*valid)
            slope, intercept, r, p, se = stats.linregress(xs, ys)
            slopes[stage_label] = {"slope": float(slope), "r_squared": float(r**2), "p_value": float(p)}
        else:
            slopes[stage_label] = {"slope": float("nan"), "r_squared": float("nan"), "p_value": float("nan")}

    # ── Print slope summary
    print("\n  === Temporal Amplification Slopes ===")
    for label, s in slopes.items():
        direction = "↑ growing" if s["slope"] > 0 else "↓ shrinking"
        print(f"  {label:<22} slope={s['slope']:+.5f}  R²={s['r_squared']:.3f}  {direction}")

    # ── Plot: bias over time per stage
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10.colors
    for i, (stage_label, values) in enumerate(temporal_bias.items()):
        ax.plot(window_labels, values, marker="o", label=stage_label,
                color=colors[i % len(colors)])
    ax.set_xlabel("Temporal Window")
    ax.set_ylabel("Bias Score")
    ax.set_title("Bias Evolution Over Time by Pipeline Stage")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "temporal_bias_over_time.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Saved: {path}")

    # ── Save JSON
    result = {
        "window_labels": window_labels,
        "temporal_bias_by_stage": temporal_bias,
        "amplification_slopes": slopes,
    }
    json_path = os.path.join(RESULTS_DIR, "temporal_results.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {json_path}")

    return result


if __name__ == "__main__":
    run()
