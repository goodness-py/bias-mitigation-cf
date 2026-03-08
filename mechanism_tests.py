import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from data_loader import load_all
from baseline_cf import train_baseline
from stage_extraction import extract_all_stages

RESULTS_DIR = "results"


def test_neighbourhood_shrinkage(neighbourhood, gender_map):
    """
    Does minority group (F) receive smaller effective neighbourhoods?
    Compares neighbourhood size distribution between M and F users.
    Returns dict with means, Mann-Whitney U stat, and p-value.
    """
    sizes = {"M": [], "F": []}
    for uid, nbrs in neighbourhood.items():
        g = gender_map.get(uid)
        if g in ("M", "F"):
            sizes[g].append(len(nbrs))

    if not sizes["M"] or not sizes["F"]:
        return {"m_mean": 0, "f_mean": 0, "u_stat": None, "p_value": None, "significant": False}

    u_stat, p_value = stats.mannwhitneyu(sizes["M"], sizes["F"], alternative="two-sided")
    return {
        "m_mean": float(np.mean(sizes["M"])),
        "f_mean": float(np.mean(sizes["F"])),
        "m_std": float(np.std(sizes["M"])),
        "f_std": float(np.std(sizes["F"])),
        "u_stat": float(u_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "interpretation": (
            "F users receive significantly smaller neighbourhoods (shrinkage)" if p_value < 0.05
            else "No significant neighbourhood size difference"
        ),
    }


def test_exposure_imbalance(neighbourhood, gender_map):
    """
    Do minority (F) users appear less frequently in others' k-neighbourhoods?
    Counts how many times each user appears as a neighbour.
    Returns dict with means, Mann-Whitney U stat, and p-value.
    """
    exposure = {uid: 0 for uid in gender_map}
    for uid, nbrs in neighbourhood.items():
        for nbr in nbrs:
            if nbr in exposure:
                exposure[nbr] += 1

    exp_by_group = {"M": [], "F": []}
    for uid, count in exposure.items():
        g = gender_map.get(uid)
        if g in ("M", "F"):
            exp_by_group[g].append(count)

    if not exp_by_group["M"] or not exp_by_group["F"]:
        return {"m_mean": 0, "f_mean": 0, "u_stat": None, "p_value": None, "significant": False}

    u_stat, p_value = stats.mannwhitneyu(exp_by_group["M"], exp_by_group["F"], alternative="two-sided")
    return {
        "m_mean": float(np.mean(exp_by_group["M"])),
        "f_mean": float(np.mean(exp_by_group["F"])),
        "m_std": float(np.std(exp_by_group["M"])),
        "f_std": float(np.std(exp_by_group["F"])),
        "u_stat": float(u_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "interpretation": (
            "F users appear significantly less in others' neighbourhoods (exposure imbalance)" if p_value < 0.05
            else "No significant exposure imbalance detected"
        ),
    }


def test_confidence_variance(pred_df, gender_map):
    """
    Do predictions for minority (F) users show higher variance (less confident)?
    Compares absolute prediction error variance between M and F groups.
    Returns dict with means, Mann-Whitney U stat, and p-value.
    """
    pred_df = pred_df.copy()
    pred_df["gender"] = pred_df["user_id"].map(gender_map)
    pred_df["abs_error"] = (pred_df["est_rating"] - pred_df["true_rating"]).abs()

    errors_m = pred_df[pred_df["gender"] == "M"]["abs_error"].values
    errors_f = pred_df[pred_df["gender"] == "F"]["abs_error"].values

    if len(errors_m) == 0 or len(errors_f) == 0:
        return {"m_mean": 0, "f_mean": 0, "u_stat": None, "p_value": None, "significant": False}

    u_stat, p_value = stats.mannwhitneyu(errors_m, errors_f, alternative="two-sided")
    return {
        "m_mean_abs_error": float(np.mean(errors_m)),
        "f_mean_abs_error": float(np.mean(errors_f)),
        "m_std_abs_error": float(np.std(errors_m)),
        "f_std_abs_error": float(np.std(errors_f)),
        "u_stat": float(u_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "interpretation": (
            "F users have significantly different prediction error variance (confidence asymmetry)" if p_value < 0.05
            else "No significant difference in prediction confidence"
        ),
    }


def plot_mechanism_results(results, filename="mechanism_tests.png"):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Mechanism 1: neighbourhood size
    ax = axes[0]
    m1 = results["neighbourhood_shrinkage"]
    ax.bar(["M", "F"], [m1["m_mean"], m1["f_mean"]],
           yerr=[m1["m_std"], m1["f_std"]], capsize=5,
           color=["#4C72B0", "#DD8452"])
    ax.set_title(f"Neighbourhood Size\np={m1['p_value']:.4f} {'*' if m1['significant'] else 'ns'}")
    ax.set_ylabel("Mean neighbourhood size")

    # Mechanism 2: exposure
    ax2 = axes[1]
    m2 = results["exposure_imbalance"]
    ax2.bar(["M", "F"], [m2["m_mean"], m2["f_mean"]],
            yerr=[m2["m_std"], m2["f_std"]], capsize=5,
            color=["#4C72B0", "#DD8452"])
    ax2.set_title(f"Neighbourhood Exposure\np={m2['p_value']:.4f} {'*' if m2['significant'] else 'ns'}")
    ax2.set_ylabel("Mean times appearing as neighbour")

    # Mechanism 3: confidence variance
    ax3 = axes[2]
    m3 = results["confidence_variance"]
    ax3.bar(["M", "F"],
            [m3["m_mean_abs_error"], m3["f_mean_abs_error"]],
            yerr=[m3["m_std_abs_error"], m3["f_std_abs_error"]],
            capsize=5, color=["#4C72B0", "#DD8452"])
    ax3.set_title(f"Mean Absolute Error\np={m3['p_value']:.4f} {'*' if m3['significant'] else 'ns'}")
    ax3.set_ylabel("Mean |error|")

    plt.suptitle("Mechanism Tests (M vs F)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n[1/3] Loading data and training CF...")
    ratings, users, items, gender_map, _ = load_all()
    algo, trainset, testset, predictions = train_baseline(ratings)

    print("[2/3] Extracting stage artefacts...")
    stages = extract_all_stages(algo, trainset, predictions, ratings, gender_map)

    print("[3/3] Running mechanism tests...")
    neighbourhood = stages["stage4_neighbourhood"]
    pred_df = stages["stage5_predictions"]

    results = {
        "neighbourhood_shrinkage": test_neighbourhood_shrinkage(neighbourhood, gender_map),
        "exposure_imbalance": test_exposure_imbalance(neighbourhood, gender_map),
        "confidence_variance": test_confidence_variance(pred_df, gender_map),
    }

    print("\n  === Mechanism Test Results ===")
    for name, r in results.items():
        sig = "SIGNIFICANT *" if r.get("significant") else "not significant"
        print(f"  [{name}] p={r.get('p_value', 'N/A'):.4f}  ({sig})")
        print(f"    {r.get('interpretation', '')}")

    plot_mechanism_results(results)

    json_path = os.path.join(RESULTS_DIR, "mechanism_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {json_path}")

    return results


if __name__ == "__main__":
    run()
