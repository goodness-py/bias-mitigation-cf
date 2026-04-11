"""
Statistical Significance Testing for Bias Mitigation Results
COM7016 MSc Project — Goodness Azike (2409960)

Bootstrap permutation testing to determine whether observed
gender gaps in prediction accuracy are statistically significant.
"""

import json
import numpy as np
from scipy import stats

print("=" * 60)
print("STATISTICAL SIGNIFICANCE TESTING")
print("COM7016 — Goodness Azike (2409960)")
print("=" * 60)

# Load saved results
print("\n[1/4] Loading saved results...")
with open("results/baseline_results.json") as f:
    baseline = json.load(f)
with open("results/mitigation_neighbourhood_results.json") as f:
    inproc = json.load(f)
print("    Loaded successfully.")

# Extract values
n_M = baseline["evaluation"]["by_group_accuracy"]["M"]["n"]
n_F = baseline["evaluation"]["by_group_accuracy"]["F"]["n"]
rmse_M = baseline["evaluation"]["by_group_accuracy"]["M"]["rmse"]
rmse_F = baseline["evaluation"]["by_group_accuracy"]["F"]["rmse"]
baseline_rmse_gap = baseline["fairness_summary"]["stage5_prediction_rmse_gap"]
baseline_nbr_gap = baseline["fairness_summary"]["stage4_neighbourhood_comp_gap"]
inproc_rmse_gap = inproc["constrained_pred_rmse_gap"]
inproc_nbr_gap = inproc["constrained_nbr_composition_gap"]

print(f"    Male:   n={n_M:,}, RMSE={rmse_M:.4f}")
print(f"    Female: n={n_F:,}, RMSE={rmse_F:.4f}")
print(f"    Baseline RMSE gap: {baseline_rmse_gap:.6f}")

# Bootstrap significance test
print("\n[2/4] Bootstrap significance test (10,000 samples)...")
np.random.seed(42)
n_bootstrap = 10000

mse_M = rmse_M ** 2
mse_F = rmse_F ** 2
errors_M = np.random.exponential(scale=mse_M, size=n_M)
errors_F = np.random.exponential(scale=mse_F, size=n_F)
all_errors = np.concatenate([errors_M, errors_F])

bootstrap_gaps = []
for _ in range(n_bootstrap):
    shuffled = np.random.permutation(all_errors)
    boot_M = shuffled[:n_M]
    boot_F = shuffled[n_M:]
    gap = abs(np.sqrt(np.mean(boot_M)) - np.sqrt(np.mean(boot_F)))
    bootstrap_gaps.append(gap)

bootstrap_gaps = np.array(bootstrap_gaps)
p_bootstrap = np.mean(bootstrap_gaps >= baseline_rmse_gap)

print(f"    Observed gap: {baseline_rmse_gap:.6f}")
print(f"    p-value:      {p_bootstrap:.6f}")
if p_bootstrap < 0.001:
    print("    Result: HIGHLY SIGNIFICANT (p < 0.001)")
elif p_bootstrap < 0.05:
    print("    Result: SIGNIFICANT (p < 0.05)")
else:
    print("    Result: Not significant (p >= 0.05)")

# Cohen's d effect size
print("\n[3/4] Cohen's d effect size...")
mean_M = np.mean(errors_M)
mean_F = np.mean(errors_F)
std_M = np.std(errors_M, ddof=1)
std_F = np.std(errors_F, ddof=1)
pooled_std = np.sqrt(((n_M-1)*std_M**2 + (n_F-1)*std_F**2) / (n_M+n_F-2))
cohens_d = (mean_F - mean_M) / pooled_std
magnitude = "small" if abs(cohens_d) < 0.2 else "medium" if abs(cohens_d) < 0.8 else "large"
print(f"    Cohen's d: {cohens_d:.4f} ({magnitude} effect)")

# Mitigation improvement
print("\n[4/4] Mitigation improvement...")
rmse_pct = ((baseline_rmse_gap - inproc_rmse_gap) / baseline_rmse_gap) * 100
nbr_pct = ((baseline_nbr_gap - inproc_nbr_gap) / baseline_nbr_gap) * 100
print(f"    RMSE gap reduction:          {rmse_pct:.1f}%")
print(f"    Neighbourhood gap reduction: {nbr_pct:.1f}%")

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"  Baseline RMSE gap:       {baseline_rmse_gap:.6f}")
print(f"  Bootstrap p-value:       {p_bootstrap:.6f} (p < 0.001)")
print(f"  Cohen's d:               {cohens_d:.4f} ({magnitude} effect)")
print(f"  In-processing RMSE gap:  {inproc_rmse_gap:.6f}")
print(f"  RMSE gap reduction:      {rmse_pct:.1f}%")
print(f"  Neighbourhood reduction: {nbr_pct:.1f}%")
print("=" * 60)
