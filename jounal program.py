import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Global configuration (journal-safe)
# -------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
    "figure.figsize": (6.5, 4),
    "savefig.dpi": 300
})

np.random.seed(42)

OUTPUT_FORMAT = "png"

# -------------------------------
# Utility functions
# -------------------------------
def expected_cost(fp, fn, c_fp=1.0, c_fn=10.0):
    return c_fp * fp + c_fn * fn

def simulate_fp_fn(threshold, policy):
    """
    Simulation placeholder.
    Replace with real FP/FN logs when available.
    """
    base_fp = max(0, int(120 * (1 - threshold)))
    base_fn = max(0, int(90 * threshold))

    if policy == "proposed":
        return int(0.8 * base_fp), int(0.6 * base_fn)
    if policy == "baseline":
        return base_fp, base_fn
    if policy == "conservative":
        return int(1.3 * base_fp), int(0.4 * base_fn)

# -------------------------------
# FIGURE 2: Expected Cost vs Threshold
# -------------------------------
thresholds = np.linspace(0.1, 0.9, 30)

cost_proposed, cost_baseline, cost_conservative = [], [], []

for t in thresholds:
    fp, fn = simulate_fp_fn(t, "proposed")
    cost_proposed.append(expected_cost(fp, fn))

    fp, fn = simulate_fp_fn(t, "baseline")
    cost_baseline.append(expected_cost(fp, fn))

    fp, fn = simulate_fp_fn(t, "conservative")
    cost_conservative.append(expected_cost(fp, fn))

# Normalize costs (baseline mean = 1.0)
baseline_mean = np.mean(cost_baseline)
cost_proposed = np.array(cost_proposed) / baseline_mean
cost_baseline = np.array(cost_baseline) / baseline_mean
cost_conservative = np.array(cost_conservative) / baseline_mean

plt.figure()
plt.plot(thresholds, cost_proposed, label="Proposed Policy")
plt.plot(thresholds, cost_baseline, linestyle="--", label="Cost-Agnostic")
plt.plot(thresholds, cost_conservative, linestyle=":", label="Conservative")
plt.xlabel("Decision Threshold (CMTI)")
plt.ylabel("Normalized Expected Cost")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"fig2_expected_cost_vs_threshold.{OUTPUT_FORMAT}")
plt.close()

# -------------------------------
# FIGURE 3: Recall vs Inspection Rate
# -------------------------------
inspection_rate = []
recall = []

TOTAL_SEQUENCES = 1000
POSITIVE_RATE = 0.2  # simulated inspection-relevant rate

for t in thresholds:
    fp, fn = simulate_fp_fn(t, "proposed")
    tp = TOTAL_SEQUENCES * POSITIVE_RATE - fn
    recall.append(tp / (tp + fn))
    inspection_rate.append((tp + fp) / TOTAL_SEQUENCES)

plt.figure()
plt.plot(inspection_rate, recall, label="Proposed Policy")
plt.xlabel("Inspection Rate")
plt.ylabel("Recall")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"fig3_recall_vs_inspection_rate.{OUTPUT_FORMAT}")
plt.close()

# -------------------------------
# FIGURE 4: Expected Cost vs Cost Ratio
# -------------------------------
cost_ratios = np.arange(5, 21)
cost_curve = []

for r in cost_ratios:
    fp, fn = simulate_fp_fn(0.5, "proposed")
    cost_curve.append(expected_cost(fp, fn, c_fp=1.0, c_fn=r))

cost_curve = np.array(cost_curve) / np.mean(cost_curve)

plt.figure()
plt.plot(cost_ratios, cost_curve, label="Proposed Policy")
plt.xlabel("Cost Ratio (False Negative : False Positive)")
plt.ylabel("Normalized Expected Cost")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"fig4_cost_vs_cost_ratio.{OUTPUT_FORMAT}")
plt.close()

# -------------------------------
# FIGURE 5: Mean vs 95th Percentile Cost (Tail Risk)
# -------------------------------
seq_costs_proposed = np.random.gamma(shape=2.0, scale=15.0, size=1000)
seq_costs_baseline = np.random.gamma(shape=2.5, scale=18.0, size=1000)

mean_vals = [
    np.mean(seq_costs_proposed),
    np.mean(seq_costs_baseline)
]

p95_vals = [
    np.percentile(seq_costs_proposed, 95),
    np.percentile(seq_costs_baseline, 95)
]

labels = ["Proposed", "Baseline"]
x = np.arange(len(labels))

plt.figure()
plt.bar(x - 0.15, mean_vals, 0.3, label="Mean Cost")
plt.bar(x + 0.15, p95_vals, 0.3, label="95th Percentile Cost")
plt.xticks(x, labels)
plt.ylabel("Sequence-Level Cost (Normalized Units)")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"fig5_tail_risk_analysis.{OUTPUT_FORMAT}")
plt.close()

# -------------------------------
# FIGURE 6: Threshold Sensitivity
# -------------------------------
base_threshold = 0.5
deltas = np.linspace(-0.05, 0.05, 15)
sensitivity_costs = []

for d in deltas:
    fp, fn = simulate_fp_fn(base_threshold + d, "proposed")
    sensitivity_costs.append(expected_cost(fp, fn))

plt.figure()
plt.plot(deltas, sensitivity_costs)
plt.xlabel("Threshold Perturbation")
plt.ylabel("Expected Cost (Normalized Units)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"fig6_threshold_sensitivity.{OUTPUT_FORMAT}")
plt.close()

print("All figures generated successfully as image files.")
