import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tensorboard.backend.event_processing import event_accumulator


RUNS_DIR = Path("runs")
OUTPUT_DIR = Path("docs")
LAMBDAS = [0.0, 0.1, 1.0]
SEEDS = [1, 2, 3, 4, 5]
MAX_STEP = 1_000_000
GRID_SIZE = 1000
EMA_WEIGHT = 0.8

RETURN_TAG = "charts/episodic_return"
VARIANCE_TAG = "diagnostics/long_advantage_variance"


def load_scalar(run_dir, tag):
    accumulator = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()

    if tag not in accumulator.Tags().get("scalars", []):
        raise KeyError(f"Missing scalar tag '{tag}' in {run_dir}")

    events = accumulator.Scalars(tag)
    steps = np.asarray([event.step for event in events], dtype=np.float64)
    values = np.asarray([event.value for event in events], dtype=np.float64)
    return average_duplicate_steps(steps, values)


def average_duplicate_steps(steps, values):
    order = np.argsort(steps)
    steps = steps[order]
    values = values[order]

    unique_steps, inverse = np.unique(steps, return_inverse=True)
    sums = np.zeros_like(unique_steps, dtype=np.float64)
    counts = np.zeros_like(unique_steps, dtype=np.float64)
    np.add.at(sums, inverse, values)
    np.add.at(counts, inverse, 1.0)
    return unique_steps, sums / counts


def interpolate_to_grid(steps, values, grid):
    if len(steps) < 2:
        raise ValueError("At least two scalar points are required for interpolation")
    return np.interp(grid, steps, values, left=values[0], right=values[-1])


def ema(values, weight=0.8):
    values = np.asarray(values, dtype=np.float64)
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for idx in range(1, len(values)):
        smoothed[idx] = weight * smoothed[idx - 1] + (1.0 - weight) * values[idx]
    return smoothed


def load_condition(lambda_aux, tag, grid):
    curves = []
    loaded_seeds = []

    for seed in SEEDS:
        run_dir = RUNS_DIR / f"ablation_variance_lambda_{lambda_aux}_seed_{seed}"
        if not run_dir.exists():
            print(f"[warning] Missing run directory: {run_dir}")
            continue

        try:
            steps, values = load_scalar(run_dir, tag)
            curve = interpolate_to_grid(steps, values, grid)
        except Exception as exc:
            print(f"[warning] Could not load {tag} from {run_dir}: {exc}")
            continue

        curves.append(curve)
        loaded_seeds.append(seed)

    if not curves:
        raise RuntimeError(f"No valid curves found for lambda_aux={lambda_aux}, tag={tag}")

    return np.asarray(curves, dtype=np.float64), loaded_seeds


def summarize_curves(curves, smooth=True):
    if smooth:
        curves = np.asarray([ema(curve, EMA_WEIGHT) for curve in curves], dtype=np.float64)
    mean = curves.mean(axis=0)
    std = curves.std(axis=0, ddof=1) if curves.shape[0] > 1 else np.zeros_like(mean)
    return mean, std


def apply_academic_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def plot_mean_std(ax, grid, mean, std, label, color):
    ax.plot(grid, mean, label=label, color=color, linewidth=2.2)
    ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)


def plot_return_comparison(grid, return_curves):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    labels = {
        0.0: r"Strict $\gamma=0.999$ Baseline",
        1.0: "Target Decoupling",
    }
    colors = {
        0.0: "#B03A2E",
        1.0: "#1F618D",
    }

    for lambda_aux in [0.0, 1.0]:
        mean, std = summarize_curves(return_curves[lambda_aux], smooth=True)
        plot_mean_std(ax, grid, mean, std, labels[lambda_aux], colors[lambda_aux])

    ax.axhline(200, color="black", linestyle="--", linewidth=1.0, alpha=0.65, label="Solved threshold")
    ax.set_title("Asymptotic Performance: Strict Long-Horizon Baseline vs. Target Decoupling")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episodic Return")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(6, 6))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "strict_baseline_return.png", bbox_inches="tight")
    plt.close(fig)


def plot_variance_reduction(grid, variance_curves):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    labels = {
        0.0: r"$\lambda_{\mathrm{aux}}=0.0$",
        0.1: r"$\lambda_{\mathrm{aux}}=0.1$",
        1.0: r"$\lambda_{\mathrm{aux}}=1.0$",
    }
    colors = {
        0.0: "#B03A2E",
        0.1: "#117A65",
        1.0: "#1F618D",
    }

    for lambda_aux in LAMBDAS:
        mean, std = summarize_curves(variance_curves[lambda_aux], smooth=True)
        plot_mean_std(ax, grid, mean, std, labels[lambda_aux], colors[lambda_aux])

    ax.set_title("Long-Horizon Advantage Variance Comparison")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(r"Variance of $\hat{A}_{\gamma=0.999}$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(6, 6))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "auxiliary_variance_reduction.png", bbox_inches="tight")
    plt.close(fig)


def run_welch_test(grid, return_curves):
    asymptotic_mask = grid >= 0.9 * MAX_STEP
    strict_means = return_curves[0.0][:, asymptotic_mask].mean(axis=1)
    decoupled_means = return_curves[1.0][:, asymptotic_mask].mean(axis=1)

    t_stat, p_value = stats.ttest_ind(decoupled_means, strict_means, equal_var=False)

    print("\nWelch's t-test on mean episodic return over last 10% of training")
    print(f"Strict gamma=0.999 baseline means (N={len(strict_means)}): {np.round(strict_means, 3)}")
    print(f"Target Decoupling means      (N={len(decoupled_means)}): {np.round(decoupled_means, 3)}")
    print(f"Strict baseline mean +/- std: {strict_means.mean():.3f} +/- {strict_means.std(ddof=1):.3f}")
    print(f"Target Decoupling mean +/- std: {decoupled_means.mean():.3f} +/- {decoupled_means.std(ddof=1):.3f}")
    print(f"T-statistic: {t_stat:.6f}")
    print(f"P-value:     {p_value:.6g}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    apply_academic_style()

    grid = np.linspace(0, MAX_STEP, GRID_SIZE)
    return_curves = {}
    variance_curves = {}

    for lambda_aux in LAMBDAS:
        return_curves[lambda_aux], return_seeds = load_condition(lambda_aux, RETURN_TAG, grid)
        variance_curves[lambda_aux], variance_seeds = load_condition(lambda_aux, VARIANCE_TAG, grid)
        print(
            f"lambda_aux={lambda_aux}: loaded returns for seeds {return_seeds}; "
            f"variance for seeds {variance_seeds}"
        )

    plot_return_comparison(grid, return_curves)
    plot_variance_reduction(grid, variance_curves)
    run_welch_test(grid, return_curves)

    print(f"\nSaved: {OUTPUT_DIR / 'strict_baseline_return.png'}")
    print(f"Saved: {OUTPUT_DIR / 'auxiliary_variance_reduction.png'}")


if __name__ == "__main__":
    main()
