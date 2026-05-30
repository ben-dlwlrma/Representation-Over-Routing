from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


RUNS_DIR = Path("runs")
DOCS_DIR = Path("docs")
ARXIV_DIR = Path("arXiv")
OUTPUT_NAME = "error_routing_diagnostic.png"

RUN_DIRS = [
    RUNS_DIR / "error_routing_seed_1",
    RUNS_DIR / "error_routing_seed_2",
    RUNS_DIR / "error_routing_seed_3",
    RUNS_DIR / "error_routing_seed_4",
    RUNS_DIR / "error_routing_seed_5",
]

MAX_STEP = 1_000_000
GRID_SIZE = 1000
EMA_WEIGHT = 0.85

RETURN_TAG = "charts/episodic_return"
WEIGHT_PREFIX = "weights/gamma_"
GAMMAS = ["0.5", "0.9", "0.99", "0.999"]


def load_event_accumulator(run_dir):
    accumulator = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()
    return accumulator


def load_scalar(run_dir, tag):
    accumulator = load_event_accumulator(run_dir)
    if tag not in accumulator.Tags().get("scalars", []):
        raise KeyError(f"Missing tag '{tag}' in {run_dir}")

    events = accumulator.Scalars(tag)
    steps = np.asarray([event.step for event in events], dtype=np.float64)
    values = np.asarray([event.value for event in events], dtype=np.float64)
    order = np.argsort(steps)
    steps = steps[order]
    values = values[order]

    unique_steps, inverse = np.unique(steps, return_inverse=True)
    sums = np.zeros_like(unique_steps, dtype=np.float64)
    counts = np.zeros_like(unique_steps, dtype=np.float64)
    np.add.at(sums, inverse, values)
    np.add.at(counts, inverse, 1.0)
    return unique_steps, sums / counts


def interpolate_curve(steps, values, grid):
    if len(steps) < 2:
        raise ValueError("Need at least two points for interpolation")
    return np.interp(grid, steps, values, left=values[0], right=values[-1])


def ema(values, weight=0.85):
    smoothed = np.empty_like(values, dtype=np.float64)
    smoothed[0] = values[0]
    for idx in range(1, len(values)):
        smoothed[idx] = weight * smoothed[idx - 1] + (1.0 - weight) * values[idx]
    return smoothed


def load_curves(run_dirs, tag, grid):
    curves = []
    for run_dir in run_dirs:
        steps, values = load_scalar(run_dir, tag)
        curves.append(interpolate_curve(steps, values, grid))

    return np.asarray(curves, dtype=np.float64)


def summarize(curves):
    curves = np.asarray([ema(curve, EMA_WEIGHT) for curve in curves], dtype=np.float64)
    mean = curves.mean(axis=0)
    std = curves.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
    return mean, std


def plot_mean_std(ax, grid, mean, std, color, label):
    ax.plot(grid, mean, color=color, linewidth=2.0, label=label)
    ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    ARXIV_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    run_dirs = RUN_DIRS
    missing = [run_dir for run_dir in run_dirs if not run_dir.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing expected error-routing TensorBoard runs:\n"
            + "\n".join(f"  {run_dir}" for run_dir in missing)
        )

    print("Using fixed error-routing diagnostic runs:")
    for run_dir in run_dirs:
        print(f"  {run_dir}")

    grid = np.linspace(0, MAX_STEP, GRID_SIZE)

    return_curves = load_curves(run_dirs, RETURN_TAG, grid)
    return_mean, return_std = summarize(return_curves)

    weight_summaries = {}
    for gamma in GAMMAS:
        tag = f"{WEIGHT_PREFIX}{gamma}"
        curves = load_curves(run_dirs, tag, grid)
        weight_summaries[gamma] = summarize(curves)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.3),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.0], "hspace": 0.16},
    )

    plot_mean_std(axes[0], grid, return_mean, return_std, "#6C3483", "episodic return")
    axes[0].set_title("Error-based temporal routing diagnostic")
    axes[0].set_ylabel("Episodic return")
    axes[0].grid(True, axis="y", alpha=0.25)

    colors = {
        "0.5": "#B03A2E",
        "0.9": "#D68910",
        "0.99": "#117A65",
        "0.999": "#1F618D",
    }
    for gamma in GAMMAS:
        if gamma not in weight_summaries:
            continue
        mean, std = weight_summaries[gamma]
        width = 2.4 if gamma == "0.5" else 1.4
        alpha = 0.20 if gamma == "0.5" else 0.08
        axes[1].plot(
            grid,
            mean,
            color=colors.get(gamma, "0.35"),
            linewidth=width,
            label=rf"$\gamma={gamma}$",
        )
        axes[1].fill_between(
            grid,
            mean - std,
            mean + std,
            color=colors.get(gamma, "0.35"),
            alpha=alpha,
            linewidth=0,
        )

    axes[1].set_ylabel("Routing weight")
    axes[1].set_xlabel("Environment steps")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(frameon=False, ncol=4, loc="upper right")

    for ax in axes:
        ax.ticklabel_format(style="sci", axis="x", scilimits=(6, 6))

    fig.text(
        0.5,
        0.01,
        "Solid lines show mean across 5 seeds; shaded regions show +/- 1 standard deviation. Curves are EMA-smoothed.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="0.25",
    )

    fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.11)
    docs_path = DOCS_DIR / OUTPUT_NAME
    arxiv_path = ARXIV_DIR / OUTPUT_NAME
    fig.savefig(docs_path, bbox_inches="tight")
    fig.savefig(arxiv_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {docs_path}")
    print(f"Saved: {arxiv_path}")


if __name__ == "__main__":
    main()
