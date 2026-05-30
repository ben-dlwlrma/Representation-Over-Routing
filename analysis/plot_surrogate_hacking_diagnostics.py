from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


RUNS_DIR = Path("runs")
OUTPUT_DIR = Path("docs")
OUTPUT_PATH = OUTPUT_DIR / "surrogate_hacking_diagnostic_triad.png"

SEEDS = [1, 2, 3, 4, 5]
MAX_STEP = 1_000_000
GRID_SIZE = 1000
EMA_WEIGHT = 0.85

TAGS = {
    "return": "charts/episodic_return",
    "hack_rate": "diagnostics/hack_rate",
    "attention_entropy": "diagnostics/attention_entropy",
}


def load_scalar(run_dir, tag):
    accumulator = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()

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
        raise ValueError("Need at least two scalar points for interpolation")
    return np.interp(grid, steps, values, left=values[0], right=values[-1])


def ema(values, weight=0.85):
    values = np.asarray(values, dtype=np.float64)
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i - 1] + (1.0 - weight) * values[i]
    return smoothed


def load_all_curves(tag, grid):
    curves = []
    loaded = []

    for seed in SEEDS:
        run_dir = RUNS_DIR / f"surrogate_hacking_seed_{seed}"
        if not run_dir.exists():
            print(f"[warning] missing run directory: {run_dir}")
            continue

        try:
            steps, values = load_scalar(run_dir, tag)
            curve = interpolate_curve(steps, values, grid)
        except Exception as exc:
            print(f"[warning] failed loading {tag} from {run_dir}: {exc}")
            continue

        curves.append(curve)
        loaded.append(seed)

    if not curves:
        raise RuntimeError(f"No curves loaded for tag {tag}")

    print(f"Loaded {tag} for seeds {loaded}")
    return np.asarray(curves, dtype=np.float64)


def summarize(curves, smooth=True):
    if smooth:
        curves = np.asarray([ema(curve, EMA_WEIGHT) for curve in curves], dtype=np.float64)
    mean = curves.mean(axis=0)
    std = curves.std(axis=0, ddof=1) if curves.shape[0] > 1 else np.zeros_like(mean)
    return mean, std


def plot_mean_std(ax, grid, mean, std, color):
    ax.plot(grid, mean, color=color, linewidth=2.0)
    ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    grid = np.linspace(0, MAX_STEP, GRID_SIZE)

    return_curves = load_all_curves(TAGS["return"], grid)
    hack_curves = load_all_curves(TAGS["hack_rate"], grid)
    entropy_curves = load_all_curves(TAGS["attention_entropy"], grid)

    return_mean, return_std = summarize(return_curves, smooth=True)
    hack_mean, hack_std = summarize(hack_curves, smooth=True)
    entropy_mean, entropy_std = summarize(entropy_curves, smooth=True)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0, 1.0], "hspace": 0.16},
    )

    color = "#8E44AD"

    plot_mean_std(axes[0], grid, return_mean, return_std, color)
    axes[0].set_ylabel("Episodic return")
    axes[0].set_title("Dynamic attention routing: performance and surrogate-exploitation diagnostics")
    axes[0].grid(True, axis="y", alpha=0.25)

    plot_mean_std(axes[1], grid, hack_mean, hack_std, color)
    axes[1].axhline(0.25, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[1].text(
        0.995,
        0.25,
        " random baseline",
        transform=axes[1].get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="0.2",
    )
    axes[1].set_ylabel("HackRate")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(True, axis="y", alpha=0.25)

    max_entropy = np.log(4.0)
    plot_mean_std(axes[2], grid, entropy_mean, entropy_std, color)
    axes[2].axhline(max_entropy, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[2].text(
        0.995,
        max_entropy,
        r" max entropy $\log 4$",
        transform=axes[2].get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="0.2",
    )
    axes[2].set_ylabel("Attention entropy")
    axes[2].set_xlabel("Environment steps")
    axes[2].set_ylim(0.0, max_entropy * 1.08)
    axes[2].grid(True, axis="y", alpha=0.25)

    for ax in axes:
        ax.ticklabel_format(style="sci", axis="x", scilimits=(6, 6))

    fig.text(
        0.5,
        0.01,
        "Solid line: mean across 5 seeds. Shaded region: +/- 1 standard deviation. Curves are EMA-smoothed.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="0.25",
    )

    fig.subplots_adjust(left=0.12, right=0.98, top=0.94, bottom=0.09)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
