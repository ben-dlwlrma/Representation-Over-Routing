from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import seaborn as sns
from scipy import stats
from tensorboard.backend.event_processing import event_accumulator


RUNS_DIR = Path("runs")
OUTPUT_DIR = Path("docs")
OUTPUT_PATH = OUTPUT_DIR / "salvage_return_boxplot.png"
RETURN_TAG = "charts/episodic_return"
LAMBDAS = [0.0, 1.0]
SEEDS = [1, 2, 3, 4, 5]
MAX_STEP = 1_000_000
ASYMPTOTIC_START = int(0.9 * MAX_STEP)

LABELS = {
    0.0: "Strict Baseline\n($\\gamma=0.999$, $\\lambda=0.0$)",
    1.0: "Target Decoupling\n($\\lambda=1.0$)",
}


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
    order = np.argsort(steps)
    return steps[order], values[order]


def extract_asymptotic_seed_mean(lambda_aux, seed):
    run_dir = RUNS_DIR / f"ablation_variance_lambda_{lambda_aux}_seed_{seed}"
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run directory: {run_dir}")

    steps, returns = load_scalar(run_dir, RETURN_TAG)
    mask = (steps >= ASYMPTOTIC_START) & (steps <= MAX_STEP)
    if not np.any(mask):
        raise ValueError(
            f"No '{RETURN_TAG}' points in [{ASYMPTOTIC_START}, {MAX_STEP}] for {run_dir}"
        )

    return float(np.mean(returns[mask]))


def collect_data():
    rows = []
    grouped = {}

    for lambda_aux in LAMBDAS:
        values = []
        for seed in SEEDS:
            seed_mean = extract_asymptotic_seed_mean(lambda_aux, seed)
            values.append(seed_mean)
            rows.append(
                {
                    "condition": LABELS[lambda_aux],
                    "lambda_aux": lambda_aux,
                    "seed": seed,
                    "asymptotic_return": seed_mean,
                }
            )
        grouped[lambda_aux] = np.asarray(values, dtype=np.float64)

    return rows, grouped


def format_stats(values):
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "worst": float(np.min(values)),
    }


def make_plot(rows, grouped):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        rc={
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.8,
            "axes.linewidth": 0.9,
            "grid.linewidth": 0.55,
        },
    )

    conditions = [LABELS[0.0], LABELS[1.0]]
    palette = {
        LABELS[0.0]: "#B03A2E",
        LABELS[1.0]: "#1F618D",
    }
    plot_data = {
        "condition": [row["condition"] for row in rows],
        "asymptotic_return": [row["asymptotic_return"] for row in rows],
    }

    fig = plt.figure(figsize=(6.1, 4.55))
    gs = GridSpec(2, 1, height_ratios=[4.3, 1.08], hspace=0.03, figure=fig)
    ax = fig.add_subplot(gs[0])
    stat_ax = fig.add_subplot(gs[1])

    stats_by_lambda = {lambda_aux: format_stats(values) for lambda_aux, values in grouped.items()}

    y_values = np.asarray([row["asymptotic_return"] for row in rows], dtype=np.float64)
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    y_span = max(y_max - y_min, 1.0)
    ax.set_ylim(y_min - 0.07 * y_span, y_max + 0.12 * y_span)
    ax.set_xlim(-0.55, 1.55)

    rng = np.random.default_rng(7)
    for x_pos, lambda_aux in enumerate(LAMBDAS):
        values = grouped[lambda_aux]
        summary = stats_by_lambda[lambda_aux]
        color = palette[LABELS[lambda_aux]]
        point_center = x_pos - 0.07
        summary_center = x_pos + 0.15
        jitter = rng.uniform(-0.045, 0.045, size=len(values))

        ax.scatter(
            np.full_like(values, point_center, dtype=np.float64) + jitter,
            values,
            s=55,
            color=color,
            edgecolor="black",
            linewidth=0.65,
            alpha=0.95,
            zorder=5,
        )
        ax.errorbar(
            summary_center,
            summary["mean"],
            yerr=summary["std"],
            fmt="D",
            color=color,
            ecolor=color,
            elinewidth=1.05,
            capsize=5.5,
            capthick=1.05,
            markersize=4.6,
            markerfacecolor="white",
            markeredgewidth=1.05,
            alpha=0.82,
            zorder=8,
        )
        ax.hlines(
            summary["worst"],
            x_pos - 0.23,
            x_pos + 0.23,
            colors=color,
            linewidth=1.6,
            alpha=0.82,
            zorder=4,
        )

    t_stat, p_value = stats.ttest_ind(grouped[1.0], grouped[0.0], equal_var=False)
    fig.suptitle(
        "Final-return reliability across seeds",
        y=0.985,
        fontsize=11.0,
        fontweight="regular",
    )
    fig.text(
        0.5,
        0.935,
        f"N=5 seeds, Welch p={p_value:.3f}",
        ha="center",
        va="center",
        fontsize=8.4,
        color="0.25",
    )

    ax.set_xlabel("")
    ax.set_ylabel("Mean episodic return over final 10%")
    ax.set_xticks([0, 1])
    ax.grid(axis="y", alpha=0.26)
    ax.grid(axis="x", visible=False)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", pad=5)
    ax.tick_params(axis="y", length=3, width=0.8)
    sns.despine(ax=ax)

    stat_ax.set_xlim(0, 1)
    stat_ax.set_ylim(0, 1)
    stat_ax.axis("off")

    x_positions = {0.0: 0.25, 1.0: 0.75}
    display_names = {
        0.0: "Strict Baseline",
        1.0: "Target Decoupling",
    }
    parameter_text = {
        0.0: "$\\gamma=0.999,\\ \\lambda=0.0$",
        1.0: "$\\lambda=1.0$",
    }
    for lambda_aux in LAMBDAS:
        summary = stats_by_lambda[lambda_aux]
        stat_ax.text(
            x_positions[lambda_aux],
            0.75,
            display_names[lambda_aux],
            ha="center",
            va="center",
            fontsize=8.7,
            color="0.15",
        )
        stat_ax.text(
            x_positions[lambda_aux],
            0.50,
            parameter_text[lambda_aux],
            ha="center",
            va="center",
            fontsize=8.4,
            color="0.15",
        )
        stat_ax.text(
            x_positions[lambda_aux],
            0.25,
            f"mean {summary['mean']:.1f}   std {summary['std']:.1f}   worst {summary['worst']:.1f}",
            ha="center",
            va="center",
            fontsize=8.3,
            color="0.20",
        )

    fig.subplots_adjust(left=0.12, right=0.985, top=0.89, bottom=0.08)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return t_stat, p_value, stats_by_lambda


def main():
    rows, grouped = collect_data()
    t_stat, p_value, stats_by_lambda = make_plot(rows, grouped)

    for lambda_aux in LAMBDAS:
        values = grouped[lambda_aux]
        summary = stats_by_lambda[lambda_aux]
        print(f"lambda_aux={lambda_aux}: {np.round(values, 3)}")
        print(
            f"  mean={summary['mean']:.3f}, std={summary['std']:.3f}, "
            f"worst={summary['worst']:.3f}"
        )

    print(f"Welch t-statistic={t_stat:.6f}, p-value={p_value:.6g}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
