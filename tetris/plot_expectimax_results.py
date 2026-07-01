"""Plot Expectimax experiment summaries for reports and presentations."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Expectimax experiment results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("expectimax_results.csv"),
        help="Input CSV produced by run_expectimax_experiments.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory where PNG figures are saved.",
    )
    return parser.parse_args()


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def to_float(row, key, default=0.0):
    value = row.get(key, "")
    if value in {"", None, "-"}:
        return default
    return float(value)


def style_axis(ax, title, ylabel):
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    ax.tick_params(axis="x", labelrotation=30, labelsize=9)
    ax.tick_params(axis="y", labelsize=10)


def plot_avg_reward(rows, output_dir):
    methods = [row["method"] for row in rows]
    rewards = [to_float(row, "average_reward") for row in rows]
    errors = [to_float(row, "reward_std") for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(methods, rewards, yerr=errors, capsize=5, color="#4C78A8")
    style_axis(ax, "Average Reward by Method", "Average Reward")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_avg_reward.png", dpi=220)
    plt.close(fig)


def plot_avg_steps(rows, output_dir):
    methods = [row["method"] for row in rows]
    steps = [to_float(row, "average_steps") for row in rows]
    errors = [to_float(row, "steps_std") for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(methods, steps, yerr=errors, capsize=5, color="#59A14F")
    style_axis(ax, "Average Steps by Method", "Average Steps")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_avg_steps.png", dpi=220)
    plt.close(fig)


def plot_reward_vs_time(rows, output_dir):
    times = [to_float(row, "average_decision_time") for row in rows]
    rewards = [to_float(row, "average_reward") for row in rows]
    methods = [row["method"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(times, rewards, s=70, color="#F28E2B")
    for time_value, reward, method in zip(times, rewards, methods):
        ax.annotate(
            method,
            (time_value, reward),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=9,
        )
    ax.set_title("Reward vs Decision Time", fontsize=14, pad=12)
    ax.set_xlabel("Average Decision Time (seconds)", fontsize=12)
    ax.set_ylabel("Average Reward", fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_reward_vs_time.png", dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    rows = read_rows(args.input)
    if not rows:
        raise ValueError(f"No rows found in {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_avg_reward(rows, args.output_dir)
    plot_avg_steps(rows, args.output_dir)
    plot_reward_vs_time(rows, args.output_dir)

    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
