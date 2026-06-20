"""Sweep depth-1 Expectimax heuristic weights and save results as CSV."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

from expectimax_agent import ExpectimaxAgent, HeuristicWeights
from main import make_env, run_episode


BASE_WEIGHTS = HeuristicWeights(
    reward=1.0,
    lines_cleared=8.0,
    complete_lines=2.0,
    aggregate_height=-0.55,
    max_height=-0.35,
    holes=-1.15,
    bumpiness=-0.35,
    wells=-0.18,
    max_well_depth=0.0,
    well_depth_squared=0.0,
    row_transitions=0.0,
    column_transitions=0.0,
    blockade_cells=0.0,
    height_std=-0.20,
    game_over=-1000.0,
)

WEIGHT_PRESETS = [
    ("baseline", BASE_WEIGHTS),
    ("holes_x1_25", replace(BASE_WEIGHTS, holes=BASE_WEIGHTS.holes * 1.25)),
    ("holes_x1_50", replace(BASE_WEIGHTS, holes=BASE_WEIGHTS.holes * 1.50)),
    (
        "height_x1_25",
        replace(
            BASE_WEIGHTS,
            aggregate_height=BASE_WEIGHTS.aggregate_height * 1.25,
            max_height=BASE_WEIGHTS.max_height * 1.25,
        ),
    ),
    ("bumpiness_x1_50", replace(BASE_WEIGHTS, bumpiness=BASE_WEIGHTS.bumpiness * 1.50)),
    ("wells_x5", replace(BASE_WEIGHTS, holes=-1.4375, wells=-0.90)),
    (
        "max_well_guard",
        replace(
            BASE_WEIGHTS,
            holes=-1.4375,
            wells=-0.90,
            max_well_depth=-0.90,
            well_depth_squared=0.0,
        ),
    ),
    (
        "transition_guard",
        replace(
            BASE_WEIGHTS,
            holes=-1.4375,
            wells=-0.90,
            max_well_depth=-0.90,
            row_transitions=-0.12,
            column_transitions=-0.08,
            blockade_cells=-0.35,
        ),
    ),
    ("lines_x1_50", replace(BASE_WEIGHTS, lines_cleared=BASE_WEIGHTS.lines_cleared * 1.50)),
    (
        "survival_balanced",
        replace(
            BASE_WEIGHTS,
            holes=-1.60,
            aggregate_height=-0.70,
            max_height=-0.45,
            bumpiness=-0.45,
        ),
    ),
    (
        "line_clear_aggressive",
        replace(
            BASE_WEIGHTS,
            lines_cleared=14.0,
            complete_lines=4.0,
            holes=-1.25,
        ),
    ),
]


def mean(values):
    return statistics.fmean(values) if values else 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Tune depth-1 Expectimax weights.")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per preset.")
    parser.add_argument("--seed", type=int, default=42, help="First environment seed.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional episode limit.")
    parser.add_argument(
        "--render-mode",
        default="ansi",
        choices=["human", "ansi", "rgb_array"],
        help="Gymnasium render mode.",
    )
    parser.add_argument("--delay-ms", type=int, default=0, help="Delay between steps.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("expectimax_weight_results.csv"),
        help="CSV output path.",
    )
    return parser.parse_args()


def summarize(name, weights, results, elapsed_time):
    rewards = [result["total_reward"] for result in results]
    steps = [result["steps"] for result in results]
    decision_times = [result["average_decision_time"] for result in results]
    row = {
        "preset": name,
        "episodes": len(results),
        "average_reward": mean(rewards),
        "best_reward": max(rewards, default=0.0),
        "average_steps": mean(steps),
        "average_decision_time": mean(decision_times),
        "total_wall_time": elapsed_time,
    }
    row.update(asdict(weights))
    return row


def run_preset(name, weights, episodes, seed, render_mode, delay_ms, max_steps):
    env = make_env(render_mode)
    agent = ExpectimaxAgent(depth=1, weights=weights)
    start_time = time.perf_counter()
    try:
        results = []
        for episode in range(episodes):
            results.append(
                run_episode(
                    env,
                    agent,
                    seed=seed + episode,
                    delay_ms=delay_ms,
                    max_steps=max_steps,
                )
            )
        return summarize(name, weights, results, time.perf_counter() - start_time)
    finally:
        env.close()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "preset",
        "episodes",
        "average_reward",
        "best_reward",
        "average_steps",
        "average_decision_time",
        "total_wall_time",
        "reward",
        "lines_cleared",
        "complete_lines",
        "aggregate_height",
        "max_height",
        "holes",
        "bumpiness",
        "wells",
        "max_well_depth",
        "well_depth_squared",
        "row_transitions",
        "column_transitions",
        "blockade_cells",
        "height_std",
        "game_over",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    summaries = []

    for index, (name, weights) in enumerate(WEIGHT_PRESETS, start=1):
        print(f"[{index}/{len(WEIGHT_PRESETS)}] preset={name}")
        summary = run_preset(
            name,
            weights,
            episodes=args.episodes,
            seed=args.seed,
            render_mode=args.render_mode,
            delay_ms=args.delay_ms,
            max_steps=args.max_steps,
        )
        summaries.append(summary)
        print(
            f"  avg_reward={summary['average_reward']:.2f}, "
            f"best_reward={summary['best_reward']:.2f}, "
            f"avg_steps={summary['average_steps']:.2f}, "
            f"avg_decision={summary['average_decision_time']:.4f}s"
        )

    summaries.sort(key=lambda row: row["average_reward"], reverse=True)
    write_csv(args.output, summaries)
    print(f"\nSaved sorted results to {args.output}")


if __name__ == "__main__":
    main()
