"""Run Expectimax experiment sweeps and save the results as CSV."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

from expectimax_agent import ExpectimaxAgent
from main import make_env, run_episode


DEFAULT_CONFIGS = [
    {"depth": 1, "beam_width": 8, "chance_samples": 4, "chance_mode": "expected"},
    {"depth": 2, "beam_width": 3, "chance_samples": 2, "chance_mode": "expected"},
    {"depth": 2, "beam_width": 3, "chance_samples": 2, "chance_mode": "queue"},
]


def mean(values):
    return statistics.fmean(values) if values else 0.0


def parse_config(value):
    """Parse one config as depth,beam_width,chance_samples[,chance_mode]."""
    parts = [part.strip() for part in value.split(",")]
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError(
            "Config must be formatted as depth,beam_width,chance_samples[,chance_mode]"
        )
    depth, beam_width, chance_samples = map(int, parts[:3])
    chance_mode = parts[3] if len(parts) == 4 else "expected"
    if chance_mode not in {"expected", "queue"}:
        raise argparse.ArgumentTypeError("chance_mode must be expected or queue")
    return {
        "depth": depth,
        "beam_width": beam_width,
        "chance_samples": chance_samples,
        "chance_mode": chance_mode,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run Expectimax experiment sweeps.")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per config.")
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
        default=Path("expectimax_results.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--config",
        action="append",
        type=parse_config,
        help=(
            "Experiment config as depth,beam_width,chance_samples. "
            "Can be passed multiple times."
        ),
    )
    return parser.parse_args()


def summarize(config, results, elapsed_time):
    rewards = [result["total_reward"] for result in results]
    steps = [result["steps"] for result in results]
    decision_times = [result["average_decision_time"] for result in results]
    return {
        "depth": config["depth"],
        "beam_width": config["beam_width"],
        "chance_samples": config["chance_samples"],
        "chance_mode": config["chance_mode"],
        "episodes": len(results),
        "average_reward": mean(rewards),
        "best_reward": max(rewards, default=0.0),
        "average_steps": mean(steps),
        "average_decision_time": mean(decision_times),
        "total_wall_time": elapsed_time,
    }


def run_config(config, episodes, seed, render_mode, delay_ms, max_steps):
    env = make_env(render_mode)
    agent = ExpectimaxAgent(
        depth=config["depth"],
        beam_width=config["beam_width"],
        sample_chance=config["depth"] > 1,
        chance_samples=config["chance_samples"],
        chance_mode=config["chance_mode"],
    )

    start_time = time.perf_counter()
    try:
        results = []
        for episode in range(episodes):
            result = run_episode(
                env,
                agent,
                seed=seed + episode,
                delay_ms=delay_ms,
                max_steps=max_steps,
            )
            results.append(result)
        return summarize(config, results, time.perf_counter() - start_time)
    finally:
        env.close()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "depth",
        "beam_width",
        "chance_samples",
        "chance_mode",
        "episodes",
        "average_reward",
        "best_reward",
        "average_steps",
        "average_decision_time",
        "total_wall_time",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    configs = args.config or DEFAULT_CONFIGS
    summaries = []

    for index, config in enumerate(configs, start=1):
        print(
            f"[{index}/{len(configs)}] "
            f"depth={config['depth']}, "
            f"beam_width={config['beam_width']}, "
            f"chance_samples={config['chance_samples']}"
            f", chance_mode={config['chance_mode']}"
        )
        summary = run_config(
            config,
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

    write_csv(args.output, summaries)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
