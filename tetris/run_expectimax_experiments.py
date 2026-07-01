"""Run Expectimax experiment sweeps and save the results as CSV."""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import time
from pathlib import Path

import numpy as np

from expectimax_agent import ExpectimaxAgent
from main import make_env, run_episode


DEFAULT_CONFIGS = [
    {
        "method": "Random Agent",
        "agent": "random",
        "depth": None,
        "beam_width": None,
        "chance_samples": None,
        "chance_mode": None,
        "heuristic_mode": None,
    },
    {
        "method": "Depth-1 Expectimax",
        "agent": "expectimax",
        "depth": 1,
        "beam_width": 8,
        "chance_samples": 4,
        "chance_mode": "expected",
        "heuristic_mode": "auto",
    },
    {
        "method": "Depth-2 Expected",
        "agent": "expectimax",
        "depth": 2,
        "beam_width": 3,
        "chance_samples": 2,
        "chance_mode": "expected",
        "heuristic_mode": "auto",
    },
    {
        "method": "Depth-2 Queue",
        "agent": "expectimax",
        "depth": 2,
        "beam_width": 3,
        "chance_samples": 2,
        "chance_mode": "queue",
        "heuristic_mode": "auto",
    },
    {
        "method": "Depth-2 Base Heuristic",
        "agent": "expectimax",
        "depth": 2,
        "beam_width": 3,
        "chance_samples": 2,
        "chance_mode": "queue",
        "heuristic_mode": "base",
    },
    {
        "method": "Depth-2 Optimized",
        "agent": "expectimax",
        "depth": 2,
        "beam_width": 5,
        "chance_samples": 7,
        "chance_mode": "expected",
        "heuristic_mode": "auto",
    },
]


class RandomAgent:
    def __init__(self, rng):
        self.rng = rng

    def select_action(self, env, observation=None, info=None):
        mask = info.get("action_mask") if isinstance(info, dict) else None
        if mask is None:
            mask = getattr(env, "legal_actions_mask", None)
        if mask is not None:
            mask_array = np.asarray(mask).reshape(-1)
            legal_actions = [index for index, allowed in enumerate(mask_array) if allowed]
            if legal_actions:
                return int(self.rng.choice(legal_actions))
        if hasattr(env.action_space, "n"):
            return int(self.rng.randrange(int(env.action_space.n)))
        return int(env.action_space.sample())


def mean(values):
    return statistics.fmean(values) if values else 0.0


def std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


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
        "method": f"Depth-{depth} Expectimax {chance_mode}",
        "agent": "expectimax",
        "depth": depth,
        "beam_width": beam_width,
        "chance_samples": chance_samples,
        "chance_mode": chance_mode,
        "heuristic_mode": "auto",
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
        "method": config["method"],
        "depth": config["depth"],
        "beam_width": config["beam_width"],
        "chance_samples": config["chance_samples"],
        "chance_mode": config["chance_mode"],
        "heuristic_mode": config["heuristic_mode"],
        "episodes": len(results),
        "average_reward": mean(rewards),
        "reward_std": std(rewards),
        "best_reward": max(rewards, default=0.0),
        "average_steps": mean(steps),
        "steps_std": std(steps),
        "average_decision_time": mean(decision_times),
        "total_wall_time": elapsed_time,
    }


def make_agent(config, seed):
    if config["agent"] == "random":
        return RandomAgent(random.Random(seed))
    return ExpectimaxAgent(
        depth=config["depth"],
        beam_width=config["beam_width"],
        sample_chance=config["depth"] > 1,
        chance_samples=config["chance_samples"],
        chance_mode=config["chance_mode"],
        heuristic_mode=config["heuristic_mode"],
        rng=random.Random(seed),
    )


def run_config(config, seeds, render_mode, delay_ms, max_steps):
    env = make_env(render_mode)
    start_time = time.perf_counter()
    try:
        results = []
        for episode_seed in seeds:
            agent = make_agent(config, episode_seed)
            result = run_episode(
                env,
                agent,
                seed=episode_seed,
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
        "method",
        "depth",
        "beam_width",
        "chance_samples",
        "chance_mode",
        "heuristic_mode",
        "episodes",
        "average_reward",
        "reward_std",
        "best_reward",
        "average_steps",
        "steps_std",
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
    seeds = [args.seed + offset for offset in range(args.episodes)]
    summaries = []

    for index, config in enumerate(configs, start=1):
        print(
            f"[{index}/{len(configs)}] "
            f"method={config['method']}, "
            f"depth={config['depth']}, "
            f"beam_width={config['beam_width']}, "
            f"chance_samples={config['chance_samples']}, "
            f"chance_mode={config['chance_mode']}, "
            f"heuristic_mode={config['heuristic_mode']}"
        )
        summary = run_config(
            config,
            seeds=seeds,
            render_mode=args.render_mode,
            delay_ms=args.delay_ms,
            max_steps=args.max_steps,
        )
        summaries.append(summary)
        print(
            f"  avg_reward={summary['average_reward']:.2f}, "
            f"reward_std={summary['reward_std']:.2f}, "
            f"best_reward={summary['best_reward']:.2f}, "
            f"avg_steps={summary['average_steps']:.2f}, "
            f"avg_decision={summary['average_decision_time']:.4f}s"
        )

    write_csv(args.output, summaries)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
