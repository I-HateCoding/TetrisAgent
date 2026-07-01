"""Sweep depth-2 Expectimax heuristic weights and save results as CSV."""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

import gymnasium as gym
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations

from expectimax_agent import Depth2HeuristicWeights, ExpectimaxAgent


BASE_CURRENT = Depth2HeuristicWeights()

WEIGHT_PRESETS = [
    ("base_current", BASE_CURRENT),
    (
        "line_plus",
        replace(
            BASE_CURRENT,
            lines_cleared=9.0,
            complete_lines=2.0,
        ),
    ),
    (
        "line_aggressive",
        replace(
            BASE_CURRENT,
            lines_cleared=11.0,
            complete_lines=2.5,
            holes=-1.90,
        ),
    ),
    (
        "less_smooth_penalty",
        replace(
            BASE_CURRENT,
            bumpiness=-0.40,
            height_std=-0.20,
        ),
    ),
]


def make_env(render_mode):
    env = gym.make("tetris_gymnasium/Tetris", render_mode=render_mode)
    return GroupedActionsObservations(env)


def run_episode(env, agent, seed=42, delay_ms=0, max_steps=None):
    observation, info = env.reset(seed=seed)
    terminated = False
    truncated = False
    total_reward = 0.0
    steps = 0
    decision_times = []

    while not (terminated or truncated):
        env.render()

        start_time = time.perf_counter()
        action = agent.select_action(env, observation, info)
        decision_times.append(time.perf_counter() - start_time)

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

        if delay_ms > 0:
            import cv2

            cv2.waitKey(delay_ms)
        if max_steps is not None and steps >= max_steps:
            break

    return {
        "total_reward": total_reward,
        "steps": steps,
        "average_decision_time": sum(decision_times) / len(decision_times)
        if decision_times
        else 0.0,
    }


def mean(values):
    return statistics.fmean(values) if values else 0.0


def std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Tune depth-2 Expectimax weights.")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per preset.")
    parser.add_argument("--seed", type=int, default=42, help="First environment seed.")
    parser.add_argument("--max-steps", type=int, default=500, help="Optional episode limit.")
    parser.add_argument("--beam-width", type=int, default=3, help="Top actions kept for search.")
    parser.add_argument(
        "--chance-samples",
        type=int,
        default=2,
        help="Number of next tetrominoes sampled at chance nodes.",
    )
    parser.add_argument(
        "--chance-mode",
        default="queue",
        choices=["expected", "queue"],
        help="Use expected random pieces or the visible queue for lookahead.",
    )
    parser.add_argument(
        "--penalty-mode",
        default="none",
        choices=["strong", "weak", "none"],
        help="Depth-2 nonlinear penalty strength.",
    )
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
        default=Path("depth2_weight_results.csv"),
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
        "reward_std": std(rewards),
        "best_reward": max(rewards, default=0.0),
        "average_steps": mean(steps),
        "steps_std": std(steps),
        "average_decision_time": mean(decision_times),
        "total_wall_time": elapsed_time,
    }
    row.update(asdict(weights))
    return row


def run_preset(name, weights, args, seeds):
    env = make_env(args.render_mode)
    start_time = time.perf_counter()
    try:
        results = []
        for episode_seed in seeds:
            agent = ExpectimaxAgent(
                depth=2,
                weights=weights,
                beam_width=args.beam_width,
                sample_chance=True,
                chance_samples=args.chance_samples,
                chance_mode=args.chance_mode,
                heuristic_mode="depth2",
                penalty_mode=args.penalty_mode,
                rng=random.Random(episode_seed),
            )
            results.append(
                run_episode(
                    env,
                    agent,
                    seed=episode_seed,
                    delay_ms=args.delay_ms,
                    max_steps=args.max_steps,
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
        "reward_std",
        "best_reward",
        "average_steps",
        "steps_std",
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
    seeds = [args.seed + offset for offset in range(args.episodes)]
    summaries = []

    for index, (name, weights) in enumerate(WEIGHT_PRESETS, start=1):
        print(f"[{index}/{len(WEIGHT_PRESETS)}] preset={name}", flush=True)
        summary = run_preset(name, weights, args, seeds)
        summaries.append(summary)
        print(
            f"  preset={name}, "
            f"avg_reward={summary['average_reward']:.2f}, "
            f"reward_std={summary['reward_std']:.2f}, "
            f"best_reward={summary['best_reward']:.2f}, "
            f"avg_steps={summary['average_steps']:.2f}, "
            f"steps_std={summary['steps_std']:.2f}, "
            f"avg_decision={summary['average_decision_time']:.4f}s",
            flush=True,
        )

    summaries.sort(key=lambda row: row["average_reward"], reverse=True)
    write_csv(args.output, summaries)
    print(f"\nSaved sorted results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
