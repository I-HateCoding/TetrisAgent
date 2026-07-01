"""Evaluate a saved after-state DQN agent over several Tetris episodes."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np

from dqn_agent import DQNAgent
from main import make_env, run_episode


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DQN on Tetris-Gymnasium.")
    parser.add_argument("--model", type=Path, required=True, help="Saved DQN model path.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=42, help="First environment seed.")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Evaluation exploration rate.")
    parser.add_argument("--device", default="auto", help="PyTorch device: auto, cpu, or cuda.")
    parser.add_argument(
        "--render-mode",
        default="ansi",
        choices=["human", "ansi", "rgb_array"],
        help="Gymnasium render mode.",
    )
    parser.add_argument("--delay-ms", type=int, default=0, help="Delay between rendered steps.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional episode limit.")
    return parser.parse_args()


def mean(values):
    return statistics.fmean(values) if values else 0.0


def main():
    args = parse_args()
    env = make_env(args.render_mode)
    agent = DQNAgent(model_path=args.model, device=args.device, epsilon=args.epsilon)

    results = []
    try:
        for episode in range(args.episodes):
            result = run_episode(
                env,
                agent,
                seed=args.seed + episode,
                delay_ms=args.delay_ms,
                max_steps=args.max_steps,
            )
            results.append(result)
            print(
                f"Episode {episode + 1:02d}: "
                f"reward={result['total_reward']:.2f}, "
                f"steps={result['steps']}, "
                f"avg_decision={result['average_decision_time']:.4f}s"
            )
    finally:
        env.close()

    print("\nSummary")
    rewards = [r["total_reward"] for r in results]
    steps = [r["steps"] for r in results]
    decision_times = [r["average_decision_time"] for r in results]
    print(f"Episodes: {len(results)}")
    print(f"Average reward: {mean(rewards):.2f}")
    print(f"Median reward: {float(np.median(rewards)) if rewards else 0.0:.2f}")
    print(f"P25 reward: {float(np.percentile(rewards, 25)) if rewards else 0.0:.2f}")
    print(f"Worst reward: {min(rewards, default=0.0):.2f}")
    print(f"Best reward: {max(rewards, default=0.0):.2f}")
    print(f"Average steps: {mean(steps):.2f}")
    print(f"Average decision time: {mean(decision_times):.4f}s")


if __name__ == "__main__":
    main()
