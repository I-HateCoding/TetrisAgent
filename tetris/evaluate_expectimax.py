"""Evaluate the Expectimax agent over several Tetris episodes."""

from __future__ import annotations

import argparse
import statistics

from expectimax_agent import ExpectimaxAgent
from main import make_env, run_episode


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Expectimax on Tetris-Gymnasium.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--depth", type=int, default=1, help="Expectimax search depth.")
    parser.add_argument("--beam-width", type=int, default=8, help="Top actions kept for search.")
    parser.add_argument(
        "--chance-samples",
        type=int,
        default=4,
        help="Number of next tetrominoes sampled at chance nodes.",
    )
    parser.add_argument("--seed", type=int, default=42, help="First environment seed.")
    parser.add_argument(
        "--render-mode",
        default="ansi",
        choices=["human", "ansi", "rgb_array"],
        help="Gymnasium render mode.",
    )
    parser.add_argument("--delay-ms", type=int, default=0, help="Delay between steps.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional episode limit.")
    return parser.parse_args()


def mean(values):
    return statistics.fmean(values) if values else 0.0


def main():
    args = parse_args()
    env = make_env(args.render_mode)
    agent = ExpectimaxAgent(
        depth=args.depth,
        beam_width=args.beam_width,
        sample_chance=args.depth > 1,
        chance_samples=args.chance_samples,
    )

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
    print(f"Episodes: {len(results)}")
    print(f"Average reward: {mean([r['total_reward'] for r in results]):.2f}")
    print(f"Best reward: {max((r['total_reward'] for r in results), default=0.0):.2f}")
    print(f"Average steps: {mean([r['steps'] for r in results]):.2f}")
    print(
        "Average decision time: "
        f"{mean([r['average_decision_time'] for r in results]):.4f}s"
    )


if __name__ == "__main__":
    main()
