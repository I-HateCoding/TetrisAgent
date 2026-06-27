import argparse
import time

import cv2
import gymnasium as gym
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations

from expectimax_agent import ExpectimaxAgent


def make_env(render_mode):
    env = gym.make("tetris_gymnasium/Tetris", render_mode=render_mode)
    return GroupedActionsObservations(env)


def run_episode(env, agent, seed=42, delay_ms=100, max_steps=None):
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Tetris Expectimax agent.")
    parser.add_argument("--depth", type=int, default=1, help="Expectimax search depth.")
    parser.add_argument(
        "--beam-width",
        type=int,
        default=8,
        help="Top heuristic actions kept for depth > 1 search.",
    )
    parser.add_argument(
        "--chance-samples",
        type=int,
        default=4,
        help="Number of next tetrominoes sampled at chance nodes.",
    )
    parser.add_argument(
        "--chance-mode",
        default="expected",
        choices=["expected", "queue"],
        help="Use expected random pieces or the visible queue for lookahead.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Environment random seed.")
    parser.add_argument(
        "--render-mode",
        default="human",
        choices=["human", "ansi", "rgb_array"],
        help="Gymnasium render mode.",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=100,
        help="Delay between rendered steps. Use 0 for no delay.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional safety limit for one episode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env = make_env(args.render_mode)
    agent = ExpectimaxAgent(
        depth=args.depth,
        beam_width=args.beam_width,
        sample_chance=args.depth > 1,
        chance_samples=args.chance_samples,
        chance_mode=args.chance_mode,
    )

    try:
        result = run_episode(
            env,
            agent,
            seed=args.seed,
            delay_ms=args.delay_ms,
            max_steps=args.max_steps,
        )
    finally:
        env.close()

    print("Game Over!")
    print(f"Total reward: {result['total_reward']:.2f}")
    print(f"Steps: {result['steps']}")
    print(f"Average decision time: {result['average_decision_time']:.4f}s")


if __name__ == "__main__":
    main()
