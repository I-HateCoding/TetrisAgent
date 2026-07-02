import argparse
import random
import time
from pathlib import Path

from expectimax_agent import ExpectimaxAgent


def make_env(render_mode):
    import gymnasium as gym
    from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations

    env = gym.make("tetris_gymnasium/Tetris", render_mode=render_mode)
    return GroupedActionsObservations(env)


def wait_for_frame(delay_ms):
    if delay_ms <= 0:
        return

    try:
        import cv2

        cv2.waitKey(delay_ms)
    except ImportError:
        time.sleep(delay_ms / 1000.0)


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

        wait_for_frame(delay_ms)
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
    parser = argparse.ArgumentParser(description="Run a Tetris placement agent.")
    parser.add_argument(
        "--agent",
        default="expectimax",
        choices=["expectimax", "dqn"],
        help="Agent model to use for action selection.",
    )
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
    parser.add_argument(
        "--heuristic-mode",
        default="auto",
        choices=["auto", "base", "depth2"],
        help="Choose base, depth-2, or automatic heuristic weights and penalties.",
    )

    parser.add_argument(
        "--penalty-mode",
        default="strong",
        choices=["strong", "weak", "none"],
        help="Depth-2 nonlinear penalty strength.",
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
    parser.add_argument(
        "--dqn-model",
        type=Path,
        default=None,
        help="Path to a saved PyTorch DQN model or checkpoint.",
    )
    parser.add_argument(
        "--dqn-device",
        default="auto",
        help="PyTorch device for DQN inference: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--dqn-epsilon",
        type=float,
        default=0.0,
        help="Exploration probability for DQN action selection.",
    )
    return parser.parse_args()


def build_agent(args):
    if args.agent == "expectimax":
        return ExpectimaxAgent(
            depth=args.depth,
            beam_width=args.beam_width,
            sample_chance=args.depth > 1,
            chance_samples=args.chance_samples,
            chance_mode=args.chance_mode,
            heuristic_mode=args.heuristic_mode,
            penalty_mode=args.penalty_mode,
            rng=random.Random(args.seed),
        )

    if args.dqn_model is None:
        raise ValueError("--dqn-model is required when --agent dqn is selected.")

    from dqn_agent import DQNAgent

    return DQNAgent(
        model_path=args.dqn_model,
        device=args.dqn_device,
        epsilon=args.dqn_epsilon,
        rng=random.Random(args.seed),
    )


def main():
    args = parse_args()
    agent = build_agent(args)
    env = make_env(args.render_mode)

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
