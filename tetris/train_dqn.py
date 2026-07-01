"""Train an afterstate DQN for the grouped Tetris environment.

The DQNAgent in dqn_agent.py scores one candidate board at a time. This trainer
uses the same afterstate representation: observation[action] is the board after
placing the current tetromino with that action, and the network outputs one Q
value for that board.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dqn_training_presets import DEFAULT_PRESET, get_preset, preset_names
from main import make_env


class TetrisAfterstateQNet(nn.Module):
    """Lightweight MLP that maps one 24x18 candidate board to one Q value."""

    def __init__(self, input_scale: float = 7.0) -> None:
        super().__init__()
        self.input_scale = float(input_scale)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24 * 18, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.float() / self.input_scale
        return self.head(x)


@dataclass
class Transition:
    board: np.ndarray
    reward: float
    next_observation: np.ndarray
    next_mask: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, rng: random.Random) -> None:
        self.items: deque[Transition] = deque(maxlen=int(capacity))
        self.rng = rng

    def add(self, transition: Transition) -> None:
        self.items.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return self.rng.sample(list(self.items), int(batch_size))

    def __len__(self) -> int:
        return len(self.items)


@dataclass(frozen=True)
class BoardShape:
    max_height: int
    aggregate_height: int
    width: int


def parse_args() -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--preset",
        default=DEFAULT_PRESET,
        choices=preset_names(),
        help="Training parameter preset.",
    )
    base_parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available training presets and exit.",
    )
    base_args, _ = base_parser.parse_known_args()

    if base_args.list_presets:
        for name in preset_names():
            preset = get_preset(name)
            print(f"{name}: {preset.get('description', '')}")
        raise SystemExit(0)

    preset = get_preset(base_args.preset)
    preset_description = str(preset.pop("description", ""))

    parser = argparse.ArgumentParser(
        description="Train an afterstate DQN for Tetris.",
        parents=[base_parser],
    )
    parser.set_defaults(**preset)
    parser.set_defaults(preset=base_args.preset, preset_description=preset_description)
    parser.add_argument("--steps", type=int, help="Environment steps to collect.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--gamma", type=float, help="Discount factor.")
    parser.add_argument("--lr", type=float, help="Adam learning rate.")
    parser.add_argument("--batch-size", type=int, help="Replay batch size.")
    parser.add_argument("--buffer-size", type=int, help="Replay capacity.")
    parser.add_argument("--warmup-steps", type=int, help="Steps before training.")
    parser.add_argument("--train-every", type=int, help="Train every N environment steps.")
    parser.add_argument("--updates-per-step", type=int, help="Gradient updates per train step.")
    parser.add_argument(
        "--target-update-interval",
        type=int,
        help="Copy online weights to target network every N gradient updates.",
    )
    parser.add_argument("--epsilon-start", type=float, help="Initial exploration rate.")
    parser.add_argument("--epsilon-end", type=float, help="Final exploration rate.")
    parser.add_argument(
        "--epsilon-decay-steps",
        type=int,
        help="Linear epsilon decay length in environment steps.",
    )
    parser.add_argument(
        "--reward-clip",
        type=float,
        help="Clip rewards to +/- this value. Use 0 to disable clipping.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=["env", "shaped"],
        help="Use raw environment rewards or shaped rewards for DQN updates.",
    )
    parser.add_argument(
        "--line-reward",
        type=float,
        help="Reward multiplier for cleared lines in shaped mode.",
    )
    parser.add_argument(
        "--line-reward-power",
        type=float,
        help="Exponent applied to lines cleared in shaped mode.",
    )
    parser.add_argument(
        "--survival-reward",
        type=float,
        help="Small reward for a non-terminal placement in shaped mode.",
    )
    parser.add_argument(
        "--height-penalty",
        type=float,
        help="Penalty per max occupied column height in shaped mode.",
    )
    parser.add_argument(
        "--aggregate-height-penalty",
        type=float,
        help="Penalty per average occupied column height in shaped mode.",
    )
    parser.add_argument(
        "--game-over-penalty",
        type=float,
        help="Terminal penalty in shaped mode.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        help="Optional training episode step cap.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        help="Greedy evaluation episodes after training.",
    )
    parser.add_argument(
        "--eval-max-steps",
        type=int,
        help="Optional evaluation episode step cap.",
    )
    parser.add_argument("--device", help="Torch device, such as cpu or cuda.")
    parser.add_argument(
        "--torch-threads",
        type=int,
        help="PyTorch CPU thread count. Small models are usually faster with 1.",
    )
    parser.add_argument("--log-interval", type=int, help="Print every N steps.")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        help="Save a checkpoint every N environment steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Directory for periodic and best checkpoints. Defaults next to --output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="TorchScript model output path.",
    )
    args = parser.parse_args()
    if not isinstance(args.output, Path):
        args.output = Path(args.output)
    if args.checkpoint_dir is not None and not isinstance(args.checkpoint_dir, Path):
        args.checkpoint_dir = Path(args.checkpoint_dir)
    return args


def set_seeds(seed: int) -> random.Random:
    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return rng


def epsilon_at_step(args: argparse.Namespace, step: int) -> float:
    if args.epsilon_decay_steps <= 0:
        return float(args.epsilon_end)
    progress = min(1.0, step / float(args.epsilon_decay_steps))
    return float(args.epsilon_start + progress * (args.epsilon_end - args.epsilon_start))


def action_mask(info: dict[str, Any] | None, action_count: int) -> np.ndarray:
    mask = info.get("action_mask") if isinstance(info, dict) else None
    if mask is None:
        return np.ones(action_count, dtype=np.bool_)
    mask_array = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if mask_array.size < action_count:
        padded = np.zeros(action_count, dtype=np.bool_)
        padded[: mask_array.size] = mask_array
        return padded
    return mask_array[:action_count]


def clip_reward(reward: float, limit: float) -> float:
    if limit <= 0:
        return float(reward)
    return float(np.clip(reward, -limit, limit))


def playable_occupancy(board: np.ndarray) -> np.ndarray:
    matrix = np.asarray(board)
    if matrix.ndim > 2:
        matrix = np.squeeze(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D board, got shape {matrix.shape!r}")

    cropped = np.asarray(matrix)
    while cropped.shape[0] > 1 and np.all(cropped[-1, :] == 1):
        cropped = cropped[:-1, :]
    while cropped.shape[1] > 1 and np.all(cropped[:, 0] == 1):
        cropped = cropped[:, 1:]
    while cropped.shape[1] > 1 and np.all(cropped[:, -1] == 1):
        cropped = cropped[:, :-1]
    return cropped > 1


def board_shape(board: np.ndarray) -> BoardShape:
    occupied = playable_occupancy(board)
    height, width = occupied.shape
    column_heights = []
    for col in range(width):
        filled = np.flatnonzero(occupied[:, col])
        column_heights.append(0 if filled.size == 0 else height - int(filled[0]))
    return BoardShape(
        max_height=max(column_heights, default=0),
        aggregate_height=sum(column_heights),
        width=width,
    )


def lines_cleared(info: dict[str, Any] | None) -> int:
    if not isinstance(info, dict):
        return 0
    return int(info.get("lines_cleared", 0))


def training_reward(
    env_reward: float,
    board: np.ndarray,
    next_info: dict[str, Any] | None,
    done: bool,
    args: argparse.Namespace,
) -> float:
    if args.reward_mode == "env":
        return clip_reward(float(env_reward), args.reward_clip)

    shape = board_shape(board)
    cleared = lines_cleared(next_info)
    average_height = shape.aggregate_height / max(1, shape.width)
    reward = (
        args.line_reward * (cleared**args.line_reward_power)
        + (0.0 if done else args.survival_reward)
        - args.height_penalty * shape.max_height
        - args.aggregate_height_penalty * average_height
    )
    if done:
        reward += args.game_over_penalty
    return clip_reward(float(reward), args.reward_clip)


@torch.no_grad()
def select_action(
    model: nn.Module,
    observation: np.ndarray,
    mask: np.ndarray,
    epsilon: float,
    rng: random.Random,
    device: torch.device,
) -> int:
    legal_actions = np.flatnonzero(mask)
    if legal_actions.size == 0:
        return 0
    if epsilon > 0.0 and rng.random() < epsilon:
        return int(rng.choice(legal_actions.tolist()))

    boards = torch.as_tensor(observation[legal_actions], dtype=torch.float32, device=device)
    boards = boards.unsqueeze(1)
    q_values = model(boards).reshape(-1)
    return int(legal_actions[int(torch.argmax(q_values).item())])


def estimate_next_values(
    target_net: nn.Module,
    transitions: list[Transition],
    device: torch.device,
) -> torch.Tensor:
    values = np.zeros(len(transitions), dtype=np.float32)
    candidate_boards: list[np.ndarray] = []
    owner_indices: list[int] = []

    for index, transition in enumerate(transitions):
        if transition.done:
            continue
        legal_actions = np.flatnonzero(transition.next_mask)
        if legal_actions.size == 0:
            continue
        candidate_boards.append(transition.next_observation[legal_actions])
        owner_indices.extend([index] * int(legal_actions.size))

    if not candidate_boards:
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    boards = np.concatenate(candidate_boards, axis=0)
    boards_tensor = torch.as_tensor(boards, dtype=torch.float32, device=device).unsqueeze(1)
    q_values = target_net(boards_tensor).reshape(-1).detach().cpu().numpy()

    values.fill(-np.inf)
    for owner_index, q_value in zip(owner_indices, q_values):
        if q_value > values[owner_index]:
            values[owner_index] = float(q_value)
    values[~np.isfinite(values)] = 0.0
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def train_batch(
    online_net: nn.Module,
    target_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    transitions: list[Transition],
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    boards = np.stack([transition.board for transition in transitions], axis=0)
    rewards = torch.tensor(
        [transition.reward for transition in transitions],
        dtype=torch.float32,
        device=device,
    )
    dones = torch.tensor(
        [transition.done for transition in transitions],
        dtype=torch.float32,
        device=device,
    )

    boards_tensor = torch.as_tensor(boards, dtype=torch.float32, device=device).unsqueeze(1)
    q_values = online_net(boards_tensor).reshape(-1)

    with torch.no_grad():
        next_values = estimate_next_values(target_net, transitions, device)
        targets = rewards + args.gamma * next_values * (1.0 - dones)

    loss = F.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def evaluate_policy(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    seed_offset: int,
) -> dict[str, float]:
    env = make_env("ansi")
    rewards: list[float] = []
    steps: list[int] = []
    try:
        for episode in range(args.eval_episodes):
            observation, info = env.reset(seed=args.seed + seed_offset + episode)
            total_reward = 0.0
            episode_steps = 0
            done = False
            while not done:
                mask = action_mask(info, int(env.action_space.n))
                action = select_action(model, observation, mask, 0.0, random.Random(0), device)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                episode_steps += 1
                done = bool(terminated or truncated)
                if args.eval_max_steps and episode_steps >= args.eval_max_steps:
                    break
            rewards.append(total_reward)
            steps.append(episode_steps)
    finally:
        env.close()

    return {
        "average_reward": float(np.mean(rewards)) if rewards else 0.0,
        "best_reward": float(np.max(rewards)) if rewards else 0.0,
        "average_steps": float(np.mean(steps)) if steps else 0.0,
    }


def save_model(model: nn.Module, output: Path, device: torch.device) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.to("cpu").eval()
    scripted = torch.jit.script(model_cpu)
    torch.jit.save(scripted, str(output))
    model.to(device)


def write_metrics(output: Path, metrics: dict[str, Any]) -> None:
    metrics_path = output.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)
        file.write("\n")


def checkpoint_directory(args: argparse.Namespace) -> Path:
    if args.checkpoint_dir is not None:
        return args.checkpoint_dir
    return args.output.parent / f"{args.output.stem}_checkpoints"


def checkpoint_path(args: argparse.Namespace, name: str) -> Path:
    return checkpoint_directory(args) / f"{args.output.stem}_{name}.pt"


def main() -> None:
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    rng = set_seeds(args.seed)
    device = torch.device(args.device)

    env = make_env("ansi")
    action_count = int(env.action_space.n)
    online_net = TetrisAfterstateQNet().to(device)
    target_net = TetrisAfterstateQNet().to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()
    optimizer = torch.optim.Adam(online_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.buffer_size, rng)

    losses: list[float] = []
    log_history: list[dict[str, float]] = []
    checkpoint_history: list[dict[str, Any]] = []
    episode_rewards: list[float] = []
    training_rewards: list[float] = []
    transition_lines: list[int] = []
    transition_max_heights: list[int] = []
    gradient_updates = 0
    episode = 0
    episode_reward = 0.0
    episode_steps = 0
    best_reward_100 = -float("inf")
    best_checkpoint: str | None = None
    start_time = time.perf_counter()

    observation, info = env.reset(seed=args.seed)
    try:
        for env_step in range(1, args.steps + 1):
            epsilon = epsilon_at_step(args, env_step)
            mask = action_mask(info, action_count)
            action = select_action(online_net, observation, mask, epsilon, rng, device)
            action_board = np.asarray(observation[action], dtype=np.uint8).copy()

            next_observation, reward, terminated, truncated, next_info = env.step(action)
            episode_steps += 1
            episode_reward += float(reward)
            terminal_done = bool(terminated or truncated)
            done = terminal_done
            if args.max_episode_steps and episode_steps >= args.max_episode_steps:
                done = True
            replay_reward = training_reward(
                float(reward),
                action_board,
                next_info,
                terminal_done,
                args,
            )
            shape = board_shape(action_board)
            training_rewards.append(replay_reward)
            transition_lines.append(lines_cleared(next_info))
            transition_max_heights.append(shape.max_height)

            replay.add(
                Transition(
                    board=action_board,
                    reward=replay_reward,
                    next_observation=np.asarray(next_observation, dtype=np.uint8).copy(),
                    next_mask=action_mask(next_info, action_count),
                    done=done,
                )
            )

            observation = next_observation
            info = next_info

            if done:
                episode_rewards.append(episode_reward)
                episode += 1
                observation, info = env.reset(seed=args.seed + episode)
                episode_reward = 0.0
                episode_steps = 0

            if (
                len(replay) >= max(args.warmup_steps, args.batch_size)
                and env_step % args.train_every == 0
            ):
                online_net.train()
                for _ in range(args.updates_per_step):
                    batch = replay.sample(args.batch_size)
                    loss = train_batch(online_net, target_net, optimizer, batch, args, device)
                    losses.append(loss)
                    gradient_updates += 1
                    if gradient_updates % args.target_update_interval == 0:
                        target_net.load_state_dict(online_net.state_dict())
                        target_net.eval()

            if (
                args.checkpoint_interval > 0
                and env_step % args.checkpoint_interval == 0
            ):
                path = checkpoint_path(args, f"step_{env_step:07d}")
                save_model(online_net, path, device)
                checkpoint_history.append(
                    {
                        "env_step": env_step,
                        "path": str(path),
                        "kind": "periodic",
                    }
                )
                print(f"checkpoint={path}", flush=True)

            if args.log_interval > 0 and env_step % args.log_interval == 0:
                recent_losses = losses[-100:]
                recent_rewards_10 = episode_rewards[-10:]
                recent_rewards_100 = episode_rewards[-100:]
                recent_training_rewards = training_rewards[-100:]
                recent_lines = transition_lines[-100:]
                recent_max_heights = transition_max_heights[-100:]
                mean_loss = float(np.mean(recent_losses)) if recent_losses else 0.0
                mean_reward_10 = (
                    float(np.mean(recent_rewards_10)) if recent_rewards_10 else 0.0
                )
                mean_reward_100 = (
                    float(np.mean(recent_rewards_100)) if recent_rewards_100 else 0.0
                )
                mean_training_reward = (
                    float(np.mean(recent_training_rewards))
                    if recent_training_rewards
                    else 0.0
                )
                mean_lines = float(np.mean(recent_lines)) if recent_lines else 0.0
                mean_max_height = (
                    float(np.mean(recent_max_heights)) if recent_max_heights else 0.0
                )

                if recent_rewards_100 and mean_reward_100 > best_reward_100:
                    best_reward_100 = mean_reward_100
                    best_path = checkpoint_path(args, "best_reward100")
                    save_model(online_net, best_path, device)
                    best_checkpoint = str(best_path)
                    checkpoint_history.append(
                        {
                            "env_step": env_step,
                            "path": str(best_path),
                            "kind": "best_reward100",
                            "mean_reward_100": mean_reward_100,
                        }
                    )
                    print(
                        f"best_checkpoint={best_path} "
                        f"mean_reward_100={mean_reward_100:.2f}",
                        flush=True,
                    )

                log_history.append(
                    {
                        "env_step": float(env_step),
                        "episodes": float(episode),
                        "epsilon": float(epsilon),
                        "gradient_updates": float(gradient_updates),
                        "mean_loss_100": mean_loss,
                        "mean_reward_10": mean_reward_10,
                        "mean_reward_100": mean_reward_100,
                        "mean_training_reward_100": mean_training_reward,
                        "mean_lines_cleared_100": mean_lines,
                        "mean_max_height_100": mean_max_height,
                    }
                )
                print(
                    f"step={env_step} episodes={episode} epsilon={epsilon:.3f} "
                    f"buffer={len(replay)} updates={gradient_updates} "
                    f"mean_loss_100={mean_loss:.4f} "
                    f"mean_reward_10={mean_reward_10:.2f} "
                    f"mean_reward_100={mean_reward_100:.2f} "
                    f"mean_training_reward_100={mean_training_reward:.2f} "
                    f"mean_lines_cleared_100={mean_lines:.3f} "
                    f"mean_max_height_100={mean_max_height:.2f}",
                    flush=True,
                )
    finally:
        env.close()

    online_net.eval()
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()
    eval_result = evaluate_policy(online_net, args, device, seed_offset=100000)
    save_model(online_net, args.output, device)

    first_losses = losses[: min(100, len(losses))]
    last_losses = losses[-min(100, len(losses)) :]
    metrics = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "preset": {
            "name": args.preset,
            "description": args.preset_description,
        },
        "reward_config": {
            "mode": args.reward_mode,
            "line_reward": args.line_reward,
            "line_reward_power": args.line_reward_power,
            "survival_reward": args.survival_reward,
            "height_penalty": args.height_penalty,
            "aggregate_height_penalty": args.aggregate_height_penalty,
            "game_over_penalty": args.game_over_penalty,
            "reward_clip": args.reward_clip,
        },
        "training": {
            "env_steps": args.steps,
            "episodes": episode,
            "gradient_updates": gradient_updates,
            "elapsed_seconds": time.perf_counter() - start_time,
            "first_loss_mean": float(np.mean(first_losses)) if first_losses else None,
            "last_loss_mean": float(np.mean(last_losses)) if last_losses else None,
            "first_logged_loss_mean": log_history[0]["mean_loss_100"] if log_history else None,
            "last_logged_loss_mean": log_history[-1]["mean_loss_100"] if log_history else None,
            "last_10_episode_reward_mean": float(np.mean(episode_rewards[-10:]))
            if episode_rewards
            else None,
            "last_100_episode_reward_mean": float(np.mean(episode_rewards[-100:]))
            if episode_rewards
            else None,
            "best_logged_reward_100": best_reward_100
            if np.isfinite(best_reward_100)
            else None,
            "best_checkpoint": best_checkpoint,
            "last_100_training_reward_mean": float(np.mean(training_rewards[-100:]))
            if training_rewards
            else None,
            "last_100_lines_cleared_mean": float(np.mean(transition_lines[-100:]))
            if transition_lines
            else None,
            "last_100_max_height_mean": float(np.mean(transition_max_heights[-100:]))
            if transition_max_heights
            else None,
        },
        "log_history": log_history,
        "checkpoint_history": checkpoint_history,
        "evaluation": eval_result,
    }
    write_metrics(args.output, metrics)

    print(f"saved_model={args.output}")
    print(f"saved_metrics={args.output.with_suffix('.metrics.json')}")
    print(json.dumps(metrics["training"], indent=2, sort_keys=True))
    print(json.dumps(metrics["evaluation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
