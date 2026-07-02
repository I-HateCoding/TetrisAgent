"""Train an after-state DQN agent for Tetris-Gymnasium."""

from __future__ import annotations

import argparse
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

from dqn_model import AfterstateQNetwork
from main import make_env, wait_for_frame


@dataclass(frozen=True)
class Transition:
    board: np.ndarray
    reward: float
    next_boards: np.ndarray
    next_mask: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, rng: random.Random) -> None:
        self.data: deque[Transition] = deque(maxlen=int(capacity))
        self.rng = rng

    def push(self, transition: Transition) -> None:
        self.data.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return self.rng.sample(list(self.data), int(batch_size))

    def __len__(self) -> int:
        return len(self.data)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an after-state DQN Tetris agent.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Initial random seed.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional steps per episode.")
    parser.add_argument("--render", action="store_true", help="Render training episodes.")
    parser.add_argument(
        "--render-mode",
        default="ansi",
        choices=["human", "ansi", "rgb_array"],
        help="Gymnasium render mode used when --render is set.",
    )
    parser.add_argument("--delay-ms", type=int, default=0, help="Render delay when --render is set.")
    parser.add_argument("--device", default="auto", help="PyTorch device: auto, cpu, or cuda.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument("--batch-size", type=int, default=64, help="Replay batch size.")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer capacity.")
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=1000,
        help="Environment steps before gradient updates begin.",
    )
    parser.add_argument("--train-every", type=int, default=1, help="Gradient update period.")
    parser.add_argument(
        "--target-update",
        type=int,
        default=1000,
        help="Environment steps between target-network syncs.",
    )
    parser.add_argument("--hidden-size", type=int, default=256, help="Model hidden layer size.")
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="Initial epsilon-greedy exploration rate.",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
        help="Final epsilon-greedy exploration rate.",
    )
    parser.add_argument(
        "--epsilon-decay-steps",
        type=int,
        default=50000,
        help="Linear epsilon decay length in environment steps.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/dqn_afterstate.pt"),
        help="Checkpoint output path.",
    )
    parser.add_argument(
        "--export-torchscript",
        type=Path,
        default=None,
        help="Optional TorchScript model export path.",
    )
    parser.add_argument("--save-every", type=int, default=50, help="Checkpoint period in episodes.")
    parser.add_argument("--log-every", type=int, default=10, help="Console log period in episodes.")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    requested = str(name).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def epsilon_by_step(step: int, start: float, end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return float(end)
    fraction = min(1.0, max(0.0, step / decay_steps))
    return float(start + fraction * (end - start))


def legal_mask(env: Any, info: Optional[dict], action_count: int) -> np.ndarray:
    mask = info.get("action_mask") if isinstance(info, dict) else None
    if mask is None:
        mask = getattr(env, "legal_actions_mask", None)
    if mask is None:
        return np.ones(action_count, dtype=bool)

    mask_array = np.asarray(mask).reshape(-1).astype(bool)
    if mask_array.size < action_count:
        padded = np.zeros(action_count, dtype=bool)
        padded[: mask_array.size] = mask_array
        return padded
    return mask_array[:action_count]


def action_boards(observation: Any, action_count: int) -> np.ndarray:
    if isinstance(observation, dict):
        observation = observation.get("board")
    boards = np.asarray(observation, dtype=np.float32)
    if boards.ndim < 3:
        raise ValueError(
            "DQN training requires grouped observations shaped like "
            "(num_actions, height, width)."
        )
    if boards.shape[0] < action_count:
        raise ValueError(
            f"Grouped observation has {boards.shape[0]} boards, "
            f"but env.action_space.n is {action_count}."
        )
    return boards[:action_count]


def select_action(
    model: nn.Module,
    boards: np.ndarray,
    mask: np.ndarray,
    epsilon: float,
    rng: random.Random,
    device: torch.device,
) -> int:
    legal_actions = np.flatnonzero(mask)
    if legal_actions.size == 0:
        return int(rng.randrange(boards.shape[0]))
    if rng.random() < epsilon:
        return int(rng.choice(legal_actions.tolist()))

    tensor = torch.as_tensor(boards, dtype=torch.float32, device=device)
    with torch.no_grad():
        values = model(tensor).detach().reshape(-1).cpu().numpy()
    values = values[: boards.shape[0]].copy()
    values[~mask] = -np.inf
    return int(np.argmax(values))


def optimize_model(
    policy_net: nn.Module,
    target_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> float:
    batch = replay.sample(batch_size)
    boards = torch.as_tensor(
        np.stack([transition.board for transition in batch]),
        dtype=torch.float32,
        device=device,
    )
    rewards = torch.as_tensor(
        [transition.reward for transition in batch], dtype=torch.float32, device=device
    )
    next_boards = torch.as_tensor(
        np.stack([transition.next_boards for transition in batch]),
        dtype=torch.float32,
        device=device,
    )
    next_masks = torch.as_tensor(
        np.stack([transition.next_mask for transition in batch]), dtype=torch.bool, device=device
    )
    dones = torch.as_tensor(
        [transition.done for transition in batch], dtype=torch.bool, device=device
    )

    current_values = policy_net(boards).reshape(-1)

    with torch.no_grad():
        batch_count, action_count = next_boards.shape[:2]
        flat_next_boards = next_boards.reshape(batch_count * action_count, *next_boards.shape[2:])
        next_values = target_net(flat_next_boards).reshape(batch_count, action_count)
        next_values = next_values.masked_fill(~next_masks, -1.0e9)
        has_next_action = next_masks.any(dim=1)
        max_next_values = next_values.max(dim=1).values
        max_next_values = torch.where(
            has_next_action & ~dones,
            max_next_values,
            torch.zeros_like(max_next_values),
        )
        targets = rewards + float(gamma) * max_next_values

    loss = nn.functional.smooth_l1_loss(current_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()
    return float(loss.detach().cpu().item())


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    episode: int,
    total_steps: int,
    observation_shape: tuple[int, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_class": "AfterstateQNetwork",
            "model_kwargs": {"hidden_size": args.hidden_size},
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode": episode,
            "total_steps": total_steps,
            "observation_shape": observation_shape,
            "gamma": args.gamma,
        },
        path,
    )


def export_torchscript(path: Path, model: nn.Module, observation_shape: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = observation_shape[-2:]
    device = next(model.parameters()).device
    example = torch.zeros((1, 1, height, width), dtype=torch.float32, device=device)
    traced = torch.jit.trace(model, example)
    traced.save(str(path))


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    env = make_env(args.render_mode if args.render else "ansi")
    action_count = int(env.action_space.n)

    policy_net = AfterstateQNetwork(hidden_size=args.hidden_size).to(device)
    target_net = AfterstateQNetwork(hidden_size=args.hidden_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.buffer_size, rng)
    total_steps = 0
    recent_rewards: deque[float] = deque(maxlen=100)
    recent_losses: deque[float] = deque(maxlen=100)
    observation_shape: tuple[int, ...] | None = None
    start_time = time.perf_counter()

    try:
        for episode in range(1, args.episodes + 1):
            observation, info = env.reset(seed=args.seed + episode - 1)
            terminated = False
            truncated = False
            episode_reward = 0.0
            episode_steps = 0

            while not (terminated or truncated):
                if args.render:
                    env.render()

                boards = action_boards(observation, action_count)
                if observation_shape is None:
                    observation_shape = tuple(boards.shape[1:])
                mask = legal_mask(env, info, action_count)
                epsilon = epsilon_by_step(
                    total_steps,
                    args.epsilon_start,
                    args.epsilon_end,
                    args.epsilon_decay_steps,
                )
                action = select_action(policy_net, boards, mask, epsilon, rng, device)

                next_observation, reward, terminated, truncated, next_info = env.step(action)
                done = bool(terminated or truncated)
                if done:
                    next_candidate_boards = np.zeros_like(boards)
                    next_mask = np.zeros(action_count, dtype=bool)
                else:
                    next_candidate_boards = action_boards(next_observation, action_count)
                    next_mask = legal_mask(env, next_info, action_count)

                replay.push(
                    Transition(
                        board=np.asarray(boards[action], dtype=np.float32),
                        reward=float(reward),
                        next_boards=np.asarray(next_candidate_boards, dtype=np.float32),
                        next_mask=np.asarray(next_mask, dtype=bool),
                        done=done,
                    )
                )

                total_steps += 1
                episode_steps += 1
                episode_reward += float(reward)
                observation, info = next_observation, next_info

                if (
                    len(replay) >= args.learning_starts
                    and total_steps % args.train_every == 0
                    and len(replay) >= args.batch_size
                ):
                    loss = optimize_model(
                        policy_net,
                        target_net,
                        optimizer,
                        replay,
                        args.batch_size,
                        args.gamma,
                        device,
                    )
                    recent_losses.append(loss)

                if total_steps % args.target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                if args.render:
                    wait_for_frame(args.delay_ms)
                if args.max_steps is not None and episode_steps >= args.max_steps:
                    break

            recent_rewards.append(episode_reward)
            if episode % args.log_every == 0 or episode == 1:
                elapsed = time.perf_counter() - start_time
                mean_reward = sum(recent_rewards) / len(recent_rewards)
                mean_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
                epsilon = epsilon_by_step(
                    total_steps,
                    args.epsilon_start,
                    args.epsilon_end,
                    args.epsilon_decay_steps,
                )
                print(
                    f"Episode {episode:04d} | "
                    f"reward={episode_reward:.2f} | "
                    f"avg100={mean_reward:.2f} | "
                    f"steps={episode_steps} | "
                    f"total_steps={total_steps} | "
                    f"epsilon={epsilon:.3f} | "
                    f"loss={mean_loss:.5f} | "
                    f"elapsed={elapsed:.1f}s"
                )

            if episode % args.save_every == 0:
                if observation_shape is None:
                    raise RuntimeError("Cannot save before seeing an observation.")
                save_checkpoint(
                    args.output,
                    policy_net,
                    optimizer,
                    args,
                    episode,
                    total_steps,
                    
                    observation_shape,
                )

        if observation_shape is None:
            raise RuntimeError("Training finished without seeing any observation.")
        save_checkpoint(args.output, policy_net, optimizer, args, args.episodes, total_steps, observation_shape)
        if args.export_torchscript is not None:
            export_torchscript(args.export_torchscript, policy_net, observation_shape)
    finally:
        env.close()

    print(f"Saved checkpoint to {args.output}")
    if args.export_torchscript is not None:
        print(f"Saved TorchScript model to {args.export_torchscript}")


if __name__ == "__main__":
    main()
