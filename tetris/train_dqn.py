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
    features: np.ndarray
    reward: float
    next_boards: np.ndarray
    next_features: np.ndarray
    next_mask: np.ndarray
    done: bool


@dataclass(frozen=True)
class BoardFeatures:
    aggregate_height: int
    max_height: int
    holes: int
    bumpiness: int


FEATURE_SIZE = 8


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
        "--reward-mode",
        default="shaped",
        choices=["raw", "shaped", "absolute"],
        help=(
            "Use raw rewards, delta-shaped Tetris rewards, or older absolute "
            "feature penalties."
        ),
    )
    parser.add_argument("--reward-scale", type=float, default=1.0, help="Raw reward multiplier.")
    parser.add_argument(
        "--line-clear-bonus",
        type=float,
        default=5.0,
        help="Extra shaped reward for cleared lines squared.",
    )
    parser.add_argument(
        "--hole-penalty",
        type=float,
        default=0.8,
        help="Penalty per hole in the placed-piece board.",
    )
    parser.add_argument(
        "--height-penalty",
        type=float,
        default=0.03,
        help="Penalty per aggregate column height.",
    )
    parser.add_argument(
        "--max-height-penalty",
        type=float,
        default=0.10,
        help="Penalty per maximum column height.",
    )
    parser.add_argument(
        "--bumpiness-penalty",
        type=float,
        default=0.05,
        help="Penalty per adjacent column height difference.",
    )
    parser.add_argument(
        "--terminal-penalty",
        type=float,
        default=50.0,
        help="Extra shaped penalty when the episode ends.",
    )
    parser.add_argument(
        "--survival-bonus",
        type=float,
        default=0.05,
        help="Small shaped reward for each non-terminal placement.",
    )
    parser.add_argument(
        "--hole-reduction-bonus",
        type=float,
        default=0.4,
        help="Shaped reward per removed hole.",
    )
    parser.add_argument(
        "--height-reduction-bonus",
        type=float,
        default=0.02,
        help="Shaped reward per reduced aggregate column height.",
    )
    parser.add_argument(
        "--max-height-reduction-bonus",
        type=float,
        default=0.05,
        help="Shaped reward per reduced maximum column height.",
    )
    parser.add_argument(
        "--bumpiness-reduction-bonus",
        type=float,
        default=0.02,
        help="Shaped reward per reduced adjacent-column bumpiness.",
    )
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
        "--use-board-features",
        action="store_true",
        help=(
            "Feed normalized Tetris board features into the DQN in addition to "
            "the board image."
        ),
    )
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
        "--exploration-policy",
        default="mixed",
        choices=["random", "heuristic", "mixed"],
        help="Action source used during epsilon exploration.",
    )
    parser.add_argument(
        "--heuristic-exploration-prob",
        type=float,
        default=0.75,
        help="For mixed exploration, probability of using the heuristic action.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/dqn_afterstate.pt"),
        help="Checkpoint output path.",
    )
    parser.add_argument(
        "--best-output",
        type=Path,
        default=None,
        help=(
            "Checkpoint path for the best training metric. Defaults to "
            "<output-stem>_best<suffix>."
        ),
    )
    parser.add_argument(
        "--best-metric",
        default="raw_avg100",
        choices=[
            "raw",
            "raw_avg100",
            "shaped",
            "shaped_avg100",
            "eval_avg",
            "eval_min",
            "eval_p25",
        ],
        help="Training metric used to decide when to update the best checkpoint.",
    )
    parser.add_argument(
        "--best-min-episodes",
        type=int,
        default=20,
        help="Minimum episodes in this run before saving best checkpoints.",
    )
    parser.add_argument(
        "--no-best-checkpoint",
        action="store_true",
        help="Disable automatic best checkpoint saving.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Run greedy evaluation every N training episodes. Use 0 to disable.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of greedy evaluation episodes when --eval-every is enabled.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=10000,
        help="First seed used for greedy evaluation episodes.",
    )
    parser.add_argument(
        "--rotate-eval-seeds",
        action="store_true",
        help=(
            "Use a different seed block for each training evaluation. By default "
            "the same validation seeds are reused so best checkpoints are comparable."
        ),
    )
    parser.add_argument(
        "--eval-max-steps",
        type=int,
        default=None,
        help="Optional step limit for greedy evaluation episodes.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint to continue training from.",
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="When resuming, load model weights but start with a fresh optimizer.",
    )
    parser.add_argument(
        "--reset-steps",
        action="store_true",
        help="When resuming, start epsilon schedule from step 0.",
    )
    parser.add_argument(
        "--export-torchscript",
        type=Path,
        default=None,
        help="Optional TorchScript model export path.",
    )
    parser.add_argument("--save-every", type=int, default=50, help="Checkpoint period in episodes.")
    parser.add_argument("--log-every", type=int, default=10, help="Console log period in episodes.")
    parser.add_argument(
        "--step-log-every",
        type=int,
        default=0,
        help="Optional environment-step progress log period. Use 0 to disable.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    requested = str(name).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is not available; falling back to CPU.", flush=True)
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


def playable_occupancy(board: Any) -> np.ndarray:
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

    occupied = cropped > 1
    if not np.any(occupied) and np.any(cropped > 0):
        occupied = cropped > 0
    return occupied


def extract_board_features(board: Any) -> BoardFeatures:
    occupied = playable_occupancy(board)
    heights = column_heights_from_occupancy(occupied)
    holes = holes_from_occupancy(occupied, heights)
    bumpiness = int(np.sum(np.abs(np.diff(heights)))) if heights.size > 1 else 0
    return BoardFeatures(
        aggregate_height=int(np.sum(heights)),
        max_height=int(np.max(heights)) if heights.size else 0,
        holes=int(holes),
        bumpiness=bumpiness,
    )


def column_heights_from_occupancy(occupied: np.ndarray) -> np.ndarray:
    height, width = occupied.shape

    column_heights = []
    for col in range(width):
        filled = np.flatnonzero(occupied[:, col])
        if filled.size == 0:
            column_heights.append(0)
            continue
        top = int(filled[0])
        column_heights.append(height - top)
    return np.asarray(column_heights, dtype=np.float32)


def holes_from_occupancy(occupied: np.ndarray, heights: np.ndarray) -> int:
    board_height = occupied.shape[0]
    holes = 0
    for col, column_height in enumerate(heights):
        if column_height <= 0:
            continue
        top = board_height - int(column_height)
        holes += int(np.count_nonzero(~occupied[top:, col]))
    return holes


def board_feature_vector(board: Any) -> np.ndarray:
    occupied = playable_occupancy(board)
    board_height, board_width = occupied.shape
    heights = column_heights_from_occupancy(occupied)
    holes = holes_from_occupancy(occupied, heights)
    aggregate_height = float(np.sum(heights))
    max_height = float(np.max(heights)) if heights.size else 0.0
    bumpiness = float(np.sum(np.abs(np.diff(heights)))) if heights.size > 1 else 0.0
    wells = 0.0
    if heights.size:
        for index, value in enumerate(heights):
            left = heights[index - 1] if index > 0 else board_height
            right = heights[index + 1] if index + 1 < heights.size else board_height
            rim = min(left, right)
            if rim > value:
                wells += float(rim - value)

    height_scale = max(1.0, float(board_height))
    width_scale = max(1.0, float(board_width))
    area_scale = height_scale * width_scale
    return np.asarray(
        [
            aggregate_height / area_scale,
            max_height / height_scale,
            float(holes) / area_scale,
            bumpiness / area_scale,
            float(np.mean(heights)) / height_scale if heights.size else 0.0,
            float(np.std(heights)) / height_scale if heights.size else 0.0,
            wells / area_scale,
            float(np.count_nonzero(heights == 0)) / width_scale,
        ],
        dtype=np.float32,
    )


def board_feature_matrix(boards: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return np.empty((boards.shape[0], 0), dtype=np.float32)
    return np.stack([board_feature_vector(board) for board in boards]).astype(np.float32)


def zero_features() -> BoardFeatures:
    return BoardFeatures(aggregate_height=0, max_height=0, holes=0, bumpiness=0)


def features_from_info(info: Optional[dict], fallback: BoardFeatures) -> BoardFeatures:
    if not isinstance(info, dict):
        return fallback
    board = info.get("board")
    if board is None:
        return fallback
    try:
        return extract_board_features(board)
    except Exception:
        return fallback


def shaped_reward(
    raw_reward: float,
    before: BoardFeatures,
    after: BoardFeatures,
    done: bool,
    info: Optional[dict],
    args: argparse.Namespace,
) -> float:
    if args.reward_mode == "raw":
        return float(raw_reward)

    lines_cleared = int(info.get("lines_cleared", 0)) if isinstance(info, dict) else 0
    value = float(raw_reward) * args.reward_scale
    value += args.line_clear_bonus * float(lines_cleared**2)

    if args.reward_mode == "absolute":
        value -= args.hole_penalty * after.holes
        value -= args.height_penalty * after.aggregate_height
        value -= args.max_height_penalty * after.max_height
        value -= args.bumpiness_penalty * after.bumpiness
    else:
        value -= args.hole_penalty * max(0, after.holes - before.holes)
        value += args.hole_reduction_bonus * max(0, before.holes - after.holes)
        value -= args.height_penalty * max(0, after.aggregate_height - before.aggregate_height)
        value += args.height_reduction_bonus * max(
            0, before.aggregate_height - after.aggregate_height
        )
        value -= args.max_height_penalty * max(0, after.max_height - before.max_height)
        value += args.max_height_reduction_bonus * max(0, before.max_height - after.max_height)
        value -= args.bumpiness_penalty * max(0, after.bumpiness - before.bumpiness)
        value += args.bumpiness_reduction_bonus * max(0, before.bumpiness - after.bumpiness)
        if not done:
            value += args.survival_bonus

    if done:
        value -= args.terminal_penalty
    return float(value)


def heuristic_score(board: Any, args: argparse.Namespace) -> float:
    features = extract_board_features(board)
    return float(
        -args.hole_penalty * features.holes
        - args.height_penalty * features.aggregate_height
        - args.max_height_penalty * features.max_height
        - args.bumpiness_penalty * features.bumpiness
    )


def best_heuristic_action(boards: np.ndarray, mask: np.ndarray, args: argparse.Namespace) -> int:
    legal_actions = np.flatnonzero(mask)
    if legal_actions.size == 0:
        return 0

    best_action = int(legal_actions[0])
    best_value = -float("inf")
    for action in legal_actions:
        value = heuristic_score(boards[int(action)], args)
        if value > best_value:
            best_action = int(action)
            best_value = value
    return best_action


def select_action(
    model: nn.Module,
    boards: np.ndarray,
    board_features: np.ndarray,
    mask: np.ndarray,
    epsilon: float,
    rng: random.Random,
    device: torch.device,
    args: argparse.Namespace,
) -> int:
    legal_actions = np.flatnonzero(mask)
    if legal_actions.size == 0:
        return int(rng.randrange(boards.shape[0]))
    if rng.random() < epsilon:
        if args.exploration_policy == "heuristic":
            return best_heuristic_action(boards, mask, args)
        if (
            args.exploration_policy == "mixed"
            and rng.random() < args.heuristic_exploration_prob
        ):
            return best_heuristic_action(boards, mask, args)
        return int(rng.choice(legal_actions.tolist()))

    tensor = torch.as_tensor(boards, dtype=torch.float32, device=device)
    feature_tensor = None
    if board_features.shape[1] > 0:
        feature_tensor = torch.as_tensor(board_features, dtype=torch.float32, device=device)
    with torch.no_grad():
        values = model(tensor, feature_tensor).detach().reshape(-1).cpu().numpy()
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
    features = torch.as_tensor(
        np.stack([transition.features for transition in batch]),
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
    next_features = torch.as_tensor(
        np.stack([transition.next_features for transition in batch]),
        dtype=torch.float32,
        device=device,
    )
    next_masks = torch.as_tensor(
        np.stack([transition.next_mask for transition in batch]), dtype=torch.bool, device=device
    )
    dones = torch.as_tensor(
        [transition.done for transition in batch], dtype=torch.bool, device=device
    )

    feature_input = features if features.shape[1] > 0 else None
    current_values = policy_net(boards, feature_input).reshape(-1)

    with torch.no_grad():
        batch_count, action_count = next_boards.shape[:2]
        flat_next_boards = next_boards.reshape(batch_count * action_count, *next_boards.shape[2:])
        flat_next_features = next_features.reshape(
            batch_count * action_count,
            next_features.shape[-1],
        )
        flat_next_feature_input = (
            flat_next_features if flat_next_features.shape[1] > 0 else None
        )
        next_policy_values = policy_net(
            flat_next_boards,
            flat_next_feature_input,
        ).reshape(batch_count, action_count)
        next_policy_values = next_policy_values.masked_fill(~next_masks, -1.0e9)
        has_next_action = next_masks.any(dim=1)
        next_actions = next_policy_values.argmax(dim=1, keepdim=True)
        next_target_values = target_net(
            flat_next_boards,
            flat_next_feature_input,
        ).reshape(batch_count, action_count)
        selected_next_values = next_target_values.gather(1, next_actions).reshape(-1)
        selected_next_values = torch.where(
            has_next_action & ~dones,
            selected_next_values,
            torch.zeros_like(selected_next_values),
        )
        targets = rewards + float(gamma) * selected_next_values

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
    extra: Optional[dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_class": "AfterstateQNetwork",
        "model_kwargs": {
            "hidden_size": args.hidden_size,
            "feature_size": FEATURE_SIZE if args.use_board_features else 0,
        },
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "episode": episode,
        "total_steps": total_steps,
        "observation_shape": observation_shape,
        "gamma": args.gamma,
        "reward_mode": args.reward_mode,
        "reward_shaping": {
            "reward_scale": args.reward_scale,
            "line_clear_bonus": args.line_clear_bonus,
            "hole_penalty": args.hole_penalty,
            "height_penalty": args.height_penalty,
            "max_height_penalty": args.max_height_penalty,
            "bumpiness_penalty": args.bumpiness_penalty,
            "terminal_penalty": args.terminal_penalty,
            "survival_bonus": args.survival_bonus,
            "hole_reduction_bonus": args.hole_reduction_bonus,
            "height_reduction_bonus": args.height_reduction_bonus,
            "max_height_reduction_bonus": args.max_height_reduction_bonus,
            "bumpiness_reduction_bonus": args.bumpiness_reduction_bonus,
        },
        "exploration_policy": args.exploration_policy,
        "heuristic_exploration_prob": args.heuristic_exploration_prob,
    }
    if extra is not None:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    reset_optimizer: bool,
) -> tuple[int, int, Optional[tuple[int, ...]]]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Expected train_dqn.py checkpoint, got: {path}")

    loaded_optimizer = load_model_state_forgiving(model, checkpoint["model_state_dict"])
    if loaded_optimizer and not reset_optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    episode = int(checkpoint.get("episode", 0))
    total_steps = int(checkpoint.get("total_steps", 0))
    observation_shape = checkpoint.get("observation_shape")
    if observation_shape is not None:
        observation_shape = tuple(observation_shape)
    return episode, total_steps, observation_shape


def load_model_state_forgiving(model: nn.Module, state_dict: dict[str, Any]) -> bool:
    try:
        model.load_state_dict(state_dict)
        return True
    except RuntimeError as exc:
        print(f"Non-strict checkpoint load: {exc}", flush=True)

    current_state = model.state_dict()
    compatible_state = {}
    partial = []
    skipped = []
    for key, value in state_dict.items():
        if key in current_state and current_state[key].shape == value.shape:
            compatible_state[key] = value
        elif key in current_state and current_state[key].ndim == value.ndim:
            target = current_state[key].clone()
            slices = tuple(
                slice(0, min(target_size, loaded_size))
                for target_size, loaded_size in zip(target.shape, value.shape)
            )
            target[slices] = value[slices]
            compatible_state[key] = target
            partial.append(key)
        else:
            skipped.append(key)

    current_state.update(compatible_state)
    model.load_state_dict(current_state)
    print(
        f"Loaded {len(compatible_state)} compatible tensors; "
        f"partially copied {len(partial)} tensors; skipped {len(skipped)} tensors. "
        "Optimizer state was not loaded.",
        flush=True,
    )
    return False


def export_torchscript(path: Path, model: nn.Module, observation_shape: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = observation_shape[-2:]
    device = next(model.parameters()).device
    example = torch.zeros((1, 1, height, width), dtype=torch.float32, device=device)
    feature_size = int(getattr(model, "feature_size", 0))
    if feature_size > 0:
        feature_example = torch.zeros((1, feature_size), dtype=torch.float32, device=device)
        traced = torch.jit.trace(model, (example, feature_example))
    else:
        traced = torch.jit.trace(model, example)
    traced.save(str(path))


def default_best_output(output: Path) -> Path:
    suffix = output.suffix or ".pt"
    return output.with_name(f"{output.stem}_best{suffix}")


def evaluate_policy(
    policy_net: nn.Module,
    device: torch.device,
    seed: int,
    episodes: int,
    max_steps: Optional[int],
    use_board_features: bool,
) -> dict[str, float]:
    eval_env = make_env("ansi")
    action_count = int(eval_env.action_space.n)
    rewards: list[float] = []
    steps_list: list[int] = []
    try:
        for index in range(int(episodes)):
            observation, info = eval_env.reset(seed=seed + index)
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0

            while not (terminated or truncated):
                boards = action_boards(observation, action_count)
                features = board_feature_matrix(boards, use_board_features)
                mask = legal_mask(eval_env, info, action_count)
                action = select_action(
                    policy_net,
                    boards,
                    features,
                    mask,
                    epsilon=0.0,
                    rng=random.Random(seed + index),
                    device=device,
                    args=None,
                )
                observation, reward, terminated, truncated, info = eval_env.step(action)
                total_reward += float(reward)
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    break

            rewards.append(total_reward)
            steps_list.append(steps)
    finally:
        eval_env.close()

    if not rewards:
        return {
            "eval_avg": 0.0,
            "eval_min": 0.0,
            "eval_p25": 0.0,
            "eval_median": 0.0,
            "eval_steps": 0.0,
        }

    reward_array = np.asarray(rewards, dtype=np.float32)
    return {
        "eval_avg": float(np.mean(reward_array)),
        "eval_min": float(np.min(reward_array)),
        "eval_p25": float(np.percentile(reward_array, 25)),
        "eval_median": float(np.median(reward_array)),
        "eval_steps": float(sum(steps_list) / len(steps_list)) if steps_list else 0.0,
    }


def main():
    args = parse_args()
    if (
        args.best_metric.startswith("eval_")
        and args.eval_every <= 0
        and not args.no_best_checkpoint
    ):
        raise ValueError("--best-metric eval_* requires --eval-every > 0.")

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}", flush=True)
    env = make_env(args.render_mode if args.render else "ansi")
    action_count = int(env.action_space.n)

    model_feature_size = FEATURE_SIZE if args.use_board_features else 0
    policy_net = AfterstateQNetwork(
        hidden_size=args.hidden_size,
        feature_size=model_feature_size,
    ).to(device)
    target_net = AfterstateQNetwork(
        hidden_size=args.hidden_size,
        feature_size=model_feature_size,
    ).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.buffer_size, rng)
    start_episode = 1
    total_steps = 0
    recent_rewards: deque[float] = deque(maxlen=100)
    recent_shaped_rewards: deque[float] = deque(maxlen=100)
    recent_losses: deque[float] = deque(maxlen=100)
    observation_shape: tuple[int, ...] | None = None
    start_time = time.perf_counter()
    best_output = args.best_output or default_best_output(args.output)
    best_metric_value = -float("inf")

    if args.resume is not None:
        loaded_episode, loaded_steps, loaded_shape = load_checkpoint(
            args.resume,
            policy_net,
            optimizer,
            device,
            reset_optimizer=args.reset_optimizer,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr
        target_net.load_state_dict(policy_net.state_dict())
        start_episode = loaded_episode + 1
        total_steps = 0 if args.reset_steps else loaded_steps
        observation_shape = loaded_shape
        print(
            f"Resumed {args.resume} at episode={loaded_episode}, "
            f"total_steps={loaded_steps}.",
            flush=True,
        )

    try:
        final_episode = start_episode + args.episodes - 1
        for episode in range(start_episode, final_episode + 1):
            observation, info = env.reset(seed=args.seed + episode - 1)
            current_features = features_from_info(info, zero_features())
            terminated = False
            truncated = False
            episode_reward = 0.0
            episode_shaped_reward = 0.0
            episode_steps = 0

            while not (terminated or truncated):
                if args.render:
                    env.render()

                boards = action_boards(observation, action_count)
                if observation_shape is None:
                    observation_shape = tuple(boards.shape[1:])
                features = board_feature_matrix(boards, args.use_board_features)
                mask = legal_mask(env, info, action_count)
                epsilon = epsilon_by_step(
                    total_steps,
                    args.epsilon_start,
                    args.epsilon_end,
                    args.epsilon_decay_steps,
                )
                action = select_action(
                    policy_net,
                    boards,
                    features,
                    mask,
                    epsilon,
                    rng,
                    device,
                    args,
                )

                next_observation, reward, terminated, truncated, next_info = env.step(action)
                done = bool(terminated or truncated)
                after_features = extract_board_features(boards[action])
                training_reward = shaped_reward(
                    raw_reward=float(reward),
                    before=current_features,
                    after=after_features,
                    done=done,
                    info=next_info,
                    args=args,
                )
                if done:
                    next_candidate_boards = np.zeros_like(boards)
                    next_candidate_features = np.zeros(
                        (action_count, model_feature_size),
                        dtype=np.float32,
                    )
                    next_mask = np.zeros(action_count, dtype=bool)
                else:
                    next_candidate_boards = action_boards(next_observation, action_count)
                    next_candidate_features = board_feature_matrix(
                        next_candidate_boards,
                        args.use_board_features,
                    )
                    next_mask = legal_mask(env, next_info, action_count)

                replay.push(
                    Transition(
                        board=np.asarray(boards[action], dtype=np.float32),
                        features=np.asarray(features[action], dtype=np.float32),
                        reward=training_reward,
                        next_boards=np.asarray(next_candidate_boards, dtype=np.float32),
                        next_features=np.asarray(next_candidate_features, dtype=np.float32),
                        next_mask=np.asarray(next_mask, dtype=bool),
                        done=done,
                    )
                )

                total_steps += 1
                episode_steps += 1
                episode_reward += float(reward)
                episode_shaped_reward += training_reward
                observation, info = next_observation, next_info
                current_features = features_from_info(next_info, after_features)

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

                if args.step_log_every > 0 and total_steps % args.step_log_every == 0:
                    epsilon = epsilon_by_step(
                        total_steps,
                        args.epsilon_start,
                        args.epsilon_end,
                        args.epsilon_decay_steps,
                    )
                    print(
                        f"Step {total_steps} | "
                        f"episode={episode} | "
                        f"episode_steps={episode_steps} | "
                        f"raw={episode_reward:.2f} | "
                        f"shaped={episode_shaped_reward:.2f} | "
                        f"epsilon={epsilon:.3f} | "
                        f"replay={len(replay)}",
                        flush=True,
                    )

                if args.render:
                    wait_for_frame(args.delay_ms)
                if args.max_steps is not None and episode_steps >= args.max_steps:
                    break

            recent_rewards.append(episode_reward)
            recent_shaped_rewards.append(episode_shaped_reward)
            completed_this_run = episode - start_episode + 1
            eval_stats_this_episode: dict[str, float] = {}

            if episode % args.log_every == 0 or episode == 1:
                elapsed = time.perf_counter() - start_time
                mean_reward = sum(recent_rewards) / len(recent_rewards)
                mean_shaped_reward = sum(recent_shaped_rewards) / len(recent_shaped_rewards)
                mean_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
                epsilon = epsilon_by_step(
                    total_steps,
                    args.epsilon_start,
                    args.epsilon_end,
                    args.epsilon_decay_steps,
                )
                print(
                    f"Episode {episode:04d} | "
                    f"raw={episode_reward:.2f} | "
                    f"raw_avg100={mean_reward:.2f} | "
                    f"shaped={episode_shaped_reward:.2f} | "
                    f"shaped_avg100={mean_shaped_reward:.2f} | "
                    f"steps={episode_steps} | "
                    f"total_steps={total_steps} | "
                    f"epsilon={epsilon:.3f} | "
                    f"loss={mean_loss:.5f} | "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

            if args.eval_every > 0 and completed_this_run % args.eval_every == 0:
                was_training = policy_net.training
                policy_net.eval()
                eval_seed = args.eval_seed
                if args.rotate_eval_seeds:
                    eval_seed += completed_this_run * max(1, args.eval_episodes)
                eval_stats_this_episode = evaluate_policy(
                    policy_net,
                    device,
                    seed=eval_seed,
                    episodes=args.eval_episodes,
                    max_steps=args.eval_max_steps,
                    use_board_features=args.use_board_features,
                )
                if was_training:
                    policy_net.train()
                print(
                    f"Eval episode={episode} | "
                    f"eval_avg={eval_stats_this_episode['eval_avg']:.2f} | "
                    f"eval_p25={eval_stats_this_episode['eval_p25']:.2f} | "
                    f"eval_min={eval_stats_this_episode['eval_min']:.2f} | "
                    f"eval_steps={eval_stats_this_episode['eval_steps']:.2f} | "
                    f"eval_episodes={args.eval_episodes}",
                    flush=True,
                )

            if not args.no_best_checkpoint and observation_shape is not None:
                metric_values = {
                    "raw": episode_reward,
                    "raw_avg100": sum(recent_rewards) / len(recent_rewards),
                    "shaped": episode_shaped_reward,
                    "shaped_avg100": sum(recent_shaped_rewards) / len(recent_shaped_rewards),
                }
                metric_values.update(eval_stats_this_episode)

                metric_value = metric_values.get(args.best_metric)
                if metric_value is not None:
                    metric_value = float(metric_value)
                if (
                    metric_value is not None
                    and completed_this_run >= args.best_min_episodes
                    and metric_value > best_metric_value
                ):
                    best_metric_value = metric_value
                    save_checkpoint(
                        best_output,
                        policy_net,
                        optimizer,
                        args,
                        episode,
                        total_steps,
                        observation_shape,
                        extra={
                            "best_metric": args.best_metric,
                            "best_metric_value": best_metric_value,
                        },
                    )
                    print(
                        f"Saved best checkpoint to {best_output} "
                        f"({args.best_metric}={best_metric_value:.2f})",
                        flush=True,
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
        save_checkpoint(
            args.output,
            policy_net,
            optimizer,
            args,
            final_episode,
            total_steps,
            observation_shape,
        )
        if args.export_torchscript is not None:
            export_torchscript(args.export_torchscript, policy_net, observation_shape)
    finally:
        env.close()

    print(f"Saved checkpoint to {args.output}", flush=True)
    if args.export_torchscript is not None:
        print(f"Saved TorchScript model to {args.export_torchscript}", flush=True)


if __name__ == "__main__":
    main()
