"""DQN inference agent for grouped Tetris placements.

This agent assumes the environment is wrapped by GroupedActionsObservations.
Each observation is a batch of candidate boards, where observation[action] is
the board after placing the current tetromino with that action.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

import numpy as np


class DQNAgent:
    """Choose the legal action whose candidate board has the highest Q value."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        epsilon: float = 0.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.epsilon = max(0.0, min(1.0, float(epsilon)))
        self.rng = rng or random.Random()

        self.torch = self._import_torch()
        self.device_name = self._resolve_device_name(device)
        self.device = self.torch.device(self.device_name)
        self.model = self._load_model()

    def select_action(self, env: Any, observation: Any = None, info: Optional[dict] = None) -> int:
        """Return the best legal grouped-placement action."""
        legal_actions = self._legal_actions(env, info)
        if not legal_actions:
            return int(env.action_space.sample())

        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal_actions))

        action_boards = self._action_boards(observation, int(env.action_space.n))
        q_values = self._predict_q_values(action_boards)
        return self._best_legal_action(q_values, legal_actions)

    def _import_torch(self):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "DQN mode requires PyTorch. Install it before running with --agent dqn."
            ) from exc
        return torch

    def _resolve_device_name(self, device: str) -> str:
        requested = str(device).lower()
        if requested == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not self.torch.cuda.is_available():
            print("CUDA was requested but is not available; falling back to CPU.")
            return "cpu"
        return str(device)

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"DQN model file not found: {self.model_path}")

        try:
            model = self.torch.jit.load(str(self.model_path), map_location=self.device)
        except Exception:
            try:
                model = self.torch.load(
                    str(self.model_path), map_location=self.device, weights_only=False
                )
            except TypeError:
                model = self.torch.load(str(self.model_path), map_location=self.device)

        if isinstance(model, dict) and "model_state_dict" in model:
            model = self._load_model_from_checkpoint(model)

        if not isinstance(model, self.torch.nn.Module):
            raise ValueError(
                "DQN model must be a TorchScript file, a saved torch.nn.Module, "
                "or a checkpoint created by train_dqn.py."
            )

        model.to(self.device)
        model.eval()
        return model

    def _load_model_from_checkpoint(self, checkpoint: dict):
        model_class = checkpoint.get("model_class", "AfterstateQNetwork")
        if model_class != "AfterstateQNetwork":
            raise ValueError(f"Unsupported DQN checkpoint model_class: {model_class!r}")

        from dqn_model import AfterstateQNetwork

        model_kwargs = dict(checkpoint.get("model_kwargs", {}))
        model = AfterstateQNetwork(**model_kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def _action_boards(self, observation: Any, action_count: int) -> np.ndarray:
        if observation is None:
            raise ValueError("DQN agent requires grouped observations, got None.")

        if isinstance(observation, dict):
            observation = observation.get("board")

        boards = np.asarray(observation, dtype=np.float32)
        if boards.ndim < 3:
            raise ValueError(
                "DQN agent requires grouped observation shaped like "
                "(num_actions, height, width)."
            )
        if boards.shape[0] < action_count:
            raise ValueError(
                f"Grouped observation has {boards.shape[0]} action boards, "
                f"but env.action_space.n is {action_count}."
            )

        return boards[:action_count]

    def _predict_q_values(self, action_boards: np.ndarray) -> np.ndarray:
        tensor = self.torch.as_tensor(action_boards, dtype=self.torch.float32, device=self.device)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)

        feature_tensor = None
        feature_size = int(getattr(self.model, "feature_size", 0))
        if feature_size > 0:
            features = self._board_feature_matrix(action_boards, feature_size)
            feature_tensor = self.torch.as_tensor(
                features,
                dtype=self.torch.float32,
                device=self.device,
            )

        with self.torch.no_grad():
            output = self.model(tensor, feature_tensor)

        q_values = output.detach().cpu().numpy()
        return np.asarray(q_values, dtype=np.float32).reshape(action_boards.shape[0], -1)[:, 0]

    def _board_feature_matrix(self, boards: np.ndarray, feature_size: int) -> np.ndarray:
        features = [self._board_feature_vector(board) for board in boards]
        matrix = np.stack(features).astype(np.float32)
        if matrix.shape[1] != feature_size:
            raise ValueError(
                f"Model expects {feature_size} board features, got {matrix.shape[1]}."
            )
        return matrix

    def _board_feature_vector(self, board: np.ndarray) -> np.ndarray:
        occupied = self._playable_occupancy(board)
        board_height, board_width = occupied.shape
        heights = self._column_heights(occupied)
        holes = self._holes(occupied, heights)
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

    def _playable_occupancy(self, board: np.ndarray) -> np.ndarray:
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

    def _column_heights(self, occupied: np.ndarray) -> np.ndarray:
        board_height, board_width = occupied.shape
        heights = []
        for col in range(board_width):
            filled = np.flatnonzero(occupied[:, col])
            heights.append(0 if filled.size == 0 else board_height - int(filled[0]))
        return np.asarray(heights, dtype=np.float32)

    def _holes(self, occupied: np.ndarray, heights: np.ndarray) -> int:
        board_height = occupied.shape[0]
        holes = 0
        for col, column_height in enumerate(heights):
            if column_height <= 0:
                continue
            top = board_height - int(column_height)
            holes += int(np.count_nonzero(~occupied[top:, col]))
        return holes

    def _best_legal_action(self, q_values: np.ndarray, legal_actions: list[int]) -> int:
        best_action = legal_actions[0]
        best_value = -np.inf
        for action in legal_actions:
            if action < q_values.size and q_values[action] > best_value:
                best_action = action
                best_value = float(q_values[action])
        return int(best_action)

    def _legal_actions(self, env: Any, info: Optional[dict]) -> list[int]:
        mask = info.get("action_mask") if isinstance(info, dict) else None
        if mask is None:
            mask = getattr(env, "legal_actions_mask", None)
        if mask is None:
            return list(range(int(env.action_space.n)))

        mask_array = np.asarray(mask).reshape(-1)
        return [int(index) for index, allowed in enumerate(mask_array) if allowed]
