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
        self.device_name = device
        self.epsilon = max(0.0, min(1.0, float(epsilon)))
        self.rng = rng or random.Random()

        self.torch = self._import_torch()
        self.device = self.torch.device(device)
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

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"DQN model file not found: {self.model_path}")

        try:
            model = self.torch.jit.load(str(self.model_path), map_location=self.device)
        except Exception:
            model = self.torch.load(str(self.model_path), map_location=self.device)

        if not isinstance(model, self.torch.nn.Module):
            raise ValueError(
                "DQN model must be a TorchScript file or a saved torch.nn.Module. "
                "For best compatibility, save with torch.jit.save(...)."
            )

        model.to(self.device)
        model.eval()
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

        with self.torch.no_grad():
            output = self.model(tensor)

        q_values = output.detach().cpu().numpy()
        return np.asarray(q_values, dtype=np.float32).reshape(action_boards.shape[0], -1)[:, 0]

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
