"""Expectimax-style placement agent for Tetris-Gymnasium.

The agent is designed for the GroupedActionsObservations wrapper. In that mode
each action directly represents a final placement: column * 4 + rotations.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class BoardFeatures:
    aggregate_height: int
    max_height: int
    holes: int
    bumpiness: int
    wells: int
    max_well_depth: int
    well_depth_squared: int
    row_transitions: int
    column_transitions: int
    blockade_cells: int
    complete_lines: int
    height_std: float


@dataclass(frozen=True)
class HeuristicWeights:
    reward: float = 1.0
    lines_cleared: float = 8.0
    complete_lines: float = 2.0
    aggregate_height: float = -0.55
    max_height: float = -0.35
    holes: float = -1.4375
    bumpiness: float = -0.35
    wells: float = -0.90
    max_well_depth: float = -0.90
    well_depth_squared: float = 0.0
    row_transitions: float = -0.12
    column_transitions: float = -0.08
    blockade_cells: float = -0.35
    height_std: float = -0.20
    game_over: float = -1000.0


class ExpectimaxAgent:
    """Choose Tetris placements with heuristic expectimax search.

    Parameters
    ----------
    depth:
        Number of placements to look ahead. Depth 1 is a greedy search over the
        current tetromino's legal placements. Depth 2+ simulates actions with
        the environment state API and averages over possible next tetrominoes
        when that API is available.
    gamma:
        Discount applied to lookahead values.
    sample_chance:
        If true, average over a random subset of possible next tetrominoes.
        Otherwise all known tetrominoes are evaluated.
    chance_samples:
        Number of tetrominoes to sample when sample_chance is true.
    chance_mode:
        "expected" averages over possible future tetrominoes. "queue" uses the
        actual next tetromino from the environment queue after a simulated step.
    """

    def __init__(
        self,
        depth: int = 1,
        gamma: float = 0.95,
        weights: HeuristicWeights = HeuristicWeights(),
        beam_width: Optional[int] = 8,
        sample_chance: bool = False,
        chance_samples: int = 4,
        chance_mode: str = "expected",
        rng: Optional[random.Random] = None,
    ) -> None:
        self.depth = max(1, int(depth))
        self.gamma = float(gamma)
        self.weights = weights
        self.beam_width = None if beam_width is None else max(1, int(beam_width))
        self.sample_chance = sample_chance
        self.chance_samples = max(1, int(chance_samples))
        if chance_mode not in {"expected", "queue"}:
            raise ValueError("chance_mode must be 'expected' or 'queue'")
        self.chance_mode = chance_mode
        self.rng = rng or random.Random()

    def select_action(self, env: Any, observation: Any = None, info: Optional[dict] = None) -> int:
        """Return the best legal action for the current environment state."""
        legal_actions = self._legal_actions(env, info)
        if not legal_actions:
            return int(env.action_space.sample())

        if self.depth <= 1 or not self._can_simulate(env):
            return self._best_immediate_action(env, observation, info, legal_actions)

        best_action = legal_actions[0]
        best_value = -math.inf
        for action in self._ranked_actions(env, observation, legal_actions):
            value = self._simulate_action(env, action, self.depth)
            if value > best_value:
                best_action = action
                best_value = value
        return int(best_action)

    def _best_immediate_action(
        self, env: Any, observation: Any, info: Optional[dict], legal_actions: Iterable[int]
    ) -> int:
        best_action = None
        best_value = -math.inf

        for action in legal_actions:
            board = self._board_after_action(observation, action)
            if board is None and self._can_simulate(env):
                value = self._simulate_action(env, action, 1)
            elif board is None:
                value = 0.0
            else:
                value = self.evaluate_board(board, env=env)

            if value > best_value:
                best_action = action
                best_value = value

        if best_action is None:
            return int(env.action_space.sample())
        return int(best_action)

    def _simulate_action(self, env: Any, action: int, depth: int) -> float:
        state = self._get_state(env)
        mask = self._copy_action_mask(env)
        try:
            observation, reward, terminated, truncated, info = env.step(int(action))
            done = bool(terminated or truncated)
            board = self._extract_board(info.get("board") if isinstance(info, dict) else None)
            if board is None:
                board = self._extract_board(observation)

            value = self.evaluate_board(
                board,
                env=env,
                reward=float(reward),
                lines_cleared=int(info.get("lines_cleared", 0)) if isinstance(info, dict) else 0,
                game_over=done,
            )

            if done or depth <= 1:
                return value
            if self.chance_mode == "queue":
                return value + self.gamma * self._queue_value(env, depth - 1)
            return value + self.gamma * self._chance_value(env, depth - 1)
        finally:
            self._set_state(env, state)
            self._restore_action_mask(env, mask)

    def _queue_value(self, env: Any, depth: int) -> float:
        observation, info = self._current_grouped_observation(env)
        return self._max_value(env, observation, info, depth)

    def _chance_value(self, env: Any, depth: int) -> float:
        """Average the next max value over possible tetrominoes.

        Tetris-Gymnasium uses a randomizer/queue internally. For expectimax we
        approximate the chance node by replacing the active tetromino with each
        known tetromino and evaluating the best placement from there.
        """
        unwrapped = self._unwrap(env)
        tetrominoes = list(getattr(unwrapped, "tetrominoes", []) or [])
        if not tetrominoes:
            observation, info = self._current_grouped_observation(env)
            return self._max_value(env, observation, info, depth)

        if self.sample_chance and len(tetrominoes) > self.chance_samples:
            tetrominoes = self.rng.sample(tetrominoes, self.chance_samples)

        state = self._get_state(env)
        mask = self._copy_action_mask(env)
        values = []
        try:
            for tetromino in tetrominoes:
                self._set_state(env, state)
                self._restore_action_mask(env, mask)
                unwrapped.active_tetromino = copy.deepcopy(tetromino)
                if hasattr(unwrapped, "reset_tetromino_position"):
                    unwrapped.reset_tetromino_position()

                if hasattr(unwrapped, "collision") and unwrapped.collision(
                    unwrapped.active_tetromino, unwrapped.x, unwrapped.y
                ):
                    values.append(self.weights.game_over)
                    continue

                observation, info = self._current_grouped_observation(env)
                values.append(self._max_value(env, observation, info, depth))
        finally:
            self._set_state(env, state)
            self._restore_action_mask(env, mask)

        return float(np.mean(values)) if values else 0.0

    def _max_value(self, env: Any, observation: Any, info: Optional[dict], depth: int) -> float:
        legal_actions = self._legal_actions(env, info)
        if not legal_actions:
            board = self._extract_board(info.get("board") if isinstance(info, dict) else observation)
            return self.evaluate_board(board, env=env) if board is not None else self.weights.game_over

        observation, _ = self._current_grouped_observation(env)
        actions = self._ranked_actions(env, observation, legal_actions)
        return max(self._simulate_action(env, action, depth) for action in actions)

    def _ranked_actions(self, env: Any, observation: Any, legal_actions: Iterable[int]) -> list[int]:
        actions = list(legal_actions)
        if self.beam_width is None or len(actions) <= self.beam_width:
            return actions

        scored_actions = []
        for action in actions:
            board = self._board_after_action(observation, action)
            value = self.evaluate_board(board, env=env) if board is not None else -math.inf
            scored_actions.append((value, action))

        scored_actions.sort(reverse=True)
        return [action for _, action in scored_actions[: self.beam_width]]

    def evaluate_board(
        self,
        board: Any,
        env: Any = None,
        reward: float = 0.0,
        lines_cleared: int = 0,
        game_over: bool = False,
    ) -> float:
        features = self.extract_features(board, env)
        value = (
            self.weights.reward * reward
            + self.weights.lines_cleared * (lines_cleared**2)
            + self.weights.complete_lines * features.complete_lines
            + self.weights.aggregate_height * features.aggregate_height
            + self.weights.max_height * features.max_height
            + self.weights.holes * features.holes
            + self.weights.bumpiness * features.bumpiness
            + self.weights.wells * features.wells
            + self.weights.max_well_depth * features.max_well_depth
            + self.weights.well_depth_squared * features.well_depth_squared
            + self.weights.row_transitions * features.row_transitions
            + self.weights.column_transitions * features.column_transitions
            + self.weights.blockade_cells * features.blockade_cells
            + self.weights.height_std * features.height_std
        )
        if game_over:
            value += self.weights.game_over
        return float(value)

    def extract_features(self, board: Any, env: Any = None) -> BoardFeatures:
        matrix = self._playable_board(board, env)
        occupied = matrix > 0
        height, width = occupied.shape

        column_heights = []
        holes = 0
        for col in range(width):
            filled = np.flatnonzero(occupied[:, col])
            if filled.size == 0:
                column_heights.append(0)
                continue
            top = int(filled[0])
            column_heights.append(height - top)
            holes += int(np.count_nonzero(~occupied[top:, col]))

        heights = np.array(column_heights, dtype=np.int32)
        bumpiness = int(np.sum(np.abs(np.diff(heights)))) if width > 1 else 0
        complete_lines = int(np.sum(np.all(occupied, axis=1)))
        wells, max_well_depth, well_depth_squared = self._well_stats(heights)
        row_transitions = self._row_transitions(occupied)
        column_transitions = self._column_transitions(occupied)
        blockade_cells = self._blockade_cells(occupied)

        return BoardFeatures(
            aggregate_height=int(np.sum(heights)),
            max_height=int(np.max(heights)) if heights.size else 0,
            holes=int(holes),
            bumpiness=bumpiness,
            wells=wells,
            max_well_depth=max_well_depth,
            well_depth_squared=well_depth_squared,
            row_transitions=row_transitions,
            column_transitions=column_transitions,
            blockade_cells=blockade_cells,
            complete_lines=complete_lines,
            height_std=float(np.std(heights)) if heights.size else 0.0,
        )

    def _row_transitions(self, occupied: np.ndarray) -> int:
        transitions = 0
        for row in occupied:
            padded = np.concatenate(([True], row, [True]))
            transitions += int(np.count_nonzero(padded[1:] != padded[:-1]))
        return transitions

    def _column_transitions(self, occupied: np.ndarray) -> int:
        transitions = 0
        for col in range(occupied.shape[1]):
            padded = np.concatenate(([False], occupied[:, col], [True]))
            transitions += int(np.count_nonzero(padded[1:] != padded[:-1]))
        return transitions

    def _blockade_cells(self, occupied: np.ndarray) -> int:
        blockades = 0
        for col in range(occupied.shape[1]):
            seen_hole = False
            for row in range(occupied.shape[0] - 1, -1, -1):
                if not occupied[row, col]:
                    seen_hole = True
                elif seen_hole:
                    blockades += 1
        return blockades

    def _well_stats(self, heights: np.ndarray) -> tuple[int, int, int]:
        wells = 0
        max_well_depth = 0
        well_depth_squared = 0
        for i, height in enumerate(heights):
            left = heights[i - 1] if i > 0 else math.inf
            right = heights[i + 1] if i + 1 < len(heights) else math.inf
            rim = min(left, right)
            if rim > height:
                depth = int(rim - height)
                wells += depth
                max_well_depth = max(max_well_depth, depth)
                well_depth_squared += depth * depth
        return wells, max_well_depth, well_depth_squared

    def _playable_board(self, board: Any, env: Any = None) -> np.ndarray:
        matrix = np.asarray(board)
        if matrix.ndim > 2:
            matrix = np.squeeze(matrix)
        if matrix.ndim != 2:
            raise ValueError(f"Expected a 2D board, got shape {matrix.shape!r}")

        unwrapped = self._unwrap(env) if env is not None else None
        if unwrapped is not None and hasattr(unwrapped, "crop_padding"):
            try:
                return np.asarray(unwrapped.crop_padding(matrix)) > 1
            except Exception:
                pass

        return self._auto_crop_padding(matrix) > 1

    def _auto_crop_padding(self, matrix: np.ndarray) -> np.ndarray:
        cropped = np.asarray(matrix)
        while cropped.shape[0] > 1 and np.all(cropped[-1, :] == 1):
            cropped = cropped[:-1, :]
        while cropped.shape[1] > 1 and np.all(cropped[:, 0] == 1):
            cropped = cropped[:, 1:]
        while cropped.shape[1] > 1 and np.all(cropped[:, -1] == 1):
            cropped = cropped[:, :-1]
        return cropped

    def _legal_actions(self, env: Any, info: Optional[dict]) -> list[int]:
        mask = None
        if isinstance(info, dict):
            mask = info.get("action_mask")
        if mask is None:
            mask = getattr(env, "legal_actions_mask", None)

        if mask is None:
            return list(range(int(env.action_space.n)))

        mask_array = np.asarray(mask).reshape(-1)
        return [int(i) for i, allowed in enumerate(mask_array) if allowed]

    def _board_after_action(self, observation: Any, action: int) -> Optional[np.ndarray]:
        if observation is None:
            return None
        if isinstance(observation, dict):
            return self._extract_board(observation)

        array = np.asarray(observation)
        if array.ndim >= 3 and 0 <= action < array.shape[0]:
            return np.asarray(array[action])
        return self._extract_board(observation)

    def _extract_board(self, observation: Any) -> Optional[np.ndarray]:
        if observation is None:
            return None
        if isinstance(observation, dict):
            board = observation.get("board")
            return np.asarray(board) if board is not None else None

        array = np.asarray(observation)
        if array.ndim >= 2:
            return array
        return None

    def _current_grouped_observation(self, env: Any) -> tuple[Any, dict]:
        unwrapped = self._unwrap(env)
        if hasattr(unwrapped, "_get_obs") and hasattr(env, "observation"):
            observation = env.observation(unwrapped._get_obs())
            return observation, {"action_mask": getattr(env, "legal_actions_mask", None)}

        observation = unwrapped._get_obs() if hasattr(unwrapped, "_get_obs") else None
        return observation, {"action_mask": getattr(env, "legal_actions_mask", None)}

    def _can_simulate(self, env: Any) -> bool:
        unwrapped = self._unwrap(env)
        return hasattr(unwrapped, "get_state") and hasattr(unwrapped, "set_state")

    def _get_state(self, env: Any) -> Any:
        return self._unwrap(env).get_state()

    def _set_state(self, env: Any, state: Any) -> None:
        self._unwrap(env).set_state(state)

    def _copy_action_mask(self, env: Any) -> Optional[np.ndarray]:
        mask = getattr(env, "legal_actions_mask", None)
        return None if mask is None else np.asarray(mask).copy()

    def _restore_action_mask(self, env: Any, mask: Optional[np.ndarray]) -> None:
        if mask is not None and hasattr(env, "legal_actions_mask"):
            env.legal_actions_mask = mask.copy()

    def _unwrap(self, env: Any) -> Any:
        return getattr(env, "unwrapped", env)
