"""Neural network used by the after-state DQN Tetris agent."""

from __future__ import annotations

import torch
from torch import nn


class AfterstateQNetwork(nn.Module):
    """Estimate the value of one placed-piece board.

    The grouped Tetris wrapper presents every legal action as a candidate board
    after the current piece has been placed. This network scores one such board
    at a time; callers can batch all candidate boards and choose the highest
    scoring legal action.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        input_scale: float = 2.0,
        feature_size: int = 0,
    ) -> None:
        super().__init__()
        self.input_scale = float(input_scale)
        self.hidden_size = int(hidden_size)
        self.feature_size = int(feature_size)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 4 * 4 + self.feature_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

    def forward(
        self,
        boards: torch.Tensor,
        board_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if boards.ndim == 3:
            boards = boards.unsqueeze(1)
        if boards.ndim != 4:
            raise ValueError(f"Expected board tensor with 3 or 4 dims, got {boards.ndim}.")

        boards = boards.float()
        if self.input_scale > 0:
            boards = torch.clamp(boards, 0.0, self.input_scale) / self.input_scale
        embedding = self.features(boards)

        if self.feature_size > 0:
            if board_features is None:
                raise ValueError("board_features is required when feature_size > 0.")
            board_features = board_features.float()
            if board_features.ndim == 1:
                board_features = board_features.unsqueeze(0)
            if board_features.shape[-1] != self.feature_size:
                raise ValueError(
                    "Expected board_features last dimension "
                    f"{self.feature_size}, got {board_features.shape[-1]}."
                )
            embedding = torch.cat([embedding, board_features], dim=1)

        return self.head(embedding)
