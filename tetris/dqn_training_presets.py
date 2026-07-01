"""Reusable DQN training parameter presets.

Add a new entry to TRAINING_PRESETS when an experiment is worth preserving.
Command-line arguments passed to train_dqn.py override values from the selected
preset, so each preset can be used as a stable baseline plus small overrides.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


BASE_PRESET: dict[str, Any] = {
    "steps": 100000,
    "seed": 42,
    "gamma": 0.99,
    "lr": 1e-4,
    "batch_size": 128,
    "buffer_size": 50000,
    "warmup_steps": 5000,
    "train_every": 4,
    "updates_per_step": 1,
    "target_update_interval": 2000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.10,
    "epsilon_decay_steps": 60000,
    "max_episode_steps": 1000,
    "eval_episodes": 20,
    "eval_max_steps": 1000,
    "device": "cpu",
    "torch_threads": 1,
    "log_interval": 1000,
    "checkpoint_interval": 10000,
    "checkpoint_dir": None,
}


TRAINING_PRESETS: dict[str, dict[str, Any]] = {
    "env_baseline": {
        **BASE_PRESET,
        "description": "Raw environment reward baseline from the first stable long run.",
        "reward_mode": "env",
        "reward_clip": 10.0,
        "line_reward": 10.0,
        "line_reward_power": 2.0,
        "survival_reward": 0.1,
        "height_penalty": 0.5,
        "aggregate_height_penalty": 0.0,
        "game_over_penalty": -10.0,
        "output": "models/dqn_afterstate.pt",
    },
    "shaped_v1": {
        **BASE_PRESET,
        "description": (
            "First shaped reward run: line clears plus absolute max-height penalty. "
            "It improved peak reward but learned a late-game suicide failure mode."
        ),
        "reward_mode": "shaped",
        "reward_clip": 200.0,
        "line_reward": 10.0,
        "line_reward_power": 2.0,
        "survival_reward": 0.1,
        "height_penalty": 0.5,
        "aggregate_height_penalty": 0.0,
        "game_over_penalty": -10.0,
        "output": "models/dqn_afterstate_shaped.pt",
    },
    "shaped_v2_no_suicide": {
        **BASE_PRESET,
        "description": (
            "Recommended next run. Keeps line clears as the main signal, makes "
            "survival positive, softens height pressure, and makes game over costly."
        ),
        "reward_mode": "shaped",
        "reward_clip": 200.0,
        "line_reward": 10.0,
        "line_reward_power": 2.0,
        "survival_reward": 1.0,
        "height_penalty": 0.15,
        "aggregate_height_penalty": 0.0,
        "game_over_penalty": -100.0,
        "output": "models/dqn_afterstate_shaped_v2.pt",
    },
}


DEFAULT_PRESET = "shaped_v2_no_suicide"


def preset_names() -> list[str]:
    return sorted(TRAINING_PRESETS)


def get_preset(name: str) -> dict[str, Any]:
    try:
        return deepcopy(TRAINING_PRESETS[name])
    except KeyError as exc:
        available = ", ".join(preset_names())
        raise ValueError(f"Unknown preset {name!r}. Available presets: {available}") from exc
