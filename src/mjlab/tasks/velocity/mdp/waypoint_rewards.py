"""Reward functions for waypoint navigation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def track_waypoint_position(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Reward for getting close to the target waypoint position."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  pos_error_sq = torch.sum(command[:, :2] ** 2, dim=1)
  return torch.exp(-pos_error_sq / std**2)


def track_waypoint_heading(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Reward for matching the target waypoint heading."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  heading_error_sq = command[:, 2] ** 2
  return torch.exp(-heading_error_sq / std**2)


def track_hand_target(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Reward for reaching the hand target position."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error_sq = torch.sum(command**2, dim=1)
  return torch.exp(-error_sq / std**2)
