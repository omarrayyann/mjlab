"""Hand target command term for reaching tasks."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class HandTargetCommand(CommandTerm):
  """Command term that generates random 3D target positions for a hand.

  Samples targets in a reachable workspace relative to the robot's torso.
  The command output is (target_x, target_y, target_z) in world frame.
  """

  cfg: HandTargetCommandCfg

  def __init__(self, cfg: HandTargetCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    # Command: target position in world frame [B, 3].
    self.target_w = torch.zeros(self.num_envs, 3, device=self.device)
    # Command output: delta from hand to target in body frame [B, 3].
    self.command_b = torch.zeros(self.num_envs, 3, device=self.device)

    _, self._site_ids = self.robot.find_sites((cfg.hand_site_name,))
    self._torso_body_id = self.robot.find_bodies((cfg.torso_body_name,))[0]

    self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.command_b

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    r = self.cfg.ranges

    # Sample target relative to torso position.
    torso_pos = self.robot.data.body_pos_w[env_ids, self._torso_body_id[0]]  # [n, 3]
    dx = torch.empty(n, device=self.device).uniform_(*r.x)
    dy = torch.empty(n, device=self.device).uniform_(*r.y)
    dz = torch.empty(n, device=self.device).uniform_(*r.z)

    self.target_w[env_ids, 0] = torso_pos[:, 0] + dx
    self.target_w[env_ids, 1] = torso_pos[:, 1] + dy
    self.target_w[env_ids, 2] = torso_pos[:, 2] + dz

  def _update_command(self) -> None:
    # Current hand position.
    hand_pos_w = self.robot.data.site_pos_w[:, self._site_ids[0]]  # [B, 3]

    # Delta in world frame.
    delta_w = self.target_w - hand_pos_w

    # Rotate to body frame using robot heading.
    heading = self.robot.data.heading_w
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)

    self.command_b[:, 0] = cos_h * delta_w[:, 0] + sin_h * delta_w[:, 1]
    self.command_b[:, 1] = -sin_h * delta_w[:, 0] + cos_h * delta_w[:, 1]
    self.command_b[:, 2] = delta_w[:, 2]

  def _update_metrics(self) -> None:
    dist = torch.norm(self.command_b, dim=1)
    self.metrics["position_error"] += dist / (
      self.cfg.resampling_time_range[1] / self._env.step_dt
    )

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    target = self.target_w[0].cpu().numpy()
    hand_pos = self.robot.data.site_pos_w[0, self._site_ids[0]].cpu().numpy()

    # Sphere at target.
    visualizer.add_sphere(target, radius=0.03, color=(0.9, 0.2, 0.2, 0.9))
    # Arrow from hand to target.
    visualizer.add_arrow(hand_pos, target, color=(0.9, 0.2, 0.2, 0.7), width=0.01)


@dataclass(kw_only=True)
class HandTargetCommandCfg(CommandTermCfg):
  """Configuration for hand target commands."""

  entity_name: str
  hand_site_name: str = "right_palm"
  """Site name on the hand to track."""
  torso_body_name: str = "torso_link"
  """Body name to sample targets relative to."""

  @dataclass
  class Ranges:
    x: tuple[float, float] = (0.15, 0.45)
    """Target x range relative to torso (forward)."""
    y: tuple[float, float] = (-0.4, -0.1)
    """Target y range relative to torso (right side for right hand)."""
    z: tuple[float, float] = (-0.2, 0.2)
    """Target z range relative to torso."""

  ranges: Ranges = field(default_factory=Ranges)

  def build(self, env: ManagerBasedRlEnv) -> HandTargetCommand:
    return HandTargetCommand(self, env)
