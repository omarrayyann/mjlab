"""Waypoint navigation command term."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class WaypointCommand(CommandTerm):
  """Command term that generates waypoint targets for navigation.

  Samples random (x, y, yaw) waypoints relative to the environment origin.
  The command output is (delta_x, delta_y, delta_yaw) in the robot's body
  frame. When the robot reaches a waypoint (within position and heading
  thresholds), it advances to the next one.
  """

  cfg: WaypointCommandCfg

  def __init__(self, cfg: WaypointCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    self.command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.waypoints_w = torch.zeros(
      self.num_envs, cfg.num_waypoints, 3, device=self.device
    )
    self.current_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self.env_origins = env.scene.env_origins

    self.metrics["waypoints_reached"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["heading_error"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.command_b

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    nw = self.cfg.num_waypoints
    r = self.cfg.ranges

    # Sample waypoint offsets relative to env origin.
    xs = torch.empty(n, nw, device=self.device).uniform_(*r.x)
    ys = torch.empty(n, nw, device=self.device).uniform_(*r.y)
    yaws = torch.empty(n, nw, device=self.device).uniform_(*r.yaw)

    # Store as absolute world positions.
    origins = self.env_origins[env_ids]
    self.waypoints_w[env_ids, :, 0] = origins[:, 0:1] + xs
    self.waypoints_w[env_ids, :, 1] = origins[:, 1:2] + ys
    self.waypoints_w[env_ids, :, 2] = yaws

    self.current_idx[env_ids] = 0

  def _update_command(self) -> None:
    # Current target waypoint per env.
    idx = self.current_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, 3)
    target = self.waypoints_w.gather(1, idx).squeeze(1)  # [B, 3]

    # Robot state.
    root_pos = self.robot.data.root_link_pos_w  # [B, 3]
    heading = self.robot.data.heading_w  # [B]

    # World-frame delta.
    dx_w = target[:, 0] - root_pos[:, 0]
    dy_w = target[:, 1] - root_pos[:, 1]

    # Rotate to body frame.
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    dx_b = cos_h * dx_w + sin_h * dy_w
    dy_b = -sin_h * dx_w + cos_h * dy_w

    # Heading error.
    d_yaw = wrap_to_pi(target[:, 2] - heading)

    self.command_b[:, 0] = dx_b
    self.command_b[:, 1] = dy_b
    self.command_b[:, 2] = d_yaw

    # Check if waypoint reached.
    dist = torch.sqrt(dx_w**2 + dy_w**2)
    reached = (dist < self.cfg.position_threshold) & (
      d_yaw.abs() < self.cfg.heading_threshold
    )

    if reached.any():
      self.metrics["waypoints_reached"][reached] += 1.0
      # Advance to next waypoint (clamp at last).
      self.current_idx[reached] = (self.current_idx[reached] + 1).clamp(
        max=self.cfg.num_waypoints - 1
      )
      # Reset per-waypoint timer.
      self.time_left[reached] = self.time_left[reached].uniform_(
        *self.cfg.resampling_time_range
      )

  def _update_metrics(self) -> None:
    dist = torch.norm(self.command_b[:, :2], dim=1)
    self.metrics["position_error"] += dist / (
      self.cfg.resampling_time_range[1] / self._env.step_dt
    )
    self.metrics["heading_error"] += self.command_b[:, 2].abs() / (
      self.cfg.resampling_time_range[1] / self._env.step_dt
    )

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    cur_idx = self.current_idx[0].item()
    waypoints = self.waypoints_w[0].cpu().numpy()
    root_pos = self.robot.data.root_link_pos_w[0].cpu().numpy()

    for i, wp in enumerate(waypoints):
      base = np.array([wp[0], wp[1], 0.01])
      top = np.array([wp[0], wp[1], 0.03])
      if i == cur_idx:
        color = (0.2, 0.9, 0.2, 0.9)
      elif i < cur_idx:
        color = (0.4, 0.4, 0.4, 0.4)
      else:
        color = (0.9, 0.5, 0.1, 0.7)
      visualizer.add_cylinder(base, top, radius=0.15, color=color)

    # Arrow from robot to current waypoint.
    target = waypoints[cur_idx]
    start = np.array([root_pos[0], root_pos[1], root_pos[2] + 1.0])
    end = np.array([target[0], target[1], start[2]])
    visualizer.add_arrow(start, end, color=(0.2, 0.9, 0.2, 0.8), width=0.02)


@dataclass(kw_only=True)
class WaypointCommandCfg(CommandTermCfg):
  """Configuration for waypoint navigation commands."""

  entity_name: str
  num_waypoints: int = 5
  """Number of waypoints to sample per episode."""
  position_threshold: float = 0.3
  """Distance in meters to consider waypoint reached."""
  heading_threshold: float = 0.2
  """Heading error in radians to consider heading achieved."""

  @dataclass
  class Ranges:
    x: tuple[float, float] = (-3.0, 3.0)
    """Waypoint x range relative to env origin."""
    y: tuple[float, float] = (-3.0, 3.0)
    """Waypoint y range relative to env origin."""
    yaw: tuple[float, float] = (-math.pi, math.pi)
    """Target heading range."""

  ranges: Ranges = field(default_factory=Ranges)

  def build(self, env: ManagerBasedRlEnv) -> WaypointCommand:
    return WaypointCommand(self, env)
