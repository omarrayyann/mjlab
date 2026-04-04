from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
  unitree_g1_with_hands_flat_env_cfg,
  unitree_g1_with_hands_nav_env_cfg,
  unitree_g1_with_hands_reach_env_cfg,
  unitree_g1_with_hands_standing_env_cfg,
  unitree_g1_with_hands_waypoint_env_cfg,
)
from .rl_cfg import unitree_g1_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-G1",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-Hands",
  env_cfg=unitree_g1_with_hands_flat_env_cfg(),
  play_env_cfg=unitree_g1_with_hands_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-Hands-Nav",
  env_cfg=unitree_g1_with_hands_nav_env_cfg(),
  play_env_cfg=unitree_g1_with_hands_nav_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Waypoint-Flat-Unitree-G1-Hands",
  env_cfg=unitree_g1_with_hands_waypoint_env_cfg(),
  play_env_cfg=unitree_g1_with_hands_waypoint_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-Hands-Standing",
  env_cfg=unitree_g1_with_hands_standing_env_cfg(),
  play_env_cfg=unitree_g1_with_hands_standing_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
