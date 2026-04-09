"""Unitree G1 velocity environment configurations."""

import math

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  G1_WITH_GRIPPER_ACTION_SCALE,
  G1_WITH_HANDS_ACTION_SCALE,
  get_g1_robot_cfg,
  get_g1_with_gripper_robot_cfg,
  get_g1_with_hands_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import (
  HandTargetCommandCfg,
  UniformVelocityCommandCfg,
  WaypointCommandCfg,
)
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    # Waist.
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    # Arms.
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
    r".*wrist.*": 0.3,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


def unitree_g1_with_hands_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 with hands flat terrain velocity configuration."""
  cfg = unitree_g1_flat_env_cfg(play=play)

  robot_cfg = get_g1_with_hands_robot_cfg()
  # Double the arm bend from default keyframe.
  robot_cfg.init_state.joint_pos[".*_elbow_joint"] = -0.2
  cfg.scene.entities = {"robot": robot_cfg}

  # Exclude only hands from the action space (legs + waist + arms).
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = (
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_ankle_.*_joint",
    "waist_.*_joint",
    ".*_shoulder_.*_joint",
    ".*_elbow_joint",
    ".*_wrist_.*_joint",
  )
  joint_pos_action.scale = G1_ACTION_SCALE

  # Foot geoms are unnamed in g1_with_hands.xml, so remove friction
  # randomization that tries to match them by name.
  cfg.events.pop("foot_friction", None)

  # Add hand joint std to pose reward (hands should stay near default).
  for std_key in ("std_walking", "std_running"):
    cfg.rewards["pose"].params[std_key][r".*_hand_.*"] = 0.05

  return cfg


def unitree_g1_with_hands_nav_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 with gripper navigation configuration.

  Same as flat walking but with higher angular velocity range and
  turn-in-place training for waypoint navigation. Uses the gripper model.
  """
  cfg = unitree_g1_with_hands_flat_env_cfg(play=play)

  # Swap to gripper model.
  gripper_cfg = get_g1_with_gripper_robot_cfg()
  gripper_cfg.init_state.joint_pos[".*_elbow_joint"] = -0.2
  gripper_cfg.init_state.joint_pos["left_gripper_joint_a"] = 0.035
  gripper_cfg.init_state.joint_pos["right_gripper_joint_a"] = 0.035
  gripper_cfg.init_state.joint_pos["left_gripper_joint_b"] = -0.035
  gripper_cfg.init_state.joint_pos["right_gripper_joint_b"] = -0.035
  cfg.scene.entities = {"robot": gripper_cfg}

  # Update action space for gripper (legs + waist + arms, no grippers).
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = (
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_ankle_.*_joint",
    "waist_.*_joint",
    ".*_shoulder_.*_joint",
    ".*_elbow_joint",
    ".*_wrist_.*_joint",
  )
  joint_pos_action.scale = G1_ACTION_SCALE

  # Update pose reward std for gripper joints (no hand joints, but gripper).
  for std_key in ("std_walking", "std_running"):
    cfg.rewards["pose"].params[std_key].pop(r".*_hand_.*", None)
    cfg.rewards["pose"].params[std_key][r".*_gripper.*"] = 0.05

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  # 40% of envs practice turning in place.
  twist_cmd.rel_turn_in_place_envs = 0.4

  # Wider std so the reward provides gradient across the full command range.
  # With std=sqrt(2), reward at ±1.5 cmd is ~0.32, at ±2.5 is ~0.04.
  cfg.rewards["track_angular_velocity"].weight = 3.0
  cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(2.0)

  # Gradual angular velocity curriculum matching the reward std.
  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
      {
        "step": 3000 * 24,
        "lin_vel_x": (-1.5, 2.0),
        "ang_vel_z": (-1.0, 1.0),
      },
      {
        "step": 7000 * 24,
        "lin_vel_x": (-1.5, 2.0),
        "ang_vel_z": (-1.5, 1.5),
      },
      {
        "step": 12000 * 24,
        "lin_vel_x": (-2.0, 3.0),
        "ang_vel_z": (-2.5, 2.5),
      },
      {
        "step": 18000 * 24,
        "lin_vel_x": (-2.0, 3.0),
        "ang_vel_z": (-3.5, 3.5),
      },
    ]

  return cfg


def unitree_g1_with_hands_standing_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 with hands standing + manipulation configuration.

  Legs-only policy trained with arm perturbation so it learns to balance
  while an external arm controller moves the upper body.
  """
  cfg = unitree_g1_with_hands_flat_env_cfg(play=play)

  # Zero out velocity commands — this policy only needs to balance in place.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

  # Disable velocity curriculum since we're not walking.
  cfg.curriculum.pop("command_vel", None)

  # Override action space to legs + waist only (arms controlled externally).
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = (
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_ankle_.*_joint",
    "waist_.*_joint",
  )
  joint_pos_action.scale = {
    k: v
    for k, v in G1_ACTION_SCALE.items()
    if "hip" in k or "knee" in k or "ankle" in k or "waist" in k
  }

  # Randomly perturb arm/hand joints during training so the legs policy
  # learns to balance under arbitrary upper-body configurations.
  arm_hand_joints = (
    ".*_shoulder_.*_joint",
    ".*_elbow_joint",
    ".*_wrist_.*_joint",
    ".*_hand_.*_joint",
  )
  cfg.events["perturb_arms"] = EventTermCfg(
    func=envs_mdp.reset_joints_by_offset,
    mode="interval",
    interval_range_s=(1.0, 3.0),
    params={
      "position_range": (-3.5, 3.5),
      "velocity_range": (-2.0, 2.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=arm_hand_joints),
    },
  )

  # Randomize hand mass to simulate holding objects (0-2kg per hand).
  from mjlab.envs.mdp import dr

  cfg.events["hand_payload"] = EventTermCfg(
    func=dr.body_mass,
    mode="startup",
    params={
      "ranges": {0: (0.0, 2.0)},
      "operation": "add",
      "asset_cfg": SceneEntityCfg(
        "robot",
        body_names=(
          "right_wrist_yaw_link",
          "left_wrist_yaw_link",
        ),
      ),
    },
  )

  return cfg


def unitree_g1_with_hands_waypoint_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 with hands waypoint navigation configuration.

  The policy receives (delta_x, delta_y, delta_yaw) to the current waypoint
  in body frame and learns to navigate between randomly sampled waypoints.
  """
  cfg = unitree_g1_with_hands_flat_env_cfg(play=play)

  # Replace velocity command with waypoint command.
  del cfg.commands["twist"]
  cfg.commands["waypoint"] = WaypointCommandCfg(
    entity_name="robot",
    resampling_time_range=(30.0, 30.0),
    num_waypoints=5,
    position_threshold=0.3,
    heading_threshold=0.2,
    debug_vis=True,
    ranges=WaypointCommandCfg.Ranges(
      x=(-3.0, 3.0),
      y=(-3.0, 3.0),
    ),
  )

  # Replace velocity tracking rewards with waypoint tracking rewards.
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=mdp.track_waypoint_position,
    weight=15.0,
    params={"command_name": "waypoint", "std": 2.0},
  )
  cfg.rewards["track_angular_velocity"] = RewardTermCfg(
    func=mdp.track_waypoint_heading,
    weight=3.0,
    params={"command_name": "waypoint", "std": 0.3},
  )

  # Update all reward params that reference "twist" to "waypoint".
  cmd_name = "waypoint"
  cfg.rewards["pose"].weight = 1.0
  cfg.rewards["upright"].weight = 3.0
  cfg.rewards["pose"].params["command_name"] = cmd_name
  # Adjust posture thresholds for position deltas (meters, not m/s).
  cfg.rewards["pose"].params["walking_threshold"] = 0.15
  cfg.rewards["pose"].params["running_threshold"] = 1.0
  # Adjust motion-gating thresholds for position deltas.
  for key in (
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "soft_landing",
  ):
    if key in cfg.rewards and "command_name" in cfg.rewards[key].params:
      cfg.rewards[key].params["command_name"] = cmd_name
      if "command_threshold" in cfg.rewards[key].params:
        cfg.rewards[key].params["command_threshold"] = 0.15

  # Update observations to reference waypoint command.
  cfg.observations["actor"].terms["command"] = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": cmd_name},
  )
  cfg.observations["critic"].terms["command"] = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": cmd_name},
  )

  # Replace velocity curriculum with waypoint distance curriculum.
  from mjlab.managers.curriculum_manager import CurriculumTermCfg

  cfg.curriculum.pop("command_vel", None)
  cfg.curriculum["command_waypoint"] = CurriculumTermCfg(
    func=mdp.commands_waypoint,
    params={
      "command_name": "waypoint",
      "waypoint_stages": [
        {"step": 0, "x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        {"step": 2000 * 24, "x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        {"step": 5000 * 24, "x": (-1.5, 1.5), "y": (-1.5, 1.5)},
        {"step": 10000 * 24, "x": (-3.0, 3.0), "y": (-3.0, 3.0)},
      ],
    },
  )

  return cfg


def unitree_g1_with_hands_reach_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 with hands right-hand reaching configuration.

  Legs + waist balance while right arm + hand reaches random targets.
  Left arm is randomly perturbed for robustness.
  """
  cfg = unitree_g1_with_hands_flat_env_cfg(play=play)

  # Zero velocity — standing only.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  cfg.curriculum.pop("command_vel", None)

  # Add hand target command.
  cfg.commands["hand_target"] = HandTargetCommandCfg(
    entity_name="robot",
    hand_site_name="right_palm",
    resampling_time_range=(5.0, 10.0),
    debug_vis=True,
  )

  # Action space: legs + waist + right arm + right hand.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = (
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_ankle_.*_joint",
    "waist_.*_joint",
    "right_shoulder_.*_joint",
    "right_elbow_joint",
    "right_wrist_.*_joint",
    "right_hand_.*_joint",
  )
  joint_pos_action.scale = {
    k: v
    for k, v in G1_WITH_HANDS_ACTION_SCALE.items()
    if "hip" in k
    or "knee" in k
    or "ankle" in k
    or "waist" in k
    or "right_shoulder" in k
    or "right_elbow" in k
    or "right_wrist" in k
    or "right_hand" in k
  }

  # Add hand target tracking reward.
  cfg.rewards["hand_target"] = RewardTermCfg(
    func=mdp.track_hand_target,
    weight=10.0,
    params={"command_name": "hand_target", "std": 0.1},
  )

  # Add hand target delta to observations.
  cfg.observations["actor"].terms["hand_target"] = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "hand_target"},
  )
  cfg.observations["critic"].terms["hand_target"] = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "hand_target"},
  )

  # Perturb left arm randomly (right arm is controlled by policy).
  left_arm_joints = (
    "left_shoulder_.*_joint",
    "left_elbow_joint",
    "left_wrist_.*_joint",
    "left_hand_.*_joint",
  )
  cfg.events["perturb_left_arm"] = EventTermCfg(
    func=envs_mdp.reset_joints_by_offset,
    mode="interval",
    interval_range_s=(2.0, 5.0),
    params={
      "position_range": (-0.5, 0.5),
      "velocity_range": (-0.5, 0.5),
      "asset_cfg": SceneEntityCfg("robot", joint_names=left_arm_joints),
    },
  )
