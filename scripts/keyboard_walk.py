"""Keyboard-controlled G1 walking with a trained velocity policy.

Controls:
  W / S     — forward / backward
  A / D     — strafe left / right
  Q / E     — turn left / right
  X         — stop (zero velocity)
  +/- or =/—  — speed up / slow down sim
  Space     — pause / resume
  Enter     — reset environment

The script loads a trained velocity checkpoint, boots the environment with
a single env, and launches MuJoCo's native viewer. Keyboard presses override
the velocity command so the policy tracks your input.
"""

import sys
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab
import mjlab.tasks  # noqa: F401  — populate registry
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer

TASK_ID = "Mjlab-Velocity-Flat-Unitree-G1"

# Velocity increments per key press.
LIN_VEL_STEP = 0.25  # m/s
ANG_VEL_STEP = 0.25  # rad/s
MAX_LIN_VEL = 2.0
MAX_ANG_VEL = 1.0


def main() -> None:
  if len(sys.argv) < 2:
    print("Usage: uv run python scripts/keyboard_walk.py <checkpoint.pt>")
    print(
      "\nExample:\n"
      "  uv run python scripts/keyboard_walk.py "
      "logs/rsl_rl/g1_velocity/wandb_checkpoints/fo3lwm1r/model_2150.pt"
    )
    sys.exit(1)

  checkpoint_path = Path(sys.argv[1])
  if not checkpoint_path.exists():
    print(f"Checkpoint not found: {checkpoint_path}")
    sys.exit(1)

  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  # Load environment in play mode (infinite episode, no perturbations).
  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(TASK_ID)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  # Load trained policy.
  runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint_path),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  policy = runner.get_inference_policy(device=device)

  # Mutable velocity state captured by the key callback.
  vel = [0.0, 0.0, 0.0]  # [lin_x, lin_y, ang_z]

  def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

  # Key codes (GLFW).
  KEY_W, KEY_S = 87, 83
  KEY_A, KEY_D = 65, 68
  KEY_Q, KEY_E = 81, 69
  KEY_X = 88

  def key_callback(key: int) -> None:
    if key == KEY_W:
      vel[0] = clamp(vel[0] + LIN_VEL_STEP, -MAX_LIN_VEL, MAX_LIN_VEL)
    elif key == KEY_S:
      vel[0] = clamp(vel[0] - LIN_VEL_STEP, -MAX_LIN_VEL, MAX_LIN_VEL)
    elif key == KEY_A:
      vel[1] = clamp(vel[1] + LIN_VEL_STEP, -MAX_LIN_VEL, MAX_LIN_VEL)
    elif key == KEY_D:
      vel[1] = clamp(vel[1] - LIN_VEL_STEP, -MAX_LIN_VEL, MAX_LIN_VEL)
    elif key == KEY_Q:
      vel[2] = clamp(vel[2] + ANG_VEL_STEP, -MAX_ANG_VEL, MAX_ANG_VEL)
    elif key == KEY_E:
      vel[2] = clamp(vel[2] - ANG_VEL_STEP, -MAX_ANG_VEL, MAX_ANG_VEL)
    elif key == KEY_X:
      vel[0] = vel[1] = vel[2] = 0.0

  # Wrap the policy to inject keyboard velocity commands each step.
  command_term = env.unwrapped.command_manager.get_term("twist")

  class KeyboardPolicy:
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
      # Override the velocity command with keyboard input.
      command_term.vel_command_b[0, 0] = vel[0]
      command_term.vel_command_b[0, 1] = vel[1]
      command_term.vel_command_b[0, 2] = vel[2]
      # Mark env 0 as standing=False so the command isn't zeroed.
      command_term.is_standing_env[0] = False
      return policy(obs)

  print("\n--- Keyboard-controlled G1 ---")
  print("W/S = forward/back | A/D = strafe | Q/E = turn | X = stop")
  print("Space = pause | Enter = reset | +/- = sim speed\n")

  viewer = NativeMujocoViewer(env, KeyboardPolicy(), key_callback=key_callback)
  viewer.run()
  env.close()


if __name__ == "__main__":
  main()
