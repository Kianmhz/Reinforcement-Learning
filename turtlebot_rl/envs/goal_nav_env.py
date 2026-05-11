"""Phase 1 — goal-reaching navigation.

Observation: 24 downsampled lidar ranges + [dist_to_goal, bearing_to_goal].
Action:      discrete {forward, left, right, stop}.
Reward:      shaped progress + step penalty, +100 on goal, -50 on collision.

Assumes a Gazebo simulator is already running with `/scan`, `/odom`, `/cmd_vel`,
and `/reset_simulation` available (e.g. `ros2 launch turtlebot3_gazebo
empty_world.launch.py`).
"""

from __future__ import annotations

import math
import time

import gymnasium as gym
import numpy as np

from turtlebot_rl.ros_bridge import Pose2D, RosBridge, shutdown_bridge, start_bridge

LIDAR_BEAMS = 24
LIDAR_MAX_RANGE = 3.5      # Burger LDS-01 max range
COLLISION_DIST = 0.18      # Burger radius ~0.105 + safety margin
GOAL_DIST = 0.30           # within this -> success
MAX_LINEAR = 0.22          # Burger spec linear m/s
MAX_ANGULAR = 1.5          # Burger spec angular rad/s
STEP_HZ = 5                # control rate
MAX_STEPS = 500
GOAL_RADIUS_MIN = 0.8
GOAL_RADIUS_MAX = 2.0
ACTIONS = {
    0: (MAX_LINEAR, 0.0),
    1: (0.05, MAX_ANGULAR),
    2: (0.05, -MAX_ANGULAR),
    3: (0.0, 0.0),
}


def _wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class GoalNavEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(LIDAR_BEAMS + 2,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(len(ACTIONS))

        self._node, self._executor, self._thread = start_bridge()
        self._goal = np.zeros(2, dtype=np.float32)
        self._steps = 0
        self._prev_dist = 0.0

    def _downsample_scan(self, scan) -> np.ndarray:
        ranges = np.array(scan.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, LIDAR_MAX_RANGE)
        ranges = np.clip(ranges, 0.0, LIDAR_MAX_RANGE)
        idx = np.linspace(0, len(ranges) - 1, LIDAR_BEAMS).astype(int)
        return ranges[idx]

    def _build_obs(self, scan, pose: Pose2D) -> tuple[np.ndarray, float, float]:
        ranges = self._downsample_scan(scan)
        dx = self._goal[0] - pose.x
        dy = self._goal[1] - pose.y
        dist = math.hypot(dx, dy)
        bearing = _wrap_angle(math.atan2(dy, dx) - pose.yaw)
        obs = np.concatenate([ranges, [dist, bearing]]).astype(np.float32)
        return obs, dist, float(ranges.min())

    def _sample_goal(self) -> np.ndarray:
        r = self.np_random.uniform(GOAL_RADIUS_MIN, GOAL_RADIUS_MAX)
        theta = self.np_random.uniform(-math.pi, math.pi)
        return np.array([r * math.cos(theta), r * math.sin(theta)], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._node.stop()
        self._node.reset_simulation()
        self._goal = self._sample_goal()
        time.sleep(0.3)  # let sim publish fresh /scan and /odom

        scan = self._node.get_scan()
        pose = self._node.get_pose()
        if scan is None or pose is None:
            raise RuntimeError(
                "No /scan or /odom data after reset — is Gazebo running?"
            )

        obs, dist, _ = self._build_obs(scan, pose)
        self._prev_dist = dist
        self._steps = 0
        return obs, {"goal": self._goal.tolist()}

    def step(self, action):
        linear, angular = ACTIONS[int(action)]
        self._node.publish_cmd(linear, angular)
        time.sleep(1.0 / STEP_HZ)

        scan = self._node.get_scan()
        pose = self._node.get_pose()
        if scan is None or pose is None:
            raise RuntimeError("Lost /scan or /odom mid-episode.")

        obs, dist, min_range = self._build_obs(scan, pose)

        terminated = False
        reward = -0.01  # step penalty (encourages getting there sooner)
        reward += 5.0 * (self._prev_dist - dist)  # progress shaping

        if min_range < COLLISION_DIST:
            reward = -50.0
            terminated = True
        elif dist < GOAL_DIST:
            reward = 100.0
            terminated = True

        self._prev_dist = dist
        self._steps += 1
        truncated = self._steps >= MAX_STEPS

        info = {"goal": self._goal.tolist(), "dist": dist, "min_range": min_range}
        return obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            self._node.stop()
        finally:
            shutdown_bridge(self._node, self._executor, self._thread)
