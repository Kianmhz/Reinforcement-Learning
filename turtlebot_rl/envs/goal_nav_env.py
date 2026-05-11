"""Phase 1 — goal-reaching navigation.

Observation: 24 downsampled lidar ranges + [dist_to_goal, angle_to_goal].
Action:      discrete {forward, left, right, stop}.
Reward:      step penalty, +large on goal, -large on collision.
"""

import gymnasium as gym
import numpy as np


class GoalNavEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)
        # TODO Phase 1: construct ros_bridge.RosBridge here.

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
