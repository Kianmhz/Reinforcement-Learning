"""Stable-Baselines3 entry point.

Usage:
    python -m turtlebot_rl.train --env goal_nav --algo ppo --total-steps 1_000_000
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["goal_nav", "exploration", "follow"], required=True)
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--logdir", default="runs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # TODO Phase 1: instantiate env via gym.make, build SB3 model, model.learn(...)
    raise NotImplementedError(args)


if __name__ == "__main__":
    main()
