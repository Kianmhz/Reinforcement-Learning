"""Roll out a saved policy in the live simulator.

Usage:
    python -m turtlebot_rl.eval --env goal_nav --model runs/ppo_goal/best.zip --episodes 20
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["goal_nav", "exploration", "follow"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    # TODO Phase 1: load model, instantiate env, run deterministic rollouts, print success rate.
    raise NotImplementedError(args)


if __name__ == "__main__":
    main()
