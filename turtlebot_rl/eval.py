"""Roll out a saved policy in the live simulator and print the success rate.

Usage:
    python -m turtlebot_rl.eval --env goal_nav --model runs/ppo_goal_nav/final.zip
"""

from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO, SAC

from turtlebot_rl.envs import GoalNavEnv
from turtlebot_rl.envs.goal_nav_env import GOAL_DIST

ENV_FACTORIES = {
    "goal_nav": GoalNavEnv,
}
ALGOS = {"ppo": PPO, "sac": SAC}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=list(ENV_FACTORIES), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--algo", choices=list(ALGOS), default="ppo")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    model = ALGOS[args.algo].load(args.model)
    env = ENV_FACTORIES[args.env]()

    successes = 0
    returns = []
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done = False
            ep_return = 0.0
            info: dict = {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += reward
                done = terminated or truncated
            ok = info.get("dist", float("inf")) < GOAL_DIST
            successes += int(ok)
            returns.append(ep_return)
            print(
                f"ep {ep + 1:>2}: return={ep_return:7.2f} "
                f"final_dist={info.get('dist', float('nan')):.2f}m "
                f"{'OK' if ok else 'FAIL'}"
            )
    finally:
        env.close()

    print()
    print(f"Success rate: {successes}/{args.episodes} "
          f"= {100 * successes / args.episodes:.1f}%")
    print(f"Mean return:  {np.mean(returns):.2f} +/- {np.std(returns):.2f}")


if __name__ == "__main__":
    main()
