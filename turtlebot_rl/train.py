"""Stable-Baselines3 training entry point.

Usage:
    python -m turtlebot_rl.train --env goal_nav --algo ppo --total-steps 1_000_000

Watch progress in another terminal:
    tensorboard --logdir runs/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from turtlebot_rl.envs import GoalNavEnv

ENV_FACTORIES = {
    "goal_nav": GoalNavEnv,
    # "exploration": ExplorationEnv,  # Phase 2
    # "follow":      FollowEnv,        # Phase 3
}


def build_model(algo: str, env, log_path: Path, seed: int):
    if algo == "ppo":
        return PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(log_path),
            seed=seed,
            policy_kwargs={"net_arch": [64, 64]},
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=3e-4,
            clip_range=0.2,
        )
    if algo == "sac":
        return SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(log_path),
            seed=seed,
            policy_kwargs={"net_arch": [64, 64]},
            learning_rate=3e-4,
            buffer_size=200_000,
            batch_size=256,
        )
    raise ValueError(f"Unknown algo: {algo}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=list(ENV_FACTORIES), required=True)
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--logdir", default="runs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_name = f"{args.algo}_{args.env}"
    log_path = Path(args.logdir) / run_name
    log_path.mkdir(parents=True, exist_ok=True)

    env = Monitor(ENV_FACTORIES[args.env]())
    model = build_model(args.algo, env, log_path, args.seed)

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=str(log_path / "checkpoints"),
        name_prefix=run_name,
    )

    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=checkpoint_cb,
            progress_bar=True,
        )
    finally:
        model.save(log_path / "final.zip")
        env.close()

    print(f"Saved final model to {log_path / 'final.zip'}")


if __name__ == "__main__":
    main()
