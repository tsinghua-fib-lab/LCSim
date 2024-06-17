import os
import sys

import yaml
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env

sys.path.append("../../")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from lcsim.envs import BicycleSingleEnv


def env_creator(env_config):
    return BicycleSingleEnv(env_config)


def main():
    register_env("BicycleSingleEnv", env_creator)
    config = yaml.safe_load(open("config/waymo_ppo.yml", "r"))
    tune.run(
        PPO,
        name="waymo_ppo",
        config=config,
        stop={"timesteps_total": 100_000_000},
        checkpoint_config=dict(
            num_to_keep=10,
            checkpoint_score_attribute="episode_reward_mean",
            checkpoint_score_order="max",
            checkpoint_frequency=10,
        ),
        local_dir="logs",
        resume=True,
    )


if __name__ == "__main__":
    main()
