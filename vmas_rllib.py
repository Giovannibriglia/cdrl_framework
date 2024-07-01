import torch
import os

has_gpu = torch.cuda.is_available()
os.environ["RLLIB_NUM_GPUS"] = "1" if has_gpu else "0"

import os
from typing import Dict
import ray
from ray import tune
from ray.tune import register_env
from vmas import make_env, Wrapper

scenario_name = "navigation"

# Scenario specific variables.
# When modifying this also modify env_config and env_creator
n_agents = 4

# Common variables
continuous_actions = True
max_steps = 200
num_vectorized_envs = 32
num_workers = 1
vmas_device = "cuda"  # or cuda


def env_creator(config: Dict):
    env = make_env(
        scenario=config["scenario_name"],
        num_envs=config["num_envs"],
        device=config["device"],
        continuous_actions=config["continuous_actions"],
        wrapper=Wrapper.RLLIB,
        max_steps=config["max_steps"],
        # Scenario specific variables
        n_agents=config["n_agents"],
    )
    return env


if not ray.is_initialized():
    ray.init()
    print("Ray init!")
register_env(scenario_name, lambda config: env_creator(config))


def train():
    RLLIB_NUM_GPUS = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    num_gpus = 0.001 if RLLIB_NUM_GPUS > 0 else 0  # Driver GPU
    num_gpus_per_worker = (
        (RLLIB_NUM_GPUS - num_gpus) / (num_workers + 1) if vmas_device == "cuda" else 0
    )

    tune.run(
        'PPO',
        stop={"training_iteration": 5000},
        checkpoint_freq=1,
        keep_checkpoints_num=2,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        # callbacks=[
        #     WandbLoggerCallback(
        #        project=f"{scenario_name}",
        #        api_key="",
        #    )
        # ],
        config={
            "seed": 0,
            "framework": "torch",
            "env": scenario_name,
            "kl_coeff": 0.01,
            "kl_target": 0.01,
            "lambda": 0.9,
            "clip_param": 0.2,
            "vf_loss_coeff": 1,
            "vf_clip_param": float("inf"),
            "entropy_coeff": 0,
            "train_batch_size": 60000,
            "rollout_fragment_length": 125,
            "sgd_minibatch_size": 4096,
            "num_sgd_iter": 40,
            "num_gpus": num_gpus,
            "num_workers": num_workers,
            "num_gpus_per_worker": num_gpus_per_worker,
            "num_envs_per_worker": num_vectorized_envs,
            "lr": 5e-5,
            "gamma": 0.99,
            "use_gae": True,
            "use_critic": True,
            "batch_mode": "truncate_episodes",
            "env_config": {
                "device": vmas_device,
                "num_envs": num_vectorized_envs,
                "scenario_name": scenario_name,
                "continuous_actions": continuous_actions,
                "max_steps": max_steps,
                # Scenario specific variables
                "n_agents": n_agents,
            },
            "evaluation_interval": 5,
            "evaluation_duration": 1,
            "evaluation_num_workers": 0,
            "evaluation_parallel_to_training": False,
            "evaluation_config": {
                "num_envs_per_worker": 1,
                "env_config": {
                    "num_envs": 1,
                },
                # "callbacks": MultiCallbacks([RenderingCallbacks, EvaluationCallbacks]),
            },
            # "callbacks": EvaluationCallbacks,
        },
    )


if __name__ == "__main__":
    train()
