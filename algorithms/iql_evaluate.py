import argparse

import gym
import torch

import epicare.evaluations as evaluations
from iql import Policy, TrainConfig, wrap_env


def load_model(checkpoint_path, config):
    env = gym.make(config.env, seed=config.env_seed)
    state_dim, action_dim = evaluations.state_and_action_dims(env, config)

    # Rehydrate the actor
    actor = Policy(
        state_dim,
        action_dim,
        temperature=config.temperature,
        dropout=config.actor_dropout,
    ).to(config.device)

    state_dict = torch.load(checkpoint_path)
    actor.load_state_dict(state_dict["actor"])
    return actor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str)
    parser.add_argument("--out_name", type=str)
    args = parser.parse_args()

    results_df = evaluations.process_checkpoints(
        args.base_path,
        "IQL",
        TrainConfig,
        load_model,
        wrap_env,
        out_name=args.out_name,
    )

    combined_stats_df = evaluations.combine_stats(results_df)
    evaluations.grand_stats(combined_stats_df)
