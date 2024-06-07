import argparse

import drn_evaluate
import gym
import torch
from awac import Actor, TrainConfig, wrap_env

import epicare.evaluations as evaluations


def load_model(checkpoint_path, config):
    env = gym.make(config.env_name)
    state_dim, action_dim = evaluations.state_and_action_dims(env, config)

    # Rehydrate the actor.
    actor = Actor(state_dim, action_dim, config.hidden_dim).to(config.device)
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
        "AWAC",
        TrainConfig,
        load_model,
        wrap_env,
        out_name=args.out_name,
        drn_evaluate=drn_evaluate,
    )

    combined_stats_df = evaluations.combine_stats(results_df)
    evaluations.grand_stats(combined_stats_df)
