import argparse

import dqn_evaluate
import gym
import torch
from edac import Actor, TrainConfig, wrap_env

import epicare.evaluations as evaluations


def load_model(checkpoint_path, config):
    # Create an environment to get state_dim and action_dim
    env = gym.make(config.env_name, seed=config.env_seed)
    state_dim, action_dim = evaluations.state_and_action_dims(env, config)

    # Initialize the actor with the correct dimensions
    actor = Actor(
        state_dim,
        action_dim,
        config.hidden_dim,
        config.temperature,
    ).to(config.device)

    # Load the state dictionary
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
        "EDAC",
        TrainConfig,
        load_model,
        wrap_env,
        out_name=args.out_name,
        dqn_evaluate=dqn_evaluate,
    )

    combined_stats_df = evaluations.combine_stats(results_df)
    evaluations.grand_stats(combined_stats_df)
