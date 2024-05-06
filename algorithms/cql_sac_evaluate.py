import argparse

import dqn_evaluate
import gym
import torch
from cql_sac import Policy, TrainConfig, wrap_env

import epicare.evaluations as evaluations


def load_model(checkpoint_path, config):
    env = gym.make(config.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Rehydrate the actor.
    actor = Policy(state_dim, action_dim, config.temperature).to(config.device)
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
        "CQL-SAC",
        TrainConfig,
        load_model,
        wrap_env,
        out_name=args.out_name,
        dqn_evaluate=dqn_evaluate,
    )

    combined_stats_df = evaluations.combine_stats(results_df)
    evaluations.grand_stats(combined_stats_df)
