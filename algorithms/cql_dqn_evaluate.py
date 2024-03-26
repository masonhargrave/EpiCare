import argparse

import epicare.evaluations as evaluations
import gym
import torch

from cql_dqn import FullyConnectedQFunction, Policy, TrainConfig, wrap_env


def load_model(checkpoint_path, config):
    env = gym.make(config.env)
    state_dim, action_dim = evaluations.state_and_action_dims(env, config)

    # Initialize both Q functions.
    q1 = FullyConnectedQFunction(state_dim, action_dim).to(config.device)
    q2 = FullyConnectedQFunction(state_dim, action_dim).to(config.device)

    # Load the state dictionary
    state_dict = torch.load(checkpoint_path)
    q1.load_state_dict(state_dict["q1"])
    q2.load_state_dict(state_dict["q2"])
    return Policy(q1, q2, config.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str)
    parser.add_argument("--out_name", type=str)
    args = parser.parse_args()

    results_df = evaluations.process_checkpoints(
        args.base_path,
        "CQL-DQN",
        TrainConfig,
        load_model,
        wrap_env,
        out_name=args.out_name,
    )

    combined_stats_df = evaluations.combine_stats(results_df)
    evaluations.grand_stats(combined_stats_df)
