import argparse

import epicare.evaluations as evaluations
import gym
import torch

from dqn import FullyConnectedQFunction, Policy, TrainConfig, wrap_env


def load_model(checkpoint_path, config):
    env = gym.make(config.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize the policy model
    q = FullyConnectedQFunction(state_dim, action_dim).to(config.device)

    # Load the state dictionary
    state_dict = torch.load(checkpoint_path)
    q.load_state_dict(state_dict["q"])
    actor = Policy(q, config.device)

    return actor, q


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="./checkpoints")
    parser.add_argument("--out_name", type=str, default="dqn_results")
    args = parser.parse_args()

    eval_episodes = 1000
    model_name = "DQN"

    results_df = evaluations.process_checkpoints(
        args.base_path,
        model_name,
        TrainConfig,
        load_model,
        wrap_env,
        eval_episodes,
        do_ope=True,
        out_name=args.out_name,
    )

    combined_stats_df = evaluations.combine_stats(results_df)
    evaluations.grand_stats(combined_stats_df)
