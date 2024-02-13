import argparse

import gym
import torch
from cql import FullyConnectedQFunction, Policy, TrainConfig, wrap_env

import epicare.evaluations as evaluations


def load_model(checkpoint_path, config):
    env = gym.make(config.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize the policy model
    actor = Policy(state_dim, action_dim, config.temperature).to(config.device)
    critic_1 = FullyConnectedQFunction(state_dim, action_dim).to(config.device)
    critic_2 = FullyConnectedQFunction(state_dim, action_dim).to(config.device)

    # Load the state dictionary
    state_dict = torch.load(checkpoint_path)
    actor.load_state_dict(state_dict["actor"])
    critic_1.load_state_dict(state_dict["critic1"])
    critic_2.load_state_dict(state_dict["critic2"])

    def critic(state, action):
        return (1 / 2) * (critic_1(state, action) + critic_2(state, action))

    return actor, critic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="./checkpoints")
    parser.add_argument("--out_name", type=str, default="cql_results")
    args = parser.parse_args()

    eval_episodes = 1000
    model_name = "CQL"

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
