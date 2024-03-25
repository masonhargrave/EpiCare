import argparse

import epicare.evaluations as evaluations
import gym
import torch

from cql_dqn import FullyConnectedQFunction, Policy, TrainConfig, wrap_env


def load_model(checkpoint_path, config):
    env = gym.make(config.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize the policy model
    q1 = FullyConnectedQFunction(state_dim, action_dim).to(config.device)
    q2 = FullyConnectedQFunction(state_dim, action_dim).to(config.device)

    # Load the state dictionary
    state_dict = torch.load(checkpoint_path)
    q1.load_state_dict(state_dict["q1"])
    q2.load_state_dict(state_dict["q2"])

    actor = Policy(q1, q2, config.device)

    def critic(state, action):
        return (1 / 2) * (q1(state, action) + q2(state, action))

    return actor, critic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="./checkpoints")
    parser.add_argument("--out_name", type=str, default="cql_dqn_results")
    args = parser.parse_args()

    eval_episodes = 1000
    model_name = "CQL-DQN"

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
