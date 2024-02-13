import argparse
import torch
import gym
from iql import StochasticPolicy, DeterministicPolicy, wrap_env, TrainConfig, TwinQ

import epicare.evaluations as evaluations


def load_model(checkpoint_path, config):
    env = gym.make(config.env, seed=config.env_seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Choose the policy type based on the configuration
    if config.iql_deterministic:
        actor = DeterministicPolicy(state_dim, action_dim).to(config.device)
    else:
        actor = StochasticPolicy(
            state_dim,
            action_dim,
            temperature=config.temperature,
            dropout=config.actor_dropout,
        ).to(config.device)

    critic = TwinQ(state_dim, action_dim).to(config.device)

    # Load the saved model
    state_dict = torch.load(checkpoint_path)
    actor.load_state_dict(state_dict["actor"])
    critic.load_state_dict(state_dict["qf"])

    return actor, critic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="./checkpoints")
    parser.add_argument("--out_name", type=str, default="iql_results")
    args = parser.parse_args()

    eval_episodes = 1000
    model_name = "IQL"

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
