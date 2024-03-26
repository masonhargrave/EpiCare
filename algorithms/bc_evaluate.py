import argparse

import epicare.evaluations as evaluations
import gym
import torch

from bc import Actor, TrainConfig, wrap_env


def load_model(checkpoint_path, config):
    env = gym.make(config.env)
    state_dim, action_dim = evaluations.state_and_action_dims(env, config)

    # Rehydrate the actor.
    actor = Actor(state_dim, action_dim)
    state_dict = torch.load(checkpoint_path)
    actor.load_state_dict(state_dict["actor"])

    # Create a stupid q function that always returns 0 because there's no way
    # to extract one from a behavior cloning agent.
    def q(s, a):
        return (0 * a).sum(-1)

    return actor, q


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="./checkpoints")
    parser.add_argument("--out_name", type=str, default="bc_results")
    args = parser.parse_args()

    eval_episodes = 1000
    model_name = "BC"

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
