import argparse

import epicare.evaluations as evaluations
import gym
import torch

from edac import Actor, TrainConfig, VectorizedCritic, wrap_env


def load_model(checkpoint_path, config):
    # Create an environment to get state_dim and action_dim
    env = gym.make(config.env_name, seed=config.env_seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize the actor with the correct dimensions
    actor = Actor(state_dim, action_dim, config.hidden_dim, config.temperature).to(
        config.device
    )

    # Initialize the critic with the correct dimensions
    critics = VectorizedCritic(
        state_dim, action_dim, config.hidden_dim, num_critics=config.num_critics
    ).to(config.device)

    # Load the state dictionary
    try:
        state_dict = torch.load(checkpoint_path)
        actor.load_state_dict(state_dict["actor"])
        critics.load_state_dict(state_dict["critic"])
    except RuntimeError as e:
        print(e)
        return None, None

    # Average the critics
    def critic(state, action):
        num_batches = 32
        batch_size = state.shape[0] // num_batches
        # Split the state and action into num_batches
        state_batches = torch.split(state, batch_size)
        action_batches = torch.split(action, batch_size)
        Qs = [
            critics(state_batch, action_batch).mean(dim=0).cpu()
            for state_batch, action_batch in zip(state_batches, action_batches)
        ]
        return torch.cat(Qs)

    return actor, critic


# Argparse for base_path and out_name
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="./checkpoints")
    parser.add_argument("--out_name", type=str, default="edac_results")
    args = parser.parse_args()

    eval_episodes = 1000
    model_name = "EDAC"

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

    if results_df is None:
        return

    try:
        combined_stats_df = evaluations.combine_stats(results_df)
        evaluations.grand_stats(combined_stats_df)
    except RuntimeError as e:
        # Print the error
        print(e)
        # Keep running


if __name__ == "__main__":
    main()
