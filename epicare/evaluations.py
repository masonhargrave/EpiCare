import os
from pathlib import Path
from typing import Tuple

import gym
import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

from epicare.policies import BasePolicy
from epicare.utils import get_cutoff, load_custom_dataset


def run_episode(
    env,
    policy: BasePolicy,
    policy_name,
    policy_stats=None,
    collect_series=False,
    verbose=False,
    normalize=True,
):
    observation = env.reset()
    total_reward = 0
    done = False
    steps = 0
    transitions = 0
    time_to_remission = (
        None  # Initialize as None, which will be updated if remission occurs
    )

    # For series collection
    observations_collected = [observation]
    states_collected = [env.current_disease]

    policy.reset()

    while not done:
        # Obtain the treatment decision from the policy
        treatment = policy.get_treatment(observation)

        # Perform the action in the environment
        old_state = env.current_disease
        observation, reward, done, info = env.step(treatment)
        policy.update(treatment, reward)
        new_state = env.current_disease
        total_reward += reward
        if verbose:
            print(f"Step {steps + 1}: Applied Treatment {treatment}, Reward {reward}")

        if collect_series:
            # Collect the series
            observations_collected.append(observation)
            states_collected.append(env.current_disease)

        # Check if remission was achieved and record the time step
        if info.get("remission", False):
            time_to_remission = steps + 1  # Plus one because steps start from 0
            if verbose:
                print(f"Remission achieved at step {time_to_remission}")
            break  # Optional: break if you want to stop the episode at remission

        steps += 1
        if old_state != new_state:
            transitions += 1

    # Update policy_stats only if it is provided
    if policy_stats is not None:
        if policy_name not in policy_stats:
            policy_stats[policy_name] = {
                "total_rewards": [],
                "remission": [],
                "times_to_remission": [],
            }

        policy_stats[policy_name]["total_rewards"].append(total_reward)
        if time_to_remission is not None:
            policy_stats[policy_name]["times_to_remission"].append(time_to_remission)
        policy_stats[policy_name]["remission"].append(
            1 if time_to_remission is not None else 0
        )

    if collect_series:
        return observations_collected, states_collected
    else:
        return total_reward, time_to_remission, steps, transitions


def evaluate_online(model, env, eval_episodes, device, frame_stack, qvalue=None):
    model.eval()
    model.to(device)
    returns = []
    remission_rates = []
    times_to_remission = []

    for _ in tqdm(range(eval_episodes), desc="Evaluating"):
        state_history = np.zeros((frame_stack, env.observation_space.shape[0]))
        state = env.reset()
        done = False
        episode_return = 0.0
        time_to_remission = None
        remission_detected = False

        while not done:
            state_history = np.roll(state_history, shift=1, axis=0)
            state_history[0] = state
            action = model.act(state_history, device=device)
            # Check if acation is OHE anad convert to discrete action if so
            if isinstance(action, np.ndarray):
                action = np.argmax(action)  # Convert to discrete action if necessary
            state, reward, done, info = env.step(action)
            episode_return += reward

            # Check for remission and record time to remission
            if info.get("remission", False) and not remission_detected:
                time_to_remission = env.visit_number
                # Throw error if time to remission is 0
                if time_to_remission == 0:
                    raise ValueError("Time to remission is 0")
                remission_detected = True

        returns.append(episode_return)
        remission_rates.append(1 if remission_detected else 0)
        times_to_remission.append(
            time_to_remission if time_to_remission is not None else np.nan
        )

    mean_return = np.mean(returns) * (100 / 64)
    std_return = np.std(returns) * (100 / 64)
    mean_remission_rate = np.mean(remission_rates)
    mean_time_to_remission = np.nanmean(
        times_to_remission
    )  # Handle cases where remission is not achieved
    std_time_to_remission = np.nanstd(times_to_remission)

    print(f"Mean return: {mean_return}")
    print(f"Time to remission: {mean_time_to_remission}")

    return (
        mean_return,
        std_return,
        mean_remission_rate,
        mean_time_to_remission,
        std_time_to_remission,
    )


def calculate_estimators(ratio_0pad, ratio_1pad, reward, discount=1.0):
    """
    Evaluate a policy offline with importance sampling.

    :param policy: Policy
    :param data: Dataset
    :param qvalue: QValue
    """

    discount = torch.tensor([discount**t for t in range(reward.size(0))]).view(-1, 1)
    discounted_reward = reward * discount

    ratio_IS = ratio_1pad
    ratio_IS = torch.prod(ratio_IS, dim=0) + 1e-45
    ep_IS = ratio_IS * torch.sum(discounted_reward, dim=0)
    IS = ep_IS.mean()
    WIS = ep_IS.sum() / ratio_IS.sum()

    ratio_PDIS = ratio_0pad
    ratio_PDIS = torch.cumprod(ratio_PDIS, dim=0) + 1e-45
    ep_PDIS = (ratio_PDIS * discounted_reward).sum(dim=0)
    PDIS = ep_PDIS.mean()
    weighted_ratio_PDIS = ratio_PDIS / ratio_PDIS.sum(dim=-1, keepdim=True)
    WPDIS = (weighted_ratio_PDIS * discounted_reward).sum()
    # weighted_ratio_CWPDIS = ep_PDIS / ratio_PDIS.sum(dim=1, keepdim=True)
    # CWPDIS = weighted_ratio_CWPDIS.sum()

    estimators = {
        "IS": IS.item(),
        "WIS": WIS.item(),
        "PDIS": PDIS.item(),
        "WPDIS": WPDIS.item(),
        # "CWPDIS": CWPDIS.item(),
    }

    return estimators


def evaluate_offline(
    h5py_filename,
    config,
    eval_policy,
    q_network,
    device="cuda",
    discount=1.0,
    num_folds=8,
    cutoff=None,
):
    """
    Calculate the CWPDIS estimate for the policy based on the provided h5py dataset.

    :param h5py_filename: The filename of the h5py dataset containing the trajectories
    :param eval_policy: The policy we are evaluating
    :param device: The device to perform calculations on
    :return: CWPDIS estimate
    """

    def dataset2episodes(X, pad):
        """
        :param X: torch.Tensor (len(data), ...)
        :param pad: padding value
        :returns: torch.Tensor (max_episode_length, num_episodes, ...)
        """
        X = torch.split(X, episode_lengths)
        X = pad_sequence(X, padding_value=pad)
        return X

    with h5py.File(h5py_filename, "r") as dataset:
        # Check if the config file has an episodes_avail attribute
        if cutoff:
            cutoff = get_cutoff(dataset, config)
        print(f"Using cutoff: {cutoff}")
        observations = np.array(dataset["observations"][:cutoff])
        actions = torch.tensor(dataset["actions"][:cutoff], dtype=torch.long)
        rewards = torch.tensor(dataset["rewards"][:cutoff])
        terminals = np.array(dataset["terminals"][:cutoff])
        behavior_probs = np.array(dataset["action_probabilities"][:cutoff])

        # Identify the start of each trajectory
        start_indices = [0] + list(np.nonzero(terminals)[0] + 1)
        episode_lengths = tuple(np.diff(start_indices))
        n_trajectories = len(episode_lengths)

        # GPU: Compute all action probabilities from the model in one batch
        obs_tensor = torch.tensor(observations, dtype=torch.float32).to(device)
        with torch.no_grad():
            all_eval_probs = (
                eval_policy.get_action_probabilities(obs_tensor).squeeze(0).cpu()
            )

        eval_probs = all_eval_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        ratio = eval_probs / behavior_probs

        # Compute the log probabilities of actions under the evaluation policy
        log_eval_probs = torch.log(eval_probs + 1e-45)
        # Resahep log probabilities into episodes
        log_eval_probs = dataset2episodes(log_eval_probs, pad=0)

        action_dim = all_eval_probs.size(-1)
        # GPU: Compute Q-values for all actions for each state in one batch
        with torch.no_grad():
            # Iterate over all ohe actions
            all_Qs = []
            for action in range(action_dim):
                # Create a tensor with the action repeated for each state
                action_tensor = torch.tensor(
                    [action] * len(observations), dtype=torch.long
                ).to(device)
                # One hot encode the action
                action_tensor = torch.nn.functional.one_hot(
                    action_tensor, num_classes=action_dim
                )
                # Compute the Q-values for the action
                Qs = q_network(obs_tensor, action_tensor)

                all_Qs.append(Qs.cpu())

        # Stack the Q-values for each action into a single tensor
        all_Qs = torch.stack(all_Qs, dim=1)
        # Use all_eval_probs to get expected reward in each state
        left_term = (all_Qs * all_eval_probs).sum(dim=-1)
        Qs = all_Qs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        right_term = ratio * (rewards - Qs)
        all_DR_estimates = left_term + right_term

        DR_estimates = dataset2episodes(all_DR_estimates, pad=0)
        ratio_0pad = dataset2episodes(ratio, pad=0)
        ratio_1pad = dataset2episodes(ratio, pad=1)
        rewards = dataset2episodes(rewards, pad=0)

        results = {"DR": [], "MeanLogProb": []}
        for i in tqdm(range(num_folds)):
            # Randomly sample trajectory indices with replacement
            sampled_indices = np.random.choice(
                n_trajectories, n_trajectories, replace=True
            )

            log_eval_probs_sub = log_eval_probs[:, sampled_indices]
            ratio_0pad_sub = ratio_0pad[:, sampled_indices]
            ratio_1pad_sub = ratio_1pad[:, sampled_indices]
            rewards_sub = rewards[:, sampled_indices]
            DR_estimates_sub = DR_estimates[:, sampled_indices]
            num_steps_sub = np.array(episode_lengths)[sampled_indices].sum()
            results["DR"].append(DR_estimates_sub.sum().item() / num_steps_sub)

            # Calculate mean log probability per episode
            mean_log_prob = log_eval_probs_sub.sum(dim=0) / torch.tensor(
                episode_lengths
            )
            results["MeanLogProb"].append(mean_log_prob.mean().item())

            estimators = calculate_estimators(
                ratio_0pad_sub, ratio_1pad_sub, rewards_sub
            )
            for estimator in estimators:
                if estimator not in results:
                    results[estimator] = []
                results[estimator].append(estimators[estimator])

        print(results)
        return results


def process_checkpoints(
    base_path,
    model_name,
    TrainConfig,
    load_model,
    wrap_env,
    eval_episodes=1000,
    do_ope=False,
    out_name=False,
):
    results = []

    def compute_mean_std(
        states: np.ndarray, eps: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        mean = states.mean(0)
        std = states.std(0) + eps
        return mean, std

    # Iterate over directories
    for dir_name in os.listdir(base_path):
        if dir_name.startswith(model_name):
            print(f"Processing {dir_name}")
            config_path = os.path.join(base_path, dir_name, "config.yaml")

            with open(config_path, "r") as f:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
                try:
                    num_checkpoints = config_dict["num_checkpoints"]
                except KeyError:
                    num_checkpoints = 1
            config = TrainConfig().update_params(config_dict)
            # Iterate over checkpoint files within the directory
            for i in range(1, num_checkpoints + 1):
                # if num_checkpoints == 1:
                #    checkpoint_file = "checkpoint.pt"
                # else:
                checkpoint_file = f"checkpoint_{i-1}.pt"
                fraction_of_steps = i / num_checkpoints
                checkpoint_path = os.path.join(base_path, dir_name, checkpoint_file)

                # Loading model and environment
                state_mean, state_std = compute_mean_std(
                    load_custom_dataset(config)["observations"], eps=1e-3
                )
                print("Checkpoint path: ", checkpoint_path)
                actor, critic = load_model(checkpoint_path, config)
                if actor is None or critic is None:
                    print(f"Failed to load model from {checkpoint_path}")
                    continue
                env_name = config.env if hasattr(config, "env") else config.env_name
                env = gym.make(env_name, seed=config.env_seed)
                env = wrap_env(env, state_mean=state_mean, state_std=state_std)

                # Online evaluation
                (
                    mean_return,
                    std_return,
                    mean_remission_rate,
                    mean_time_to_remission,
                    std_time_to_remission,
                ) = evaluate_online(
                    actor, env, eval_episodes, config.device, config.frame_stack
                )

                # Offline evaluation (if applicable)
                offline_estimates = {}
                if do_ope:
                    hdf5_path = os.path.join(
                        "data", f"test_seed_{config.env_seed}.hdf5"
                    )
                    # hdf5_path = os.path.join("./data", f"seed_{config.env_seed}.hdf5")
                    estimates = evaluate_offline(
                        hdf5_path,
                        config,
                        actor,
                        critic,
                        device=config.device,
                        discount=1.0,
                    )
                    for estimator in estimates:
                        key = estimator.lower()
                        offline_estimates[f"mean_{key}_estimate"] = np.mean(
                            estimates[estimator]
                        ) * (100 / 64)
                        offline_estimates[f"std_{key}_estimate"] = np.std(
                            estimates[estimator]
                        ) * (100 / 64)

                result = {
                    "env_seed": config.env_seed,
                    "episodes_avail": getattr(config, "episodes_avail", None),
                    "checkpoint": checkpoint_file,
                    "fraction_of_steps": fraction_of_steps,
                    "mean_return": mean_return,
                    "std_return": std_return,
                    "mean_remission_rate": mean_remission_rate,
                    "mean_time_to_remission": mean_time_to_remission,
                    "std_time_to_remission": std_time_to_remission,
                }
                result.update(offline_estimates)
                results.append(result)

    # Save results to csv
    results_df = pd.DataFrame(results)
    Path("./results").mkdir(parents=True, exist_ok=True)
    csv_filename = (
        f"./results/{out_name}.csv"
        if out_name
        else f"./results/{model_name.lower()}_results.csv"
    )
    results_df.to_csv(csv_filename, index=False)

    return results_df


def combine_stats(data, mean_key="mean_return", std_key="std_return"):
    grouped_data = data.groupby("env_seed")
    combined_means = grouped_data[mean_key].mean()

    def pooled_std(group):
        size = len(group)
        return np.sqrt((group[std_key] ** 2).sum() / size)

    combined_stds = grouped_data.apply(pooled_std)
    combined_stats = pd.DataFrame(
        {"combined_mean": combined_means, "combined_std": combined_stds}
    ).reset_index()
    print(combined_stats)
    return combined_stats


def grand_stats(df):
    size = len(df)
    grand_mean = df["combined_mean"].mean()
    grand_std = np.sqrt((df["combined_std"] ** 2).sum() / size)
    print(f"Grand mean: {grand_mean}")
    print(f"Grand std: {grand_std}")
