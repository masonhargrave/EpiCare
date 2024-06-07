import functools
import os
from pathlib import Path
from typing import Tuple

import gym
import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from epicare.policies import BasePolicy
from epicare.utils import get_cutoff, load_custom_dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm


def state_and_action_dims(env, config):
    # Action dim is the NUMBER of actions because of OHE.
    action_dim = underlying_action_count = env.action_space.n

    # State dim is affected by frame stacking and previous action inclusion.
    underlying_state_dim = env.observation_space.shape[0]
    state_dim = underlying_state_dim * config.frame_stack
    if config.include_previous_action:
        state_dim += underlying_action_count

    return state_dim, action_dim


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
    had_adverse = False

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
        steps += 1

        if verbose:
            print(f"Step {steps}: Applied Treatment {treatment}, Reward {reward}")

        if collect_series:
            # Collect the series
            observations_collected.append(observation)
            states_collected.append(env.current_disease)

        # Check if remission was achieved and record the time step
        if info.get("remission", False):
            time_to_remission = steps
            if verbose:
                print(f"Remission achieved at step {steps}")

        # Also log adverse events.
        if info.get("adverse_event", False):
            had_adverse = True
            if verbose:
                print(f"Adverse event detected at step {steps}")

        if old_state != new_state:
            transitions += 1

    # Update policy_stats only if it is provided
    if policy_stats is not None:
        if policy_name not in policy_stats:
            policy_stats[policy_name] = {
                "total_rewards": [],
                "remission": [],
                "times_to_remission": [],
                "adverse_event": [],
            }

        policy_stats[policy_name]["total_rewards"].append(total_reward)
        if time_to_remission is not None:
            policy_stats[policy_name]["times_to_remission"].append(time_to_remission)
        policy_stats[policy_name]["adverse_event"].append(1 if had_adverse else 0)
        policy_stats[policy_name]["remission"].append(
            1 if time_to_remission is not None else 0
        )

    if collect_series:
        return observations_collected, states_collected
    else:
        return total_reward, time_to_remission, steps, transitions


def evaluate_online(
    model, env, eval_episodes, device, frame_stack, include_previous_action
):
    model.eval()
    model.to(device)
    returns = []
    remission_rates = []
    adverse_event_rates = []
    times_to_remission = []

    for _ in tqdm(range(eval_episodes), desc="Evaluating"):
        state_history = np.zeros((frame_stack, env.observation_space.shape[0]))
        state = env.reset()
        done = False
        episode_return = 0.0
        time_to_remission = None
        remission_detected = False
        had_adverse = False

        prev_action = np.zeros(env.action_space.n)
        while not done:
            state_history = np.roll(state_history, shift=1, axis=0)
            state_history[0] = state
            if include_previous_action:
                input = np.concatenate((state_history.flatten(), prev_action))
            else:
                input = state_history.flatten()
            action = model.act(input, device=device)
            # Check if action is OHE anad convert to discrete action if so
            if isinstance(action, np.ndarray):
                action = np.argmax(action)  # Convert to discrete action if necessary
            state, reward, done, info = env.step(action)
            episode_return += reward
            prev_action = np.eye(env.action_space.n)[action]

            # Check for remission and record time to remission
            if info.get("remission", False) and not remission_detected:
                time_to_remission = env.visit_number
                remission_detected = True

            # Also log adverse events.
            if info.get("adverse_event", False):
                had_adverse = True

        returns.append(episode_return)
        remission_rates.append(1 if remission_detected else 0)
        adverse_event_rates.append(1 / env.visit_number if had_adverse else 0)
        times_to_remission.append(
            time_to_remission if time_to_remission is not None else np.nan
        )

    mean_return = np.mean(returns) * (100 / 64)
    std_return = np.std(returns) * (100 / 64)
    mean_remission_rate = np.mean(remission_rates)
    mean_adverse_event_rate = np.mean(adverse_event_rates)
    mean_time_to_remission = np.nanmean(
        times_to_remission
    )  # Handle cases where remission is not achieved
    std_time_to_remission = np.nanstd(times_to_remission)

    def sem(x, n_bootstrap=50):
        "Bootstrap to estimate the SEM of returns and adverse event rate."
        boots = np.random.choice(x, (n_bootstrap, len(x)))
        return np.std(np.mean(boots, axis=1))

    sem_return = sem(returns)
    sem_adverse_event_rate = sem(adverse_event_rates)

    print(f"Mean return: {mean_return} ± {sem_return}")
    print(f"Time to remission: {mean_time_to_remission}")

    return (
        mean_return,
        std_return,
        sem_return,
        mean_remission_rate,
        mean_time_to_remission,
        std_time_to_remission,
        mean_adverse_event_rate,
        sem_adverse_event_rate,
    )


def importance_sampling(ratio_0pad, ratio_1pad, reward, discount=1.0):
    "Calculate importance-sampling estimators for OPE."

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


@functools.lru_cache(maxsize=1)
def load_dataset(h5py_filename, cutoff, frame_stack, device="cuda"):
    with h5py.File(h5py_filename, "r") as dataset:
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

    # Compute all action probabilities from the model in one batch.
    # This is the only part of this that uses the model (or the GPU).
    # First calculate the input dimension depending on the amount of frame
    # stacking, then put the observations in using code derivative of the
    # ReplayBuffer class. Finally add the previous action if necessary.
    n_samples = observations.shape[0]
    base_obs_dim = observations.shape[1]
    fss = torch.zeros((n_samples, frame_stack, base_obs_dim)).to(device)

    print("Performing frame stacking etc...")
    obs_tensor = torch.tensor(observations, dtype=torch.float32).to(device)
    for start, end in zip(start_indices[:-1], start_indices[1:]):
        for i in range(start, end):
            for j in range(i, min(i + frame_stack, end)):
                fss[j, j - i] = obs_tensor[i]

    # Squish the observations and return everything.
    fss = fss.view(n_samples, -1)
    return (
        fss,
        actions,
        rewards,
        behavior_probs,
        episode_lengths,
        terminals,
        n_trajectories,
    )


def add_previous_action(fss, actions, terminals, device):
    prev_actions = (
        torch.where(
            torch.tensor(terminals, dtype=bool),
            torch.zeros_like(actions),
            actions,
        )
        .roll(1, dims=0)
        .to(device)
    )
    ohe_actions = torch.nn.functional.one_hot(prev_actions)
    fss_full = torch.cat([fss, ohe_actions], dim=-1)
    return fss_full


def evaluate_offline(
    h5py_filename,
    config,
    eval_policy,
    q_nets=[],
    device="cuda",
    discount=1.0,
    num_folds=8,
    cutoff=None,
):
    """
    Run offline evaluation of a policy.

    :param h5py_filename: The filename of the h5py dataset containing the trajectories
    :param config: The run configuration object
    :param eval_policy: The policy we are evaluating
    :param device: The device to perform calculations on
    :return: A dictionary of estimators
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

    # Check if the config file has an episodes_avail attribute
    if cutoff is not None:
        with h5py.File(h5py_filename, "r") as dataset:
            cutoff = get_cutoff(dataset, config)

    (
        fss,
        actions,
        rewards,
        behavior_probs,
        episode_lengths,
        terminals,
        n_trajectories,
    ) = load_dataset(
        h5py_filename,
        cutoff,
        config.frame_stack,
        device=device,
    )

    fss_full = add_previous_action(fss, actions, terminals, device)
    if config.include_previous_action:
        fss = fss_full

    with torch.no_grad():
        all_eval_probs = eval_policy.get_action_probabilities(fss)

    eval_probs = all_eval_probs.cpu().gather(-1, actions.unsqueeze(-1)).squeeze()
    ratio = eval_probs / behavior_probs

    # Compute the log probabilities of actions under the evaluation policy
    log_eval_probs = torch.log(eval_probs + 1e-45)

    # Reshape log probabilities into episodes
    log_eval_probs = dataset2episodes(log_eval_probs, pad=0)
    ratio_0pad = dataset2episodes(ratio, pad=0)
    ratio_1pad = dataset2episodes(ratio, pad=1)
    rewards = dataset2episodes(rewards, pad=0)

    # Use bootstrapping to produce multiple different OPE estimates using importance
    # sampling so we can show it's not just a question of variance.
    results = {"MeanLogProb": []}
    for i in tqdm(range(num_folds)):
        # Randomly sample trajectory indices with replacement
        sampled_indices = np.random.choice(n_trajectories, n_trajectories, replace=True)

        log_eval_probs_sub = log_eval_probs[:, sampled_indices]
        ratio_0pad_sub = ratio_0pad[:, sampled_indices]
        ratio_1pad_sub = ratio_1pad[:, sampled_indices]
        rewards_sub = rewards[:, sampled_indices]

        # Calculate mean log probability per episode
        mean_log_prob = log_eval_probs_sub.sum(dim=0) / torch.tensor(episode_lengths)
        results["MeanLogProb"].append(mean_log_prob.mean().item())

        estimators = importance_sampling(ratio_0pad_sub, ratio_1pad_sub, rewards_sub)
        for estimator in estimators:
            if estimator not in results:
                results[estimator] = []
            results[estimator].append(estimators[estimator])

    # Implement an ad-hoc OPE method that averages the Q values across all steps.
    meanq_vals = []
    if q_nets:
        for q_net in q_nets:
            with torch.no_grad():
                q_every_step = torch.sum(q_net(fss_full) * all_eval_probs, dim=-1)
                meanq_vals.append(
                    q_every_step.sum().cpu().item() / len(episode_lengths)
                )

        results["Direct"] = meanq_vals

    print(results)
    return results


def process_checkpoints(
    base_path,
    model_name,
    TrainConfig,
    load_model,
    wrap_env,
    load_q_nets=None,
    eval_episodes=2000,
    eval_all=False,
    out_name=None,
):
    results = []

    def compute_mean_std(
        states: np.ndarray, eps: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        mean = states.mean(0)
        std = states.std(0) + eps
        return mean, std

    if base_path is None:
        base_path = "checkpoints"

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
            checkpoints = range(num_checkpoints) if eval_all else [num_checkpoints - 1]
            for i in checkpoints:
                checkpoint_file = f"checkpoint_{i}.pt"
                fraction_of_steps = (i + 1) / num_checkpoints
                checkpoint_path = os.path.join(base_path, dir_name, checkpoint_file)

                # Loading model and environment
                state_mean, state_std = compute_mean_std(
                    load_custom_dataset(config)["observations"], eps=1e-3
                )
                print("Checkpoint path: ", checkpoint_path)
                try:
                    actor = load_model(checkpoint_path, config)
                except Exception as e:
                    print(f"Failed to load model from {checkpoint_path}", e)
                    continue
                env_name = config.env if hasattr(config, "env") else config.env_name
                env = gym.make(env_name, seed=config.env_seed)
                env = wrap_env(env, state_mean=state_mean, state_std=state_std)
                q_nets = load_q_nets(base_path, config) if load_q_nets else None

                # Online evaluation
                (
                    mean_return,
                    std_return,
                    sem_return,
                    mean_remission_rate,
                    mean_time_to_remission,
                    std_time_to_remission,
                    mean_adverse_event_rate,
                    sem_adverse_event_rate,
                ) = evaluate_online(
                    actor,
                    env,
                    eval_episodes,
                    config.device,
                    config.frame_stack,
                    config.include_previous_action,
                )

                # Offline evaluation
                offline_estimates = {}
                hdf5_path = os.path.join(
                    "data/smart", f"test_seed_{config.env_seed}.hdf5"
                )
                estimates = evaluate_offline(
                    hdf5_path,
                    config,
                    actor,
                    q_nets,
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
                    "seed": config.seed,
                    "behavior_policy": config.behavior_policy,
                    "checkpoint": checkpoint_file,
                    "fraction_of_steps": fraction_of_steps,
                    "mean_return": mean_return,
                    "std_return": std_return,
                    "sem_return": sem_return,
                    "mean_remission_rate": mean_remission_rate,
                    "mean_time_to_remission": mean_time_to_remission,
                    "std_time_to_remission": std_time_to_remission,
                    "mean_adverse_event_rate": mean_adverse_event_rate,
                    "sem_adverse_event_rate": sem_adverse_event_rate,
                }
                if hasattr(config, "episodes_avail"):
                    result["episodes_avail"] = config.episodes_avail
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


def combine_stats(data):
    grouped_data = data.groupby("env_seed")

    def pooled_std(group):
        size = len(group)
        return np.sqrt((group["std_return"] ** 2).sum() / size)

    combined_means = grouped_data["mean_return"].mean()
    combined_stds = grouped_data.apply(pooled_std)
    combined_sems = grouped_data["sem_return"].mean()

    combined_stats = pd.DataFrame(
        {
            "combined_mean": combined_means,
            "combined_std": combined_stds,
            "combined_sem": combined_sems,
        }
    ).reset_index()
    print(combined_stats)
    return combined_stats


def grand_stats(df):
    grand_mean = df["combined_mean"].mean()
    grand_std = np.sqrt((df["combined_std"] ** 2).sum() / len(df))
    grand_sem = df["combined_sem"].mean()
    print(f"Grand mean: {grand_mean}")
    print(f"Grand std: {grand_std}")
    print(f"Grand sem: {grand_sem}")
