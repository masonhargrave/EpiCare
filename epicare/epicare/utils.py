import numpy as np
import h5py
import gym
from typing import Dict

from epicare.envs import EpiCare


def get_cutoff(dataset, config):
    try:
        episodes_avail = config.episodes_avail
    except AttributeError:
        return len(dataset["terminals"])
    # Find index by which episodes_avail terminals are reached
    terminals = dataset["terminals"]
    # cumsum terminals to find episode boundaries
    terminals_cumsum = np.cumsum(terminals)
    # find index of episodes_avail terminals
    cutoff = np.argwhere(terminals_cumsum == episodes_avail)[0][0] + 1
    return cutoff


def load_custom_dataset(config) -> Dict[str, np.ndarray]:
    # Load your custom dataset from an HDF5 file
    with h5py.File(config.dataset_path, "r") as dataset_file:
        cutoff = get_cutoff(dataset_file, config)

        # Here, 'observations', 'actions', etc. are keys in your HDF5 file that correspond to your data.
        # If they are named differently in your file, you'll need to adjust the keys accordingly.
        observations = dataset_file["observations"][:cutoff]
        actions = dataset_file["actions"][:cutoff]
        rewards = dataset_file["rewards"][:cutoff]
        next_observations = dataset_file["next_observations"][:cutoff]
        terminals = dataset_file["terminals"][:cutoff]

    # Convert to float32 for consistency with other Gym environments and D4RL datasets
    observations = observations.astype(np.float32)
    actions = actions.astype(np.float32)
    rewards = rewards.astype(np.float32)
    next_observations = next_observations.astype(np.float32)
    terminals = terminals.astype(np.float32)

    # Ensure terminals are boolean
    terminals = terminals.astype(np.bool_)

    # Create the dataset in the expected format
    custom_dataset = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "next_observations": next_observations,
        "terminals": terminals,
    }

    return custom_dataset
