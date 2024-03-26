from multiprocessing import Pool

import h5py
import numpy as np
from epicare.envs import EpiCare
from epicare.policies import ClinicalTrial

# Set range of seeds to generate data for
seeds = range(1, 9)

# Number of episodes you want to run
num_episodes = 65536 * 2


def generate_one(seed, prefix):
    # Initialize environment
    env = EpiCare(seed=seed)

    # Choose the policy type here based on your needs
    policy = ClinicalTrial(env)

    data = {
        "observations": [],
        "actions": [],
        "action_probabilities": [],
        "rewards": [],
        "terminals": [],
        "next_observations": [],
    }

    for episode in range(num_episodes):
        policy.reset()
        obs = env.reset()
        done = False
        while not done:
            # Get the treatment probabilities from the policy for importance
            # sampling OPE methods, some of which need to know the probability
            # of the action the sampling policy chose.
            treatment_probs = policy.get_treatment_probs(obs)
            # Don't double-compute them, just sample from that distribution.
            action = np.random.choice(env.n_treatments, p=treatment_probs)
            # Step the environment using this action.
            next_obs, reward, done, _ = env.step(action)
            # Also step the policy to update its internal state, even though
            # technically the clinical trial policy doesn't need to do this.
            policy.update(action, reward)

            data["observations"].append(obs)
            data["actions"].append(action)
            data["action_probabilities"].append(treatment_probs[action])
            data["rewards"].append(reward)
            data["terminals"].append(done)
            data["next_observations"].append(next_obs)

            obs = next_obs  # Update the current observation

    # Convert lists to numpy arrays
    for key in data:
        data[key] = np.array(data[key])

    # Create filename based on seed
    filename = f"{prefix}_seed_{seed}.hdf5"

    # Save the data as an HDF5 file
    with h5py.File(filename, "w") as f:
        for key, value in data.items():
            f.create_dataset(key, data=value)

    print("Generated", filename)


todo = [(seed, prefix) for seed in seeds for prefix in ("train", "test")]

with Pool(12) as p:
    p.starmap(generate_one, todo)
