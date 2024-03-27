import argparse
import os
from multiprocessing import Pool

import h5py
import numpy as np
from epicare.envs import EpiCare
from epicare.policies import ClinicalTrial, Oracle, Random, StandardOfCare


def generate_one(seed, prefix):
    filename = f"{args.policy}/{prefix}_seed_{seed}.hdf5"
    if os.path.exists(filename):
        print(filename, "already exists, skipping")
        return

    # Initialize environment
    env = EpiCare(seed=seed)

    policy = policy_factory(env)

    data = {
        "observations": [],
        "actions": [],
        "action_probabilities": [],
        "rewards": [],
        "terminals": [],
        "next_observations": [],
    }

    for episode in range(args.num_episodes):
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

    # Make sure the directory exists.
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    # Save the data as an HDF5 file
    with h5py.File(filename, "w") as f:
        for key, value in data.items():
            f.create_dataset(key, data=value)

    print("Generated", filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=65536 * 2,
        help="Number of episodes to generate",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=8,
        help="Number of seeds to use for data generation",
    )
    parser.add_argument(
        "--policy",
        choices=["random", "smart", "soc", "oracle"],
        default="smart",
        help="Which policy to use for data generation",
    )
    args = parser.parse_args()

    policy_factory = dict(
        random=Random,
        smart=ClinicalTrial,
        soc=StandardOfCare,
        oracle=Oracle,
    )[args.policy]

    print(
        f"Generating {args.seeds} seeds × {args.num_episodes} episodes",
        f"of training and test data from {args.policy}"
    )

    todo = [
        (seed + 1, prefix) for seed in range(args.seeds) for prefix in ("train", "test")
    ]
    with Pool(12) as p:
        p.starmap(generate_one, todo)
