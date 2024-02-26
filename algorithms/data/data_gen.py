import h5py
import numpy as np
from epicare.envs import EpiCare
from epicare.policies import ClinicalTrial
from tqdm import tqdm

# Set range of seeds to generate data for
seeds = range(1, 9)

with tqdm(total=len(seeds) * 2) as pbar:
    for dataset in "train", "test":
        for seed in seeds:
            # Initialize environment
            env = EpiCare(seed=seed)

            print(env.num_symptoms_for_disease_range)

            # Choose the policy type here based on your needs
            policy = ClinicalTrial(env)

            num_episodes = 65536  # Number of episodes you want to run
            data = {
                "observations": [],
                "actions": [],
                "action_probabilities": [],
                "rewards": [],
                "terminals": [],
                "next_observations": [],
            }

            for episode in range(num_episodes):
                obs = env.reset()
                done = False
                while not done:
                    # Get the treatment probabilities from the policy
                    treatment_probs = policy.get_treatment_probs(
                        env.current_disease, env.visit_number
                    )
                    # Use the get_treatment method from the chosen policy
                    action = policy.get_treatment(env.current_disease, env.visit_number)
                    next_obs, reward, done, _ = env.step(action)

                    data["observations"].append(obs)
                    data["actions"].append(action)
                    data["action_probabilities"].append(
                        treatment_probs[action]
                    )  # Store the probability of the chosen action
                    data["rewards"].append(reward)
                    data["terminals"].append(done)
                    data["next_observations"].append(next_obs)

                    obs = next_obs  # Update the current observation

                policy.reset()

            # Convert lists to numpy arrays
            for key in data:
                data[key] = np.array(data[key])

            # Create filename based on seed
            filename = f"{dataset}_seed_{seed}.hdf5"

            # Save the data as an HDF5 file
            with h5py.File(filename, "w") as f:
                for key, value in data.items():
                    f.create_dataset(key, data=value)

            pbar.update()
