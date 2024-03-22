import numpy as np


def _marginal_instantaneous_expected_reward(env, treatment):
    expected_value = 0.0
    for d, disease in enumerate(env.diseases.keys()):
        instantaneous_reward = env.expected_instantaneous_reward(disease, treatment)
        expected_value += instantaneous_reward * env.stationary_distribution[d]
    return expected_value


def _q_star_values(env):
    return {
        treatment: _marginal_instantaneous_expected_reward(env, treatment)
        for treatment in range(env.n_treatments)
    }


class StandardOfCare:
    """A state-agnostic greedy policy that selects the treatment with the highest remission probability."""

    def __init__(self, env, alpha=0.5):
        self.env = env
        self.alpha = alpha  # Learning rate for updating estimates
        self.remission_reward = env.remission_reward
        self.reset()

    def get_treatment(self, current_disease=None, current_step=None):
        # Select the treatment with the highest Q-value
        treatment = max(self.Q_values, key=self.Q_values.get)
        return treatment

    def update(self, treatment, reward):
        # Update the Q-value for the chosen treatment based on the reward received
        self.Q_values[treatment] = (1 - self.alpha) * self.Q_values[
            treatment
        ] + self.alpha * reward

    def step(self, observation):
        # Select a treatment based on the highest Q-value
        treatment = self.get_treatment()

        return treatment

    def reset(self):
        # Reset Q-values to initial state
        self.Q_values = _q_star_values(self.env)


class ClinicalTrial:
    """Policy that samples actions in proportion to their Q-values, mimicking Thompson Sampling."""

    def __init__(self, env, alpha=0.5, verbose=False):
        self.env = env
        self.alpha = alpha  # Learning rate for updating estimates
        self.verbose = verbose
        self.remission_reward = env.remission_reward
        self.reset()

    def get_treatment(self, current_disease=None, current_step=None):
        available_treatments = list(self.Q_values.keys())
        return np.random.choice(
            available_treatments, p=self._normalize_Q_values(available_treatments)
        )

    def get_treatment_probs(self, current_disease=None, current_step=None):
        available_treatments = list(self.Q_values.keys())
        return self._normalize_Q_values(available_treatments)

    def _normalize_Q_values(self, available_treatments):
        """Enforce numerical stability by ensuring the max Q-value is 0
        and variability in the actions by making sure the difference
        between the max and min Q-values is ln(2)"""
        Q_values = [self.Q_values[t] for t in available_treatments]
        Q_values = Q_values - np.max(Q_values)
        Q_values = (Q_values / min(Q_values)) * -np.log(8)
        exp_Q_values = np.exp(Q_values)
        probabilities = exp_Q_values / np.sum(exp_Q_values)
        return probabilities

    def reset(self):
        self.Q_values = _q_star_values(self.env)


class Random:
    """Policy for Thompson sampling that probabilistically selects the treatment with the highest remission probability.
    This is intended to imitate a Sequential Multiple Assignment Randomized Trial (SMART) .
    """

    def __init__(self, env, verbose=False):
        self.env = env

    def get_treatment(self, current_disease, current_step):
        available_treatments = np.arange(self.env.n_treatments)
        return np.random.choice(available_treatments)

    def reset(self):
        pass


class Oracle:
    """A state-aware greedy policy that selects the treatment with the highest expected reward.
    This is intended to represent a near-optimal policy."""

    def __init__(self, env):
        self.env = env  # The environment instance
        self.remission_reward = env.remission_reward

    def select_action(self, state):
        # Calculate the expected reward for each treatment in this state.
        expected_rewards = {
            i: self.env.expected_instantaneous_reward(state, action)
            for i, action in enumerate(self.env.treatments)
        }

        # Find the treatment with the highest expected reward
        best_treatment = max(expected_rewards, key=expected_rewards.get)

        return best_treatment

    def step(self, observation):
        current_disease = (
            self.env.current_disease
        )  # Directly accessing the current disease state (which is "cheating")

        # Select the action (treatment) with the highest remission probability
        action = self.select_action(current_disease)

        return action
