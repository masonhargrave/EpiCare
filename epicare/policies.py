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


class BasePolicy:
    def __init__(self, env):
        self.env = env
        self.reset()

    def get_treatment_probs(self, observation):
        """
        Calculate the probability of each treatment given the current observation.
        Overload this to create a stochastic policy, and the concrete treatment for each
        step will be chosen from the returned distribution.
        """
        out = np.zeros(self.env.n_treatments)
        out[self.get_treatment(observation)] = 1
        return out

    def get_treatment(self, observation):
        """
        Select a treatment given the current observation. Overload this to create a
        deterministic policy, and treatment probabilities will automatically be one-hot
        encoded from the output of this method.
        """
        return np.random.choice(
            self.env.n_treatments, p=self.get_treatment_probs(observation)
        )

    def update(self, treatment, reward):
        """
        Update the policy's internal state based on the treatment and reward received.
        """
        pass

    def reset(self):
        """
        Reset the policy's per-episode internal state.
        """
        pass


class StandardOfCare(BasePolicy):
    """
    A state-agnostic greedy policy that selects the treatment with the highest remission probability.
    """

    def __init__(self, env, alpha=0.5, kappa=0.2):
        self.alpha = alpha  # Learning rate for updating estimates
        self.max_allowable_symptom = 1 - kappa
        super().__init__(env)

    def get_treatment(self, observation):
        # Limit the possible treatments to only those which do not increase any symptom
        # that is currently at or above the policy's adverse event threshold.
        high_symptoms = observation > self.max_allowable_symptom
        allowed = [
            i
            for i, treatment in enumerate(self.env.treatments.values())
            if np.all(treatment["treatment_effects"][high_symptoms] <= 0)
        ]

        # Select the allowed treatment with the highest Q-value, falling back on
        # considering all treatments if none are safe.
        return max(allowed or self.Q, key=self.Q.get)

    def update(self, treatment, reward):
        # Update the Q-value for the chosen treatment based on the reward received
        self.Q[treatment] = (1 - self.alpha) * self.Q[treatment] + self.alpha * reward

    def reset(self):
        self.Q = _q_star_values(self.env)


class ClinicalTrial(BasePolicy):
    """
    Policy that samples actions in proportion to their Q-values, mimicking Thompson Sampling.
    """

    def get_treatment_probs(self, observation):
        return self.treatment_probs

    def reset(self):
        """
        Set the fixed action probabilities of the policy based on the expected reward of
        each state.

        Since this is supposed to model a clinical trial, enforce sufficient exploration
        by normalizing the probabilities to sum to 1 and have a ratio of 8 between the
        most and least probable treatments.
        """
        Q_values = [q for q in _q_star_values(self.env).values()]
        Q_values = Q_values - np.max(Q_values)
        Q_values = (Q_values / min(Q_values)) * -np.log(8)
        exp_Q_values = np.exp(Q_values)
        self.treatment_probs = exp_Q_values / np.sum(exp_Q_values)


class Random(BasePolicy):
    """
    Policy that selects treatments uniformly at random to present a semi worst-case
    baseline.
    """

    def get_treatment_probs(self, observation):
        return np.ones(self.env.n_treatments) / self.env.n_treatments


class Oracle(BasePolicy):
    """
    A state-aware greedy policy that selects the treatment with the highest expected
    reward given knowledge of the true underlying state. This is intended to represent a
    near-optimal policy.
    """

    def get_treatment(self, observation):
        # Calculate the expected reward for each treatment in this state.
        expected_rewards = {
            action: self.env.expected_instantaneous_reward(
                self.env.current_disease, action
            )
            for action in range(self.env.n_treatments)
        }

        # Find the treatment with the highest expected reward
        return max(expected_rewards, key=expected_rewards.get)
