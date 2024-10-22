import functools

import gym
import numpy as np
import scipy.linalg as la
from gym.envs.registration import register
from scipy import stats
from scipy.sparse.csgraph import connected_components
from scipy.special import expit, logit


def generate_orthogonal_matrix(n, rng):
    "Generate a random orthogonal matrix of size n×n."
    A = rng.normal(0, 1, (n, n))
    Q, R = np.linalg.qr(A)
    return Q


def get_communicating_classes(matrix):
    "Find the communicating classes in the Markov chain."
    # Use scipy's connected_components to find the strongly connected components
    n_components, labels = connected_components(
        csgraph=matrix, directed=True, return_labels=True
    )

    # Create a list of lists to hold states for each communicating class
    classes = [[] for _ in range(n_components)]
    for index, label in enumerate(labels):
        classes[label].append(index)

    return classes


def get_stationary_distribution_for_class(matrix, communicating_class):
    "Find the stationary distribution for a communicating class."
    # Extract the submatrix corresponding to the communicating class
    submatrix = matrix[communicating_class, :][:, communicating_class]

    # Number of states in the class
    n = len(submatrix)

    # Create an augmented matrix to account for the normalization condition
    # We stack an additional row for the constraint that probabilities sum to 1
    A = np.vstack((submatrix - np.eye(n), np.ones(n)))
    b = np.zeros(n + 1)
    b[-1] = 1  # The sum of the probabilities should be 1

    # Solve the system of linear equations
    pi, residuals, rank, s = la.lstsq(
        A.T, b
    )  # Transpose A to solve the left eigenvector problem

    # Return the stationary distribution for the class, with zero-padding for the other states
    full_pi = np.zeros(len(matrix))
    for i, state in enumerate(communicating_class):
        full_pi[state] = pi[i]

    return full_pi


def combine_distributions(distributions, weights):
    "Combine distributions weighted by the size of each class."
    combined_distribution = np.zeros_like(distributions[0])
    for dist, weight in zip(distributions, weights):
        combined_distribution += dist * weight
    return combined_distribution / combined_distribution.sum()


def analyze_markov_chain(matrix):
    "Analyze the Markov chain to find the stationary distribution."
    # Find the communicating classes
    classes = get_communicating_classes(matrix)

    # Handle the reducible case
    if len(classes) > 1:
        # Calculate the stationary distribution for each class
        distributions = [
            get_stationary_distribution_for_class(matrix, c) for c in classes
        ]
        # Assuming a uniform initial distribution, the weight is the relative size of the class
        weights = [len(c) / len(matrix) for c in classes]
        # Combine the distributions
        return combine_distributions(distributions, weights)
    # Handle the irreducible case
    else:
        return get_stationary_distribution_for_class(matrix, classes[0])


def calculate_stationary_distribution(
    matrix, communicating_classes, even_class_distribution=False
):
    """
    Calculates the stationary distribution for the entire Markov chain
    given the communicating classes and an initial distribution that is either uniform over states
    or uniform over communicating classes based on the even_class_distribution flag.
    """
    n = matrix.shape[0]  # Total number of states in the Markov chain
    full_stationary_distribution = np.zeros(n)

    # If even_class_distribution is True, each class has the same total initial probability
    class_probability = (
        1 / len(communicating_classes) if even_class_distribution else None
    )

    for states in communicating_classes:
        submatrix = matrix[states][:, states]
        num_states_in_class = len(submatrix)
        A = np.vstack(
            (submatrix.T - np.eye(num_states_in_class), np.ones(num_states_in_class))
        )
        b = np.zeros(num_states_in_class + 1)
        b[-1] = 1  # The sum of probabilities should be 1

        # Solve the system
        pi, _, _, _ = la.lstsq(A, b)

        # Weight each element of pi according to the distribution scheme
        if even_class_distribution:
            # Each state in the class gets an equal share of the class's total probability
            weight = class_probability
        else:
            # Uniform distribution over states
            weight = num_states_in_class / n

        for i, state in enumerate(states):
            full_stationary_distribution[state] = pi[i] * weight

    # Normalize the stationary distribution to ensure it sums to 1
    full_stationary_distribution /= full_stationary_distribution.sum()

    return full_stationary_distribution


class EpiCare(gym.Env):
    """Environment to model disease treatment using RL."""

    def __init__(
        self,
        n_diseases=16,
        n_treatments=16,
        n_symptoms=8,
        disease_cost_range=(1, 10),
        symptom_modulation_range=(-2.0, 1.0),
        symptom_std_range=(1.0, 2.0),
        symptom_mean_range=(0.0, 2.0),
        remission_reward=64,
        remission_prob_range=(0.8, 1.0),
        adverse_event_reward=-64,
        adverse_event_threshold=0.999,
        baseline_symptom_range=(0.0, 0.1),
        seed=1,
        max_visits=8,
        use_gymnasium=False,
        use_symptom_rewards=True,
        use_disease_rewards=False,
        treatment_affect_observation=True,
        even_class_distribution=False,
        patient_modifiers=True,
    ):
        super(EpiCare, self).__init__()

        # Create a non-defaut numpy rng
        rng = np.random.RandomState(seed)

        # Configurable parameters
        self.disease_cost_range = disease_cost_range
        self.symptom_modulation_range = symptom_modulation_range
        self.symptom_std_range = symptom_std_range
        self.symptom_mean_range = symptom_mean_range
        self.baseline_symptom_range = baseline_symptom_range
        self.remission_reward = remission_reward
        self.remission_prob_range = remission_prob_range
        self.adverse_event_reward = adverse_event_reward
        self.adverse_event_threshold = adverse_event_threshold
        self.max_visits = max_visits
        self.use_symptom_rewards = use_symptom_rewards
        self.use_disease_rewards = use_disease_rewards
        self.treatment_affect_observation = treatment_affect_observation
        self.even_class_distribution = even_class_distribution
        self.treatment_cost_range = (1, max(1, (remission_reward / (2 * max_visits))))
        self.num_diseases_for_treatment_range = (1, max(2, n_diseases // 8))
        self.patient_modifiers = patient_modifiers

        # Step tracking
        self.visit_number = 0

        # Action and observation spaces
        if use_gymnasium:
            import gymnasium

            self.action_space = gymnasium.spaces.Discrete(n_treatments)
            self.observation_space = gymnasium.spaces.Box(
                low=0.0, high=1.0, shape=(n_symptoms,), dtype=np.float32
            )
        else:
            self.action_space = gym.spaces.Discrete(n_treatments)
            self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(n_symptoms,), dtype=np.float32
            )

        self.n_diseases = n_diseases
        self.n_treatments = n_treatments
        self.n_symptoms = n_symptoms
        self.connection_probability = 1 / n_diseases

        # Since symptom rewards are optional, set the reward multiplier to zero if it
        # won't be used anyway.
        self.symptom_reward_multiplier = (
            self.remission_reward / (2 * self.max_visits * self.n_symptoms)
            if self.use_symptom_rewards
            else 0
        )

        self.generate_diseases(rng)
        self.disease_list = list(self.diseases.keys())
        self.treatments = self.generate_treatments(rng)
        self.generate_transition_matrix(rng)
        self.compute_stationary_distribution(
            even_class_distribution=self.even_class_distribution
        )
        self.reset()

    def generate_transition_matrix(self, rng):
        # Initialize a matrix filled with zeros for off-diagonal elements, and ones on the diagonal
        self.transition_matrix = np.eye(self.n_diseases)

        for i in range(self.n_diseases):
            for j in range(self.n_diseases):
                if i != j:
                    if rng.rand() < self.connection_probability:  # Check for connection
                        self.transition_matrix[i][j] = rng.uniform(0.01, 0.2)
                        self.transition_matrix[j][i] = rng.uniform(0.01, 0.2)

        # Normalize rows to sum to 1, representing probability distributions
        row_sums = (
            self.transition_matrix.sum(axis=1) - 1
        )  # Exclude self-transition in normalization
        self.transition_matrix[
            np.arange(self.n_diseases), np.arange(self.n_diseases)
        ] = (1 - row_sums)

    def compute_stationary_distribution(self, even_class_distribution=False):
        # Compute the stationary distribution for the Markov chain
        self.communicating_classes = get_communicating_classes(self.transition_matrix)
        self.stationary_distribution = calculate_stationary_distribution(
            self.transition_matrix, self.communicating_classes, even_class_distribution
        )

    def generate_diseases(self, rng):
        self.diseases = {}
        all_treatments = set(range(self.n_treatments))  # Set of all treatments

        for i in range(self.n_diseases):
            means = rng.uniform(
                self.symptom_mean_range[0],
                self.symptom_mean_range[1],
                size=self.n_symptoms,
            )
            std_devs = rng.uniform(
                self.symptom_std_range[0],
                self.symptom_std_range[1],
                size=self.n_symptoms,
            )
            std_devs_sorted = np.sort(std_devs)[::-1]
            P = generate_orthogonal_matrix(self.n_symptoms, rng)
            Sigma = P @ np.diag(std_devs_sorted**2) @ P.T  # Covariance matrix

            # Check if Sigma is symmetric
            assert np.allclose(Sigma, Sigma.T)
            # Check if Sigma is positive semi-definite
            assert np.all(np.linalg.eigvals(Sigma) >= 0)

            base_cost = (
                rng.uniform(*self.disease_cost_range) if self.use_disease_rewards else 0
            )

            self.diseases[f"Disease_{i}"] = {
                "symptoms": self.n_symptoms,
                "treatments": [],
                "symptom_means": means,
                "symptom_covariances": Sigma,
                "remission_probs": dict(),
                "base_cost": base_cost,
            }

        # Sprinkle treatments among the diseases
        for treatment in all_treatments:
            num_diseases_for_treatment = rng.randint(
                *self.num_diseases_for_treatment_range
            )
            diseases_for_treatment = rng.choice(
                self.n_diseases, size=num_diseases_for_treatment, replace=False
            )
            for disease in diseases_for_treatment:
                disease = f"Disease_{rng.randint(0, self.n_diseases)}"
                self.diseases[disease]["treatments"] = np.append(
                    self.diseases[disease]["treatments"], treatment
                )
                self.diseases[disease]["remission_probs"][treatment] = rng.uniform(
                    *self.remission_prob_range
                )

        # Ensure that each disease has at least one treatment
        for disease in self.diseases.values():
            if len(disease["treatments"]) == 0:
                disease["treatments"] = np.array([rng.randint(0, self.n_treatments)])
                disease["remission_probs"][disease["treatments"][0]] = rng.uniform(
                    *self.remission_prob_range
                )

    def handle_remission(self, action, reward):
        # Update remission-related states and compute reward
        self.current_disease = "Remission"
        self.reward_components["remission"] += self.remission_reward
        self.time_to_remission = (
            self.visit_number
            if self.time_to_remission is None
            else self.time_to_remission
        )
        reward += self.remission_reward
        self.current_symptoms = self.sample_symptoms()
        return (
            self.current_symptoms,
            reward,
            True,
            {
                "treatment": action,
                "disease_pre_treatment": self.current_disease,
                "remission": True,
                "adverse_event": False,
            },
        )

    def apply_treatment_and_transition_disease(self, treatment):
        treatment_modifiers = treatment["transition_modifiers"]
        modified_transitions = (
            self.transition_matrix[self.current_disease_index]
            * treatment_modifiers
            * self.patient_transition_modifiers
        )
        modified_transitions /= modified_transitions.sum()
        new_disease_index = np.random.choice(self.n_diseases, p=modified_transitions)
        self.current_disease = f"Disease_{new_disease_index}"
        self.current_disease_index = new_disease_index

        # Deduct the base reward for the current disease from the total reward
        disease_cost = self.diseases[self.current_disease]["base_cost"]
        self.reward_components["state_based"] -= disease_cost

        # Fluctuate symptoms based on disease distributions and then adjust them based on treatment effects
        self.current_symptoms = self.sample_symptoms(
            mod=treatment["treatment_effects"] + self.patient_symptom_modifiers
        )

        symptom_cost = self.symptom_reward_multiplier * self.current_symptoms.sum()
        self.reward_components["symptom_based"] -= symptom_cost
        return -disease_cost - symptom_cost

    def step(self, action):
        self.visit_number += 1
        reward = 0  # Reset step reward

        # Handle treatment cost
        treatment = self.treatments[f"Treatment_{action}"]
        treatment_cost = treatment["base_cost"]
        self.reward_components["treatment_based"] -= treatment_cost
        reward -= treatment_cost

        # Handle potential remission
        if (
            np.random.rand()
            < self.diseases[self.current_disease]["remission_probs"].get(action, 0)
            * self.patient_remission_modifiers[action]
        ):
            return self.handle_remission(action, reward)

        # Handle disease transition and symptom changes
        reward += self.apply_treatment_and_transition_disease(treatment)

        # Handle the other two types of termination: adverse events when any symptom is
        # too severe, and reaching the maximum number of visits.
        adverse_event = (
            self.current_symptoms.max()
            > self.adverse_event_threshold * self.patient_ae_modifier
        )
        if adverse_event:
            self.reward_components["adverse_event"] += self.adverse_event_reward
            reward += self.adverse_event_reward
            terminated = True
        else:
            terminated = self.visit_number == self.max_visits

        # Reduce the precision of symptom measurements and return as observations.
        return (
            np.round(self.current_symptoms, 1),
            reward,
            terminated,
            {
                "treatment": action,
                "disease_pre_treatment": self.current_disease,
                "remission": False,
                "adverse_event": adverse_event,
            },
        )

    def sample_symptoms(self, mod=0.0):
        if self.current_disease == "Remission":
            return np.random.uniform(
                self.baseline_symptom_range[0],
                self.baseline_symptom_range[1],
                self.n_symptoms,
            )
        else:
            symptom_means = self.diseases[self.current_disease]["symptom_means"]
            symptom_covariances = self.diseases[self.current_disease][
                "symptom_covariances"
            ]

            # Sample values from a multivariate normal distribution
            symptom_values = np.random.multivariate_normal(
                symptom_means, symptom_covariances
            )
            # Add modulation and ensure values are within valid range
            symptom_values = expit(symptom_values + mod)

            return symptom_values

    def generate_treatments(self, rng):
        treatments = {}
        for i in range(self.n_treatments):
            # Each treatment can have a cost between the configurable range
            base_cost = rng.uniform(*self.treatment_cost_range)

            affected_symptoms_count = rng.randint(1, max(2, self.n_symptoms))
            affected_symptoms = rng.choice(
                self.n_symptoms, size=affected_symptoms_count, replace=False
            )

            # Change in symptom severity due to treatment is generated within the configurable range
            symptom_changes = {
                symptom: rng.uniform(*self.symptom_modulation_range)
                for symptom in affected_symptoms
            }

            # Convert to vector form for easier processing
            symptom_changes = np.array(
                [symptom_changes.get(i, 0.0) for i in range(self.n_symptoms)]
            )

            # Generate treatment-specific transition modifiers for each disease transition
            transition_modifiers = rng.uniform(0.5, 1.5, size=self.n_diseases)

            treatments[f"Treatment_{i}"] = dict(
                base_cost=base_cost,
                treatment_effects=symptom_changes,
                transition_modifiers=transition_modifiers,
            )
        return treatments

    def reset(self, *, seed=None, options=None):
        self.current_disease = np.random.choice(
            self.disease_list, p=self.stationary_distribution
        )
        if self.patient_modifiers:
            self.patient_transition_modifiers = np.random.uniform(
                0.25, 1.75, size=self.n_diseases
            )
            self.patient_remission_modifiers = np.random.uniform(
                0.25, 1.75, size=self.n_treatments
            )
            self.patient_ae_modifier = np.random.uniform(
                self.adverse_event_threshold, 1 / self.adverse_event_threshold
            )
            self.patient_symptom_modifiers = np.random.uniform(
                self.symptom_modulation_range[0],
                self.symptom_modulation_range[1],
                size=self.n_symptoms,
            )

        else:
            self.patient_transition_modifiers = np.ones(self.n_diseases)
            self.patient_remission_modifiers = np.ones(self.n_treatments)
            self.patient_ae_modifier = 1
            self.patient_symptom_modifiers = np.zeros(self.n_symptoms)
        self.current_disease_index = self.disease_list.index(self.current_disease)
        self.current_symptoms = (
            self.sample_symptoms()
        )  # Initialize symptoms based on the current disease
        self.visit_number = 0

        # For tracking reward components and time to remission
        self.reward_components = {
            "remission": 0,
            "state_based": 0,
            "treatment_based": 0,
            "symptom_based": 0,
            "adverse_event": 0,
        }
        self.time_to_remission = -1  # -1 indicates no remission achieved

        return self.current_symptoms

    def render(self, mode="human"):
        # Simple visualization for now
        print(f"Current Disease: {self.current_disease}")
        print(f"Current Symptoms: {self.current_symptoms}")
        print("\n")

    def get_metrics(self):
        return self.reward_components, self.time_to_remission

    def get_normalized_score(self, returns):
        return np.mean(returns) / self.remission_reward

    def get_normalized_score_stds(self, returns):
        return np.std(returns) / self.remission_reward

    def _adverse_event_probability(self, disease, treatment):
        # The probability of an adverse event is the probability that the tanh'ed sum of
        # the symptoms exceeds the threshold. Use the inverse tanh to find the bounds in
        # the multivariate normal space, and use the CDF to find the probability that
        # any symptom exceeds the threshold.
        effect = self.treatments[f"Treatment_{treatment}"]["treatment_effects"]
        threshold = logit(self.adverse_event_threshold) - effect
        mu = self.diseases[disease]["symptom_means"]
        cov = self.diseases[disease]["symptom_covariances"]
        p_ok = stats.multivariate_normal(mu, cov).cdf(threshold)
        return 1 - p_ok

    def _adverse_event_probability_empirical(self, disease, treatment, N=1_000_000):
        # Same as _adverse_event_probability, but uses an empirical estimate to
        # double-check consistency.
        effect = self.treatments[f"Treatment_{treatment}"]["treatment_effects"]
        threshold = logit(self.adverse_event_threshold) - effect
        mu = self.diseases[disease]["symptom_means"]
        cov = self.diseases[disease]["symptom_covariances"]
        return np.any(stats.multivariate_normal(mu, cov).rvs(N) > threshold, -1).mean()

    def _expected_symptom_cost(self, disease):
        # The expected symptom cost is the expected value of the sum of the symptoms,
        # times a conversion factor that scales the symptom cost to the reward scale.
        mean_symptom_sum = np.sum(self.diseases[disease]["symptom_means"])
        return self.symptom_reward_multiplier * mean_symptom_sum

    @functools.lru_cache(maxsize=None)
    def expected_instantaneous_reward(self, disease, treatment):
        """
        Calculate the expected reward for an action in a given state. This is cached
        because it's reasonably expensive and the baseline policies need to call it many
        times in their setup.
        """

        # The remission reward has already been granted and you can't perform actions
        # anymore because the state is terminal.
        if disease == "Remission":
            return 0.0

        # Remission probability isn't defined for all states, so default to 0.
        remission_prob = self.diseases[disease]["remission_probs"].get(treatment, 0)
        adverse_prob = self._adverse_event_probability(disease, treatment)
        return (
            remission_prob * self.remission_reward
            + adverse_prob * self.adverse_event_reward
            - self.treatments[f"Treatment_{treatment}"]["base_cost"]
            - self._expected_symptom_cost(disease)
            - self.diseases[disease]["base_cost"]
        )


register(
    id="EpiCare-v0",  # Use the same ID when calling gym.make()
    entry_point="epicare.envs:EpiCare",  # Change this to the correct import path
)
