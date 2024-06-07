import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import h5py
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from epicare.envs import EpiCare  # noqa: F401
from tqdm import trange

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    # path to the dataset
    dataset_path: Optional[str] = None
    # wandb project name
    project: str = "AWAC-Benchmark"
    # wandb group name
    group: str = "AWAC-EpiCare"
    # wandb run name
    name: str = "AWAC"
    # training dataset and evaluation environment
    env: str = "EpiCare-v0"
    # actor and critic hidden dim
    hidden_dim: int = 256
    # actor and critic learning rate
    learning_rate: float = 3e-4
    # discount factor
    gamma: float = 1.0
    # coefficient for the talambrget critic Polyak's update
    tau: float = 5e-3
    # awac actor loss temperature, controlling balance
    # between behaviour cloning and Q-value maximization
    awac_lambda: float = 1.0
    # total number of gradient updated during training
    num_train_ops: int = 200_000
    # training batch size
    batch_size: int = 256
    # maximum size of the replay buffer
    buffer_size: int = 1_000_000
    # evaluation frequency, will evaluate every eval_frequency
    # training steps
    eval_frequency: int = 1000
    # number of episodes to run during evaluation
    n_test_episodes: int = 100
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = "./checkpoints"
    # configure PyTorch to use deterministic algorithms instead
    # of nondeterministic ones
    deterministic_torch: bool = False
    # training random seed
    seed: int = 1
    # evaluation random seed
    test_seed: int = 1
    # training device
    device: str = "cuda"
    # environment seed
    env_seed: int = 1
    # number of checkpoints to save
    num_checkpoints: int = 0
    # frame stacking memory
    frame_stack: int = 1
    # behavior policy
    behavior_policy: str = "smart"
    # include previous action in the observation
    include_previous_action: bool = False

    sweep_config: Optional[dict] = field(default=None)

    # Update the parameters with the parameters of the sweep
    def update_params(self, params: Dict[str, Any]) -> "TrainConfig":
        for key, value in params.items():
            setattr(self, key, value)
        self.dataset_path = (
            f"./data/{self.behavior_policy}/train_seed_{self.env_seed}.hdf5"
        )
        self.name = f"{self.name}-{self.env_name}-{self.seed}-{self.env_seed}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)
        return self


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
        frame_stack: int = 1,
        include_previous_action: bool = False,
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._frame_stack = frame_stack
        self._prev_action = include_previous_action
        self._state_dim = state_dim

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros(
            (buffer_size, 1), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def preprocess_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        # Check if actions are already one-hot encoded
        if len(data["actions"].shape) == 1:
            # One-hot encode the actions if they are not already
            print("One-hot encoding actions")
            print(f"Actions shape: {self._actions.shape}")
            actions = np.eye(self._actions.shape[1])[data["actions"].astype(int)]
            data["actions"] = actions
        else:
            # Actions are already in the correct shape
            print("Actions are already one-hot encoded")
            actions = data["actions"]
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

        # Frame stack the states
        frame_stacked_states = torch.zeros(
            (self._frame_stack,) + data["observations"].shape
        )
        frame_stacked_next_states = torch.zeros_like(frame_stacked_states)

        boundaries = [0] + [i + 1 for i, x in enumerate(self._dones.squeeze()) if x]
        observations = torch.tensor(data["observations"], dtype=torch.float32)
        next_observations = torch.tensor(data["next_observations"], dtype=torch.float32)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            for i in range(start, end):
                for j in range(i, min(i + self._frame_stack, end)):
                    frame_stacked_states[j - i, j] = observations[i, ...]
                    frame_stacked_next_states[j - i, j] = next_observations[i, ...]

        frame_stacked_states = frame_stacked_states.moveaxis(0, 1).to(self._device)
        frame_stacked_next_states = frame_stacked_next_states.moveaxis(0, 1).to(
            self._device
        )

        if self._prev_action:
            # Get the next action with zero for the terminal states
            next_actions = torch.where(
                self._dones[:n_transitions].bool(),
                torch.zeros_like(self._actions[:n_transitions]),
                self._actions[:n_transitions],
            )
            prev_actions = next_actions.roll(1, dims=0)
            up_to = -self._actions.shape[1]
            self._states[:n_transitions, up_to:] = prev_actions
            self._next_states[:n_transitions, up_to:] = next_actions
        else:
            up_to = self._states.shape[1]

        self._states[:n_transitions, :up_to] = frame_stacked_states.reshape(
            n_transitions, -1
        )
        self._next_states[:n_transitions, :up_to] = frame_stacked_next_states.reshape(
            n_transitions, -1
        )

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def get_action_probabilities(self, state: torch.Tensor) -> torch.Tensor:
        logits = self._mlp(state)
        return F.softmax(logits, dim=-1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self._mlp(state)
        log_prob = torch.log(F.softmax(logits, dim=-1) + 1e-10)
        action = F.one_hot(torch.argmax(logits, dim=-1), num_classes=logits.shape[-1])
        return action, log_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        state_flattened = state.flatten()[None, :]
        state_t = torch.tensor(state_flattened, dtype=torch.float32, device=device)
        logits = self._mlp(state_t)
        action_t = torch.argmax(logits, dim=-1)
        if not self._mlp.training:
            action_t = F.one_hot(action_t, num_classes=logits.shape[-1])
        action = action_t[0].cpu().numpy()
        return action

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        _, logits = self.forward(state)
        # Compute the log probabilities using the logits and the taken actions
        log_probs = -F.cross_entropy(logits, action.argmax(dim=-1), reduction="none")
        return log_probs


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_value = self._mlp(torch.cat([state, action], dim=-1))
        return q_value


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class AdvantageWeightedActorCritic:
    def __init__(
        self,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 5e-3,  # parameter for the soft target update,
        awac_lambda: float = 1.0,
        exp_adv_max: float = 100.0,
    ):
        self._actor = actor
        self._actor_optimizer = actor_optimizer

        self._critic_1 = critic_1
        self._critic_1_optimizer = critic_1_optimizer
        self._target_critic_1 = deepcopy(critic_1)

        self._critic_2 = critic_2
        self._critic_2_optimizer = critic_2_optimizer
        self._target_critic_2 = deepcopy(critic_2)

        self._gamma = gamma
        self._tau = tau
        self._awac_lambda = awac_lambda
        self._exp_adv_max = exp_adv_max

    def _actor_loss(self, states, actions):
        with torch.no_grad():
            pi_action, _ = self._actor(states)
            v = torch.min(
                self._critic_1(states, pi_action), self._critic_2(states, pi_action)
            )

            q = torch.min(
                self._critic_1(states, actions), self._critic_2(states, actions)
            )
            adv = q - v
            weights = torch.clamp_max(
                torch.exp(adv / self._awac_lambda), self._exp_adv_max
            )

        action_log_prob = self._actor.log_prob(states, actions)
        loss = (-action_log_prob * weights).mean()
        return loss

    def _critic_loss(self, states, actions, rewards, dones, next_states):
        with torch.no_grad():
            next_actions, _ = self._actor(next_states)

            q_next = torch.min(
                self._target_critic_1(next_states, next_actions),
                self._target_critic_2(next_states, next_actions),
            )
            q_target = rewards + self._gamma * (1.0 - dones) * q_next

        q1 = self._critic_1(states, actions)
        q2 = self._critic_2(states, actions)

        q1_loss = nn.functional.mse_loss(q1, q_target)
        q2_loss = nn.functional.mse_loss(q2, q_target)
        loss = q1_loss + q2_loss
        return loss

    def _update_critic(self, states, actions, rewards, dones, next_states):
        loss = self._critic_loss(states, actions, rewards, dones, next_states)
        self._critic_1_optimizer.zero_grad()
        self._critic_2_optimizer.zero_grad()
        loss.backward()
        self._critic_1_optimizer.step()
        self._critic_2_optimizer.step()
        return loss.item()

    def _update_actor(self, states, actions):
        loss = self._actor_loss(states, actions)
        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()
        return loss.item()

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        critic_loss = self._update_critic(states, actions, rewards, dones, next_states)
        actor_loss = self._update_actor(states, actions)

        soft_update(self._target_critic_1, self._critic_1, self._tau)
        soft_update(self._target_critic_2, self._critic_2, self._tau)

        result = {"critic_loss": critic_loss, "actor_loss": actor_loss}
        return result

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self._actor.state_dict(),
            "critic_1": self._critic_1.state_dict(),
            "critic_2": self._critic_2.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._actor.load_state_dict(state_dict["actor"])
        self._critic_1.load_state_dict(state_dict["critic_1"])
        self._critic_2.load_state_dict(state_dict["critic_2"])


def load_custom_dataset(config: TrainConfig) -> Dict[str, np.ndarray]:
    # Load your custom dataset from an HDF5 file
    with h5py.File(config.dataset_path, "r") as dataset_file:
        # Here, 'observations', 'actions', etc. are keys in your HDF5 file that correspond to your data.
        # If they are named differently in your file, you'll need to adjust the keys accordingly.
        observations = dataset_file["observations"][:]
        actions = dataset_file["actions"][:]
        rewards = dataset_file["rewards"][:]
        next_observations = dataset_file["next_observations"][:]
        terminals = dataset_file["terminals"][:]

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


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    env = gym.wrappers.TransformObservation(env, normalize_state)
    return env


@torch.no_grad()
def eval_actor(
    env: gym.Env,
    actor: nn.Module,
    device: str,
    n_episodes: int,
    seed: int,
    frame_stack: int,
    include_previous_action: bool = False,
    action_dim: int = 0,
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state_history = np.zeros((frame_stack, env.observation_space.shape[0]))
        prev_action = np.zeros((action_dim,))
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            state_history = np.roll(state_history, shift=1, axis=0)
            state_history[0] = state

            # Prepare the actor input depending on whether previous action is included
            if include_previous_action:
                state = np.concatenate((state_history.flatten(), prev_action))
            else:
                state = state_history.flatten()

            action = actor.act(state, device=device)
            # Convert back from one-hot encoding
            action_idx = np.argmax(action)
            state, reward, done, _ = env.step(action_idx)
            episode_reward += reward

            # Update prev_action for the next iteration
            if include_previous_action:
                prev_action = np.arange(action_dim) == action_idx

        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


def train(config: TrainConfig):
    env = gym.make(config.env_name, seed=config.env_seed)
    set_seed(config.seed, env, deterministic_torch=config.deterministic_torch)
    state_dim = env.observation_space.shape[0] * config.frame_stack
    if config.include_previous_action:
        state_dim += env.action_space.n
    action_dim = env.action_space.n
    dataset = load_custom_dataset(config)

    state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
        frame_stack=config.frame_stack,
        include_previous_action=config.include_previous_action,
    )
    replay_buffer.preprocess_dataset(dataset)

    actor_critic_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": config.hidden_dim,
    }

    actor = Actor(**actor_critic_kwargs)
    actor.to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_1 = Critic(**actor_critic_kwargs)
    critic_2 = Critic(**actor_critic_kwargs)
    critic_1.to(config.device)
    critic_2.to(config.device)
    critic_1_optimizer = torch.optim.Adam(
        critic_1.parameters(), lr=config.learning_rate
    )
    critic_2_optimizer = torch.optim.Adam(
        critic_2.parameters(), lr=config.learning_rate
    )

    awac = AdvantageWeightedActorCritic(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic_1=critic_1,
        critic_1_optimizer=critic_1_optimizer,
        critic_2=critic_2,
        critic_2_optimizer=critic_2_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        awac_lambda=config.awac_lambda,
    )
    wandb_init(asdict(config))

    if config.num_checkpoints > 0:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Generate a list of training steps as close as possible to evenly spaced
    # throughout the training process.
    checkpoint_num = 0
    checkpoint_steps = [
        int(round(x))
        for x in np.linspace(
            config.num_train_ops - 1, 0, config.num_checkpoints, endpoint=False
        )
    ]

    for t in trange(config.num_train_ops, ncols=80):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        update_result = awac.update(batch)
        wandb.log(update_result, step=t)
        if (t + 1) % config.eval_frequency == 0:
            eval_scores = eval_actor(
                env,
                actor,
                config.device,
                config.n_test_episodes,
                config.test_seed,
                frame_stack=config.frame_stack,
                include_previous_action=config.include_previous_action,
                action_dim=action_dim,
            )

            normalized_scores_mean = env.get_normalized_score(eval_scores) * 100.0
            normalized_scores_std = (
                env.get_normalized_score(np.std(eval_scores)) * 100.0
            )
            wandb.log(
                {
                    "normalized_scores_mean": normalized_scores_mean,
                    "normalized_scores_std": normalized_scores_std,
                },
                step=t,
            )

        if config.num_checkpoints and t == checkpoint_steps[-1]:
            checkpoint_steps.pop()
            torch.save(
                awac.state_dict(),
                os.path.join(
                    config.checkpoints_path, f"checkpoint_{checkpoint_num}.pt"
                ),
            )
            checkpoint_num += 1

    wandb.finish()


if __name__ == "__main__":
    with open("./sweep_configs/all_data_sweeps/awac_final_config.yaml", "r") as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)

    # Start a new wandb run
    run = wandb.init(config=sweep_config, group="AWAC-EpiCare_final")

    # Update the TrainConfig instance with parameters from wandb
    # This assumes that update_params will handle single value parameters correctly
    config = TrainConfig()
    config.update_params(dict(wandb.config))

    # Now pass the updated config to the train function
    train(config)
