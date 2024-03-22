import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
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

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    alpha: float = 1.0  # Multiplier for alpha in loss
    batch_size: int = 256  # Batch size for all networks
    buffer_size: int = 2000000  # Replay buffer size
    checkpoints_path: Optional[str] = "./checkpoints"  # Save path
    dataset_path: Optional[str] = None  # Path to the dataset
    device: str = "cuda"
    env: str = "EpiCare-v0"  # OpenAI gym environment name
    env_seed: int = 1  # Environment seed
    episodes_avail: int = 65536  # Number of episodes
    eval_freq: int = 5000  # How often (time steps) we evaluate
    frame_stack: int = 8  # Number of frames to stack
    gamma: float = 1.0  # Discount factor
    load_model: str = ""  # Model load file name, "" doesn't load
    max_timesteps: int = 80000  # Max time steps to run environment
    n_episodes: int = 100  # How many episodes run during evaluation
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    num_checkpoints: int = 32  # Number of checkpoints to save
    orthogonal_init: bool = True  # Orthogonal initialization
    qf_lr: float = 3e-05  # Critics learning rate
    q_n_hidden_layers: int = 3  # Number of hidden layers in Q networks
    reward_bias: float = -1.0  # Reward bias for normalization
    reward_scale: float = 5.0  # Reward scale for normalization
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds

    # Update the parameters with the parameters of the sweep
    def update_params(self, params: Dict[str, Any]) -> "TrainConfig":
        for key, value in params.items():
            setattr(self, key, value)
        self.dataset_path = f"./data/hard/train_seed_{self.env_seed}.hdf5"
        self.name = f"{self.name}-{self.env}-{self.seed}-{self.env_seed}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)
        return self


def init_module_weights(module: torch.nn.Sequential, orthogonal_init: bool = False):
    # Specific orthgonal initialization for inner layers
    # If orthogonal init is off, we do not change default initialization
    if orthogonal_init:
        for submodule in module[:-1]:
            if isinstance(submodule, nn.Linear):
                nn.init.orthogonal_(submodule.weight, gain=np.sqrt(2))
                nn.init.constant_(submodule.bias, 0.0)

    # Lasy layers should be initialzied differently as well
    if orthogonal_init:
        nn.init.orthogonal_(module[-1].weight, gain=1e-2)
    else:
        nn.init.xavier_uniform_(module[-1].weight, gain=1e-2)

    nn.init.constant_(module[-1].bias, 0.0)


class FullyConnectedQFunction(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        orthogonal_init: bool = False,
        n_hidden_layers: int = 3,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.orthogonal_init = orthogonal_init

        layers = [
            nn.Linear(observation_dim, 256),
            nn.ReLU(),
        ]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(256, action_dim))

        self.network = nn.Sequential(*layers)

        init_module_weights(self.network, orthogonal_init)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class DiscreteCQL:
    def __init__(
        self,
        q1,
        q1_optimizer,
        q2,
        q2_optimizer,
        gamma,
        alpha: float = 5.0,
        device: str = "cpu",
    ):
        self.q1 = q1
        self.q1_optimizer = q1_optimizer
        self.q2 = q2
        self.q2_optimizer = q2_optimizer
        self.gamma = gamma
        self.alpha = alpha
        self.device = device

        self.total_it = 0

    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        state = torch.tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            q1 = self.q1(state)
            q2 = self.q2(state)
        return torch.min(q1, q2).argmax(dim=-1).cpu().numpy()

    def q_func(self, observations: torch.Tensor, actions: torch.Tensor):
        return torch.min(self.q1(observations, actions), self.q2(observations, actions))

    def _dqn_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
    ):
        q1_values = torch.sum(self.q1(observations) * actions, dim=-1)
        next_q1_values = self.q1(next_observations).argmax(dim=-1)
        pred_q1 = rewards + self.gamma * next_q1_values * (1 - terminals)
        q1_loss = F.huber_loss(q1_values, pred_q1)

        q2_values = torch.sum(self.q2(observations) * actions, dim=-1)
        next_q2_values = self.q2(next_observations).argmax(dim=-1)
        pred_q2 = rewards + self.gamma * next_q2_values * (1 - terminals)
        q2_loss = F.huber_loss(q2_values, pred_q2)

        loss = q1_loss + q2_loss

        log_dict = dict(
            q1_loss=q1_loss.item(),
            q2_loss=q2_loss.item(),
            q1_value=q1_values.mean().item(),
            q2_value=q2_values.mean().item(),
        )

        return loss, log_dict

    def _conservative_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        values = self.q_func(observations, actions)
        logsumexp = torch.logsumexp(values, dim=1, keepdim=True)
        data_values = (values * actions).sum(dim=1, keepdim=True)
        return (logsumexp - data_values).mean()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        self.total_it += 1

        dqn_loss, log_dict = self._dqn_loss(
            observations, actions, next_observations, rewards, dones
        )

        conservative_loss = self._conservative_loss(observations, actions)
        log_dict["conservative_loss"] = conservative_loss.item()

        loss = dqn_loss + self.alpha * conservative_loss

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return dict(
            total_it=self.total_it,
            q1=self.q1.state_dict(),
            q1_optimizer=self.q1_optimizer.state_dict(),
            q2=self.q2.state_dict(),
            q2_optimizer=self.q2_optimizer.state_dict(),
        )

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.total_it = state_dict["total_it"]
        self.q1.load_state_dict(state_dict["q1"])
        self.q1_optimizer.load_state_dict(state_dict["q1_optimizer"])
        self.q2.load_state_dict(state_dict["q2"])
        self.q2_optimizer.load_state_dict(state_dict["q2_optimizer"])


def load_custom_dataset(config: TrainConfig) -> Dict[str, np.ndarray]:
    # Load your custom dataset from an HDF5 file
    with h5py.File(config.dataset_path, "r") as dataset_file:
        # Find index by which episodes_avail terminals are reached
        terminals = dataset_file["terminals"][:]
        # cumsum terminals to find episode boundaries
        terminals_cumsum = np.cumsum(terminals)
        # find index of episodes_avail terminals
        cutoff = np.argwhere(terminals_cumsum == config.episodes_avail)[0][0] + 1

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


def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float]:
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


def modify_reward(
    dataset: Dict,
    env_name: str,
    max_episode_steps: int = 1000,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    dataset["rewards"] = dataset["rewards"] * reward_scale + reward_bias


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
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
        frame_stack: int = 1,
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._frame_stack = frame_stack

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

        self._states[:n_transitions] = frame_stacked_states.reshape(n_transitions, -1)
        self._next_states[:n_transitions] = frame_stacked_next_states.reshape(
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


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


@torch.no_grad()
def eval_actor(
    env: gym.Env,
    actor: nn.Module,
    device: str,
    n_episodes: int,
    seed: int,
    frame_stack: int,
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state_history = np.zeros((frame_stack, env.observation_space.shape[0]))
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            state_history = np.roll(state_history, shift=1, axis=0)
            state_history[0] = state
            action = actor.act(state_history, device=device)
            # Convert back from one-hot encoding
            action = np.argmax(action)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def train(config: TrainConfig):
    env = gym.make(config.env, seed=config.env_seed)

    state_dim = env.observation_space.shape[0] * config.frame_stack
    action_dim = env.action_space.n

    dataset = load_custom_dataset(config)

    if config.normalize_reward:
        modify_reward(
            dataset,
            config.env,
            reward_scale=config.reward_scale,
            reward_bias=config.reward_bias,
        )

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

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
    )
    replay_buffer.preprocess_dataset(dataset)

    if config.num_checkpoints != None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    q1 = FullyConnectedQFunction(
        state_dim,
        action_dim,
        config.orthogonal_init,
        config.q_n_hidden_layers,
    ).to(config.device)
    q2 = FullyConnectedQFunction(
        state_dim, action_dim, config.orthogonal_init, config.q_n_hidden_layers
    ).to(config.device)
    q1_optimizer = torch.optim.Adam(list(q1.parameters()), config.qf_lr)
    q2_optimizer = torch.optim.Adam(list(q2.parameters()), config.qf_lr)

    print("---------------------------------------")
    print(f"Training CQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = DiscreteCQL(
        q1,
        q1_optimizer,
        q2,
        q2_optimizer,
        gamma=config.gamma,
        alpha=config.alpha,
        device=config.device,
    )

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))

    wandb_init(asdict(config))

    # Generate a list of training steps as close as possible to evenly spaced
    # throughout the training process.
    checkpoint_num = 0
    checkpoint_steps = [
        int(round(x))
        for x in np.linspace(
            config.max_timesteps - 1, 0, config.num_checkpoints, endpoint=False
        )
    ]

    evaluations = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)

        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env,
                trainer,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
                frame_stack=config.frame_stack,
            )
            eval_score_mean = eval_scores.mean()
            eval_score_std = eval_scores.std()
            normalized_eval_score_mean = env.get_normalized_score(eval_scores) * 100.0
            normalized_eval_score_std = env.get_normalized_score(eval_score_std) * 100.0
            evaluations.append(normalized_eval_score_mean)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score_mean:.3f} , Normalized score: {normalized_eval_score_mean:.3f}"
            )
            print("---------------------------------------")

            wandb.log(
                {
                    "normalized_score_mean": normalized_eval_score_mean,
                    "normalized_score_std": normalized_eval_score_std,
                },
                step=trainer.total_it,
            )

        if config.num_checkpoints and t == checkpoint_steps[-1]:
            checkpoint_steps.pop()
            torch.save(
                trainer.state_dict(),
                os.path.join(
                    config.checkpoints_path, f"checkpoint_{checkpoint_num}.pt"
                ),
            )
            checkpoint_num += 1


if __name__ == "__main__":
    with open(
        "./sweep_configs/data_restriction_sweeps/cql_restriction_config.yaml", "r"
    ) as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)

    # Start a new wandb run
    run = wandb.init(config=sweep_config, group="CQL_EpiCare_sweep")

    # Update the TrainConfig instance with parameters from wandb
    # This assumes that update_params will handle single value parameters correctly
    config = TrainConfig()
    config.update_params(dict(wandb.config))

    # Now pass the updated config to the train function
    train(config)
