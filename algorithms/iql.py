# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import h5py
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR

from epicare.envs import EpiCare

TensorBatch = List[torch.Tensor]


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class TrainConfig:
    # number of episodes
    episodes_avail: int = 65536 * 2
    # path to the dataset
    dataset_path: Optional[str] = "./data/train_seed_1.hdf5"
    # wandb project name
    project: str = "IQL-Benchmark"
    # wandb group name
    group: str = "IQL-EpiCare"
    # wandb run name
    name: str = "IQL"
    # training dataset and evaluation environment
    env: str = "EpiCare-v0"
    # discount factor
    discount: float = 1.0
    # coefficient for the target critic Polyak's update
    tau: float = 0.005
    # actor update inverse temperature, similar to AWAC
    # small beta -> BC, big beta -> maximizing Q-value
    beta: float = 3.0
    # coefficient for asymmetric critic loss
    iql_tau: float = 0.9
    # whether to use deterministic actor
    iql_deterministic: bool = False
    # total gradient updates during training
    max_timesteps: int = int(5e5)
    # maximum size of the replay buffer
    buffer_size: int = 2_000_000
    # training batch size
    batch_size: int = 256
    # whether to normalize states
    normalize: bool = True
    # whether to normalize reward (like in IQL)
    normalize_reward: bool = False
    # V-critic function learning rate
    vf_lr: float = 3e-4
    # Q-critic learning rate
    qf_lr: float = 3e-4
    # actor learning rate
    actor_lr: float = 3e-4
    #  where to use dropout for policy network, optional
    actor_dropout: Optional[float] = 0.0
    # evaluation frequency, will evaluate every eval_freq training steps
    eval_freq: int = int(5e3)
    # number of episodes to run during evaluation
    n_episodes: int = 100
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = "./checkpoints"
    # file name for loading a model, optional
    load_model: str = ""
    # training random seed
    seed: int = 0
    # training device
    device: str = "cuda"
    # environment seed
    env_seed: int = 1
    # Gumbel softmax temperature
    temperature: float = 0.5
    # number of checkpoints
    num_checkpoints: Optional[int] = 1
    # frame stacking memory
    frame_stack: int = 8

    sweep_config: Optional[dict] = field(default=None)

    # Update the parameters with the parameters of the sweep
    def update_params(self, params: Dict[str, Any]) -> "TrainConfig":
        for key, value in params.items():
            setattr(self, key, value)
        self.dataset_path = f"./data/hard/train_seed_{self.env_seed}.hdf5"
        self.name = f"{self.name}-{self.env}-{self.seed}-{self.env_seed}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)
        return self


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


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
    # Verify that the seed has been set
    print(f"Seed: {seed}")
    print(f"Random state: {np.random.get_state()}")
    print(f"Torch state: {torch.random.get_rng_state()}")
    print(f"Torch cuda state: {torch.cuda.random.get_rng_state()}")
    print(f"Python hash seed: {os.environ['PYTHONHASHSEED']}")


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


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


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


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StochasticPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            dropout=dropout,
        )
        self.temperature = temperature

        print("Stochastic policy")

    def get_action_probabilities(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.net(obs)
        return F.softmax(logits + self.temperature * np.euler_gamma, dim=-1)

    def forward(self, obs: torch.Tensor) -> Normal:
        # Gumbel softmax
        logits = self.net(obs)
        return F.gumbel_softmax(logits, tau=self.temperature, hard=True)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        action = self(state)[0].cpu().numpy()
        return action


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )

        print("Deterministic policy")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return state.cpu().data.numpy().flatten()


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class ImplicitQLearning:
    def __init__(
        self,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy(adv, observations, actions, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]


def train(config: TrainConfig):
    env = gym.make(config.env, seed=config.env_seed)

    state_dim = env.observation_space.shape[0] * config.frame_stack
    action_dim = env.action_space.n

    dataset = load_custom_dataset(config)

    if config.normalize_reward:
        modify_reward(dataset, config.env)

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

    if config.num_checkpoints is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    print("Initializing actor")
    actor = (
        DeterministicPolicy(state_dim, action_dim, dropout=config.actor_dropout)
        if config.iql_deterministic
        else StochasticPolicy(
            state_dim,
            action_dim,
            dropout=config.actor_dropout,
            temperature=config.temperature,
        )
    ).to(config.device)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    kwargs = {
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
    }

    print("---------------------------------------")
    print(f"Training IQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ImplicitQLearning(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    # Generate a list of training steps as close as possible to evenly spaced
    # throughout the training process.
    checkpoint_num = 0
    checkpoint_steps = [
        int(round(x))
        for x in np.linspace(
            config.max_timesteps - 1, 0, config.num_checkpoints or 0, endpoint=False
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
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
                frame_stack=config.frame_stack,
            )
            eval_score = eval_scores.mean()
            eval_score_std = eval_scores.std()
            normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            normalized_eval_score_std = env.get_normalized_score(eval_score_std) * 100.0
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , normalized score: {normalized_eval_score:.3f}"
            )
            print("---------------------------------------")
            wandb.log(
                {
                    "normalized_score": normalized_eval_score,
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
        "./sweep_configs/data_restriction_sweeps/iql_restriction_config.yaml", "r"
    ) as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)

    # Start a new wandb run
    run = wandb.init(config=sweep_config, group="IQL-EpiCare-restrict")

    # Update the TrainConfig instance with parameters from wandb
    # This assumes that update_params will handle single value parameters correctly
    config = TrainConfig()
    config.update_params(dict(wandb.config))

    # Now pass the updated config to the train function
    train(config)
