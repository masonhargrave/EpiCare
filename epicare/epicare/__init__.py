from .envs import EpiCare
from gym.envs.registration import register

register(
    id="EpiCare-v0",
    entry_point="epicare:EpiCare",
)
