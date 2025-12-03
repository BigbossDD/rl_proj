import torch
import numpy as np

from helper_classes.wrappers import make_atari_env
from models.ppo_agent import PPOAgent


def train_ppo(
    env_id="ALE/BattleZone-v5",
    total_timesteps=1_000_000,
    steps_per_update=2048,
    save_path="models/ppo_battlezone.pth"
):
    # training of PPO agent 
    
pass
