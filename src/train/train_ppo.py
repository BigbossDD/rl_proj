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
    '''
    Train the PPO agent using GAE and the specified hyperparameters
    The training loop will interact with the environment, collect experiences, 
    compute advantages using GAE, and update the policy and value networks
    Args:
        env_id (str): Atari environment ID
        total_timesteps (int): Total number of timesteps to train
        steps_per_update (int): Number of steps to collect before each policy update
        save_path (str): Path to save the trained model
    
    '''

    pass