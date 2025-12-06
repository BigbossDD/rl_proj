import torch
import numpy as np
from collections import deque
import time

from helper_classes.wrappers import make_atari_env
from models.rainbowDQN_agent import RainbowDQNAgent


def train_rainbow_dqn(
    env_id="ALE/BattleZone-v5",
    total_episodes=2000,
    max_steps=10000,
    batch_size=32,
    save_path="models/rainbow_dqn_battlezone.pth"
):
#raining of rainbow DQN agent 
    '''
    Train the Rainbow DQN agent using the specified hyperparameters
    Args:
        env_id (str): Atari environment ID
        total_episodes (int): Total number of episodes to train
        max_steps (int): Maximum steps per episode
        batch_size (int): Batch size for training
        save_path (str): Path to save the trained model
  
    '''
    pass
   

