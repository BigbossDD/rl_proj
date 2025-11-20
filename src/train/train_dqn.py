import gymnasium as gym
import torch
import numpy as np
from collections import deque

from helper_classes.replay_buffer import ReplayBuffer
from models.dqn_agent import DQN_Agent
from helper_classes.wrappers import make_atari_env


def train_dqn(
    env_id="ALE/BattleZone-v5",
    num_episodes=5000,
    replay_size=100_000,
    batch_size=32,
    start_learning=10_000,
    train_freq=4,
    target_update_freq=1000,
    gamma=0.99,
    lr=1e-4,
    device="cuda"
):
    '''
    
    Train the agent will happen here where it will also use the replay buffer , and the wrappers for atari environment 
    depending on the hyperparameters provided like the number of episodes ,replay size ,batch size ,gamma ,learning rate...
    and then will update the target network 
    
    
    '''
    pass