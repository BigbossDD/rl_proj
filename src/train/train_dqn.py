import gymnasium as gym
import torch
import numpy as np
from collections import deque

from helper_classes.replay_buffer import ReplayBuffer
from models.dqn_agent import DQN_Agent
from helper_classes.wrappers import make_atari_env


def train_dqn(
    
    
    #an example of what a will go to the agent as parameters (we will change them later )
    
    
    env_id="ALE/BattleZone-v5",
    num_episodes=None,
    replay_size=None,
    batch_size=None,
    start_learning=None,
    train_freq=None,
    target_update_freq=None,
    gamma=None,
    lr=None,
    device="cuda"
):
    '''
    
    Train the agent will happen here where it will also use the replay buffer , and the wrappers for atari environment 
    depending on the hyperparameters provided like the number of episodes ,replay size ,batch size ,gamma ,learning rate...
    and then will update the target network 
    
    
    '''
    pass
