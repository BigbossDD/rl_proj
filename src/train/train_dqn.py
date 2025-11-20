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
    # -------------------------
    # Environment
    # -------------------------
    env = make_atari_env(env_id)
    num_actions = env.action_space.n
    input_shape = (4, 84, 84)

    # -------------------------
    # Agent + Replay Buffer
    # -------------------------
    agent = DQN_Agent(
        input_shape=input_shape,
        num_actions=num_actions,
        gamma=gamma,
        lr=lr,
        device=device
    )

    replay_buffer = ReplayBuffer(
        capacity=replay_size,
        state_shape=input_shape,
        device=device
    )

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 1_000_000  # linear schedule

    total_steps = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            # -------------------------
            # Epsilon schedule
            # -------------------------
            epsilon = max(epsilon_min, 1.0 - total_steps / epsilon_decay)

            # -------------------------
            # Action selection
            # -------------------------
            action = agent.select_action(state, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            total_steps += 1

            # -------------------------
            # Train
            # -------------------------
            if total_steps > start_learning and total_steps % train_freq == 0:
                batch = replay_buffer.sample(batch_size)
                loss = agent.train_step(batch)

            # -------------------------
            # Update Target Net
            # -------------------------
            if total_steps % target_update_freq == 0:
                agent.update_target()

            if done:
                print(f"Episode {episode} | Reward: {episode_reward} | Epsilon: {epsilon:.3f}")
                break

    env.close()
