'''
import gymnasium as gym
import numpy as np
import torch

from helper_classes.replay_buffer import ReplayBuffer
from models.dqn_agent import DQN_Agent
from helper_classes.wrappers import make_atari_env


def train_dqn(
    env_id="ALE/BattleZone-v5",
    num_episodes=500,
    replay_size=100_000,
    batch_size=64,
    start_learning=10_000,
    train_freq=4,
    target_update_freq=1000,
    gamma=0.99,
    lr=1e-4,
    device="cuda"
):
    state_shape = (4, 84, 84)

    env = make_atari_env(env_id)
    num_actions = env.action_space.n

    replay_buffer = ReplayBuffer(replay_size, state_shape)
    agent = DQN_Agent(
        state_shape,
        num_actions,
        replay_buffer,
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        device=device
    )

    global_step = 0
    
    episode_rewards = []
    try:
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                global_step += 1

                # Train
                if global_step > start_learning and global_step % train_freq == 0:
                    agent.train_step()

            episode_rewards.append(episode_reward)

            print(
                f"Episode {episode} | Reward: {episode_reward:.1f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )


    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user (Ctrl+C). Saving progress...")

    finally:
        # Always save — even on normal finish
        agent.save_model("dqn_checkpoint.pt")
        np.save("episode_rewards.npy", np.array(episode_rewards))
        env.close()

        print("[INFO] Model saved to dqn_checkpoint.pt")
        print("[INFO] Rewards saved to episode_rewards.npy")
    
    env.close()
    return {
        'episode_rewards': episode_rewards
    }
'''
import gymnasium as gym
import numpy as np
import torch
import os

from helper_classes.replay_buffer import ReplayBuffer
from models.dqn_agent import DQN_Agent
from helper_classes.wrappers import make_atari_env


def train_dqn(
    env_id="ALE/BattleZone-v5",
    num_episodes=500,
    replay_size=100_000,
    batch_size=64,
    start_learning=10_000,
    train_freq=4,
    target_update_freq=1000,
    gamma=0.99,
    lr=1e-4,
    device="cuda",
    mode="train"   # <<< NEW
):
    """
    mode:
        - "train"  : no rendering, fast training
        - "deploy" : load weights and play 5 episodes with rendering
    """

    state_shape = (4, 84, 84)

    render = (mode == "deploy")
    env = make_atari_env(env_id, render=render)

    num_actions = env.action_space.n

    replay_buffer = ReplayBuffer(replay_size, state_shape)
    agent = DQN_Agent(
        state_shape,
        num_actions,
        replay_buffer,
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        device=device
    )

    # ===== Checkpoint paths =====
    WEIGHT_FILE = "dqn_weights.pt"
    REWARD_FILE = "dqn_episode_rewards.npy"

    episode_rewards = []
    global_step = 0

    if mode == "deploy":
        # Load saved weights if available
        if os.path.exists(WEIGHT_FILE):
            agent.load_model(WEIGHT_FILE)
            print(f"[INFO] Loaded weights from {WEIGHT_FILE}")
        else:
            print(f"[WARNING] No saved weights found at {WEIGHT_FILE}. Cannot deploy.")
            env.close()
            return None

        # Play 5 episodes with rendering
        for episode in range(5):
            state, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                state = next_state
                episode_reward += reward

            print(f"[DEPLOY] Episode {episode + 1} Reward: {episode_reward:.1f}")
            episode_rewards.append(episode_reward)

        env.close()
        return {"episode_rewards": episode_rewards}

    # ===== Training mode =====
    try:
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                global_step += 1

                # Train step
                if global_step > start_learning and global_step % train_freq == 0:
                    agent.train_step()

            episode_rewards.append(episode_reward)

            print(
                f"Episode {episode} | Reward: {episode_reward:.1f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user (Ctrl+C).")

    finally:
        # ===== Always save progress =====
        agent.save_model(WEIGHT_FILE)
        np.save(REWARD_FILE, np.array(episode_rewards))

        env.close()

        print(f"[INFO] Model saved to {WEIGHT_FILE}")
        print(f"[INFO] Rewards saved to {REWARD_FILE}")

    return {
        "episode_rewards": episode_rewards
    }
