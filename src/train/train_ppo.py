import torch
import gymnasium as gym
import numpy as np
import os

from helper_classes.PPO_helper_classes.Actor_Critic_Model_Wrapper import ActorCritic
from helper_classes.PPO_helper_classes.rollout_buffer import RolloutBuffer
from helper_classes.PPO_helper_classes.GAE import GAE
from helper_classes.PPO_helper_classes.RunningMeanStd_Normalizer import RunningMeanStd
from helper_classes.PPO_helper_classes.MiniBatch_generator import minibatch_generator
from models.ppo_agent import PPO_PolicyNet

from helper_classes.wrappers import make_atari_env

'''
this is the training script for PPO , it is a connection hub between all elements of PPO 
also here where weights are saved 
it recivies parameters from main 

'''


def train_PPO(
    env_id="ALE/BattleZone-v5",
    num_episodes=1000,
    rollout_length=2048,
    batch_size=64,
    mini_batch_size=32,
    epochs=10,
    gamma=0.99,
    lam=0.95,
    clip_epsilon=0.2,
    lr=3e-4,
    device="cuda",
    mode="train"   # <<< NEW
):
    """
    mode:
        - "train"  : fast, no rendering
        - "deploy" : render environment and run 5 episodes using saved weights
    """

    # Use your Atari preprocessing wrapper, which sets correct input shape and processing
    env = make_atari_env(env_id, frame_stack=4, render=(mode == "deploy"))

    obs_shape = env.observation_space.shape  # should be (4, 84, 84)
    num_actions = env.action_space.n

    print(f"[INFO] PPO env observation shape: {obs_shape}")

    # ===== Initialize policy =====
    model = PPO_PolicyNet(obs_shape, num_actions).to(device)
    agent = ActorCritic(model, device)

    optimizer = torch.optim.Adam(agent.model.parameters(), lr=lr)

    # ===== Buffers =====
    rollout_buffer = RolloutBuffer(rollout_length, obs_shape, device)
    obs_rms = RunningMeanStd(obs_shape)

    # ===== Checkpoint paths =====
    WEIGHT_FILE = "ppo_weights.pt"
    REWARD_FILE = "ppo_episode_rewards.npy"

    episode_rewards = []

    if mode == "deploy":
        # Load weights if available
        if os.path.exists(WEIGHT_FILE):
            agent.model.load_state_dict(torch.load(WEIGHT_FILE, map_location=device))
            agent.model.eval()
            print(f"[INFO] Loaded PPO weights from {WEIGHT_FILE}")
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
                state_norm = obs_rms.normalize(state)
                with torch.no_grad():
                    action, _, _ = agent.act(torch.tensor(state_norm).float().to(device))

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                state = next_state
                episode_reward += reward

            print(f"[DEPLOY] Episode {episode + 1} Reward: {episode_reward:.1f}")
            episode_rewards.append(episode_reward)

        env.close()
        return {"episode_rewards": episode_rewards}

    # ===== Training mode =====
    episode_reward = 0
    state, _ = env.reset()

    try:
        for episode in range(num_episodes):

            for step in range(rollout_length):
                state_norm = obs_rms.normalize(state)

                action, log_prob, value = agent.act(
                    torch.tensor(state_norm).float().to(device)
                )

                next_state, reward, done, truncated, _ = env.step(action)
                episode_reward += reward

                rollout_buffer.add(
                    state, action, reward, done, log_prob, value
                )

                state = next_state

                if done or truncated:
                    episode_rewards.append(episode_reward)
                    episode_reward = 0
                    state, _ = env.reset()

            # ===== GAE computation =====
            last_value = agent.forward(
                torch.tensor(obs_rms.normalize(state)).float().to(device)
            )[1].item()

            rollout_buffer.compute_returns_and_advantages(
                last_value, gamma, lam
            )

            # ===== PPO update =====
            for _ in range(epochs):
                for batch in rollout_buffer.get(mini_batch_size):
                    states, actions, old_log_probs, values, advantages, returns = batch

                    # Implement PPO loss and optimization here (your existing code)
                    pass

            rollout_buffer.clear()

            print(
                f"Episode {episode+1}/{num_episodes} | "
                f"Avg Reward (last 10): {np.mean(episode_rewards[-10:]):.2f}"
            )

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user (Ctrl+C).")

    finally:
        # ===== Always save =====
        torch.save(agent.model.state_dict(), WEIGHT_FILE)
        np.save(REWARD_FILE, np.array(episode_rewards))
        env.close()

        print(f"[INFO] PPO weights saved to {WEIGHT_FILE}")
        print(f"[INFO] Rewards saved to {REWARD_FILE}")

    return {
        "episode_rewards": episode_rewards
    }
