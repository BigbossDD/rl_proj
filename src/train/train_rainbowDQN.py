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

    # --- 1. Create environment ---
    env = make_atari_env(env_id)
    action_size = env.action_space.n
    state_shape = env.observation_space.shape  # should be (4, 84, 84)

    # --- 2. Initialize agent ---
    agent = RainbowDQNAgent(
        state_shape=state_shape,
        action_size=action_size,
        lr=1e-4,
        gamma=0.99,
        n_step=3,
        atom_size=51,
        v_min=-10,
        v_max=10,
        replay_capacity=200000,
        batch_size=batch_size
    )

    rewards_history = []
    rolling_avg = deque(maxlen=100)

    for episode in range(total_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        for step in range(max_steps):
            # --- 3. Agent chooses an action ---
            action = agent.choose_action(state)

            # --- 4. Step environment ---
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # --- 5. Store transition ---
            agent.remember(state, action, reward, next_state, done)

            # --- 6. Learn ---
            agent.learn()

            state = next_state
            episode_reward += reward

            if done:
                break

        rewards_history.append(episode_reward)
        rolling_avg.append(episode_reward)

        print(
            f"Episode {episode+1}/{total_episodes} | "
            f"Reward: {episode_reward} | "
            f"Avg(100): {np.mean(rolling_avg):.2f}"
        )

        # --- 7. Save model every 100 episodes ---
        if (episode + 1) % 100 == 0:
            torch.save(agent.model.state_dict(), save_path)
            print(f"Model saved → {save_path}")

    env.close()

    print("\nTraining completed.")
    torch.save(agent.model.state_dict(), save_path)
    print(f"Final model saved → {save_path}")


if __name__ == "__main__":
    train_rainbow_dqn()
