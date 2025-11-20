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

    env = make_atari_env(env_id)
    state, _ = env.reset()

    action_size = env.action_space.n
    state_shape = env.observation_space.shape

    agent = PPOAgent(
        state_shape=state_shape,
        action_size=action_size,
        steps_per_update=steps_per_update,
        lr=2.5e-4
    )

    total_steps = 0
    episode_reward = 0
    episode_num = 0

    while total_steps < total_timesteps:
        for _ in range(steps_per_update):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_reward(reward, done)

            state = next_state
            episode_reward += reward
            total_steps += 1

            if done:
                print(f"Episode: {episode_num}, Reward: {episode_reward}")
                state, _ = env.reset()
                episode_reward = 0
                episode_num += 1

        # PPO update
        agent.learn(next_state)

        # Save checkpoint
        torch.save(agent.model.state_dict(), save_path)
        print("Model saved →", save_path)

    env.close()


if __name__ == "__main__":
    train_ppo()
