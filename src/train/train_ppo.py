import torch
import gymnasium as gym
import numpy as np
from helper_classes.PPO_helper_classes.Actor_Critic_Model_Wrapper import ActorCritic  # your helper class
from helper_classes.PPO_helper_classes.rollout_buffer import RolloutBuffer
from helper_classes.PPO_helper_classes.GAE import GAE
from helper_classes.PPO_helper_classes.RunningMeanStd_Normalizer import RunningMeanStd
from helper_classes.PPO_helper_classes.MiniBatch_generator import minibatch_generator
from models.ppo_agent import PPO_PolicyNet

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
    device="cuda"
):
    env = gym.make(env_id)
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n

    # Initialize policy network & wrapper
    model = PPO_PolicyNet(obs_shape, num_actions).to(device)  # your NN model with actor & critic outputs
    agent = ActorCritic(model, device)

    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    # Buffers
    rollout_buffer = RolloutBuffer(rollout_length, obs_shape, device)
    obs_rms = RunningMeanStd(obs_shape)  # for normalization if needed

    episode_rewards = []
    episode_reward = 0
    state, _ = env.reset()

    for episode in range(num_episodes):
        for step in range(rollout_length):
            state_norm = obs_rms.normalize(state)
            action, log_prob, value = agent.act(torch.tensor(state_norm).float().to(device))

            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward

            rollout_buffer.add(state, action, reward, done, log_prob, value)
            state = next_state

            if done or truncated:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                state, _ = env.reset()

        # Compute advantages and returns after rollout collection
        last_value = agent.forward(torch.tensor(obs_rms.normalize(state)).float().to(device))[1].item()
        rollout_buffer.compute_returns_and_advantages(last_value, gamma, lam)

        # PPO update
        for epoch in range(epochs):
            for batch in rollout_buffer.get(mini_batch_size):
                states, actions, old_log_probs, values, advantages, returns = batch
                # Compute PPO loss & optimize here, call agent.evaluate(...)
                # (You implement PPO loss inside here)

        rollout_buffer.clear()

        # Optional: print progress
        print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {np.mean(episode_rewards[-10:]):.2f}")

    env.close()
    return {"episode_rewards": episode_rewards}
