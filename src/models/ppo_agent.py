import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


class PPOActorCritic(nn.Module):
    """Shared CNN feature extractor + separate actor & critic heads."""

    def __init__(self, action_size):
        super().__init__()

        # CNN feature extractor (Atari-standard)
        self.feature_net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )

        # Compute final feature size
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 84, 84)
            n_features = self.feature_net(dummy).shape[1]

        self.actor = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

        self.critic = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.feature_net(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value


class PPOAgent:
    def __init__(
        self,
        state_shape,
        action_size,
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.1,
        epochs=4,
        batch_size=128,
        steps_per_update=2048
    ):
        """
        Proximal Policy Optimization (PPO) Agent.
        """
        self.gamma = gamma
        self.lambda_ = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.steps_per_update = steps_per_update
        self.action_size = action_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = PPOActorCritic(action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Storage buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def choose_action(self, state):
        """Sample an action using the actor's probability distribution."""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits, value = self.model(state)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(dist.log_prob(action))
        self.values.append(value)

        return action.item()

    def store_reward(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(float(done))

    def compute_gae(self, next_value):
        """Generalized Advantage Estimation."""
        values = self.values + [next_value]
        gae = 0
        advantages = []

        for t in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[t]
                + self.gamma * values[t + 1] * (1 - self.dones[t])
                - values[t]
            )
            gae = delta + self.gamma * self.lambda_ * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + v for adv, v in zip(advantages, self.values)]
        return advantages, returns

    def learn(self, next_state):
        """Perform PPO update after collecting rollout of steps_per_update."""
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        _, next_value = self.model(next_state)

        # Convert storage to tensors
        advantages, returns = self.compute_gae(next_value.detach())

        states = torch.cat(self.states).to(self.device)
        actions = torch.stack(self.actions).to(self.device)
        log_probs_old = torch.stack(self.log_probs).detach().to(self.device)
        values_old = torch.cat(self.values).detach().to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(states)
        for _ in range(self.epochs):
            idxs = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch = idxs[start:end]

                logits, value = self.model(states[batch])
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions[batch])

                ratio = (log_probs - log_probs_old[batch]).exp()
                surr1 = ratio * advantages[batch]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[batch]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(value.squeeze(), returns[batch])

                entropy_loss = dist.entropy().mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

        # Reset rollout buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
