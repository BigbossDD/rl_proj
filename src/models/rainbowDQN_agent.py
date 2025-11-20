import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# using -->  Dueling + Noisy + Categorical DQN + PER + N-step

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters(sigma_init)
        self.reset_noise()
##########################################
    def reset_parameters(self, sigma_init):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(sigma_init / np.sqrt(self.out_features))
##########################################
    def reset_noise(self):
        eps_in = torch.randn(self.in_features)
        eps_out = torch.randn(self.out_features)
        f = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = f(eps_in)
        eps_out = f(eps_out)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)
##########################################
    def forward(self, x):
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

####################################################################################
# -------------------------
# Categorical DQN Network
# -------------------------
class RainbowNetwork(nn.Module):
    def __init__(self, action_size, atom_size=51, v_min=-10, v_max=10):
        super().__init__()
        self.action_size = action_size
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max

        self.support = torch.linspace(v_min, v_max, atom_size)

        # CNN torso (Atari standard)
        self.feature = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )

        self.fc_hidden = NoisyLinear(3136, 512)
        self.fc_value = NoisyLinear(512, atom_size)
        self.fc_advantage = NoisyLinear(512, action_size * atom_size)
##########################################
    def forward(self, x):
        dist = self.dist(x)
        return torch.sum(dist * self.support, dim=2)
##########################################
    def dist(self, x):
        x = self.feature(x / 255.0)
        x = F.relu(self.fc_hidden(x))

        value = self.fc_value(x).view(-1, 1, self.atom_size)
        advantage = self.fc_advantage(x).view(-1, self.action_size, self.atom_size)

        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        dist = F.softmax(q_atoms, dim=2)
        return dist
##########################################
    def reset_noise(self):
        self.fc_hidden.reset_noise()
        self.fc_value.reset_noise()
        self.fc_advantage.reset_noise()

####################################################################################
# -------------------------
# Prioritized Replay Buffer
# -------------------------
class PERBuffer:
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
##########################################
    def push(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
##########################################
    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return samples, indices, torch.tensor(weights, dtype=torch.float32)
##########################################
    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

################################################################################################
# -------------------------
# Rainbow DQN Agent
# -------------------------
class RainbowDQNAgent:
    def __init__(self, action_size):
        self.action_size = action_size

        self.v_min = -10
        self.v_max = 10
        self.atom_size = 51
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size)

        self.gamma = 0.99
        self.lr = 1e-4

        self.n_step = 3
        self.nstep_buffer = deque(maxlen=self.n_step)

        self.replay = PERBuffer()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = RainbowNetwork(action_size).to(self.device)
        self.target = RainbowNetwork(action_size).to(self.device)
        self.target.load_state_dict(self.model.state_dict())

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
##########################################
    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.model(state)
        return q.argmax(1).item()
##########################################
    def remember(self, s, a, r, s2, done):
        self.nstep_buffer.append((s, a, r, s2, done))
        if len(self.nstep_buffer) < self.n_step:
            return

        R, s_n, done_n = 0, None, False
        for idx, (ss, aa, rr, ss2, dd) in enumerate(self.nstep_buffer):
            R += rr * (self.gamma ** idx)
            s_n = ss2
            if dd:
                done_n = True
                break

        s0, a0, _, _, _ = self.nstep_buffer[0]
        self.replay.push((s0, a0, R, s_n, done_n))
##########################################
    def learn_from_replay(self, batch_size=32):
        if len(self.replay.buffer) < batch_size:
            return

        samples, idxs, weights = self.replay.sample(batch_size)

        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        weights = weights.to(self.device)

        dist = self.model.dist(states)
        next_dist = self.target.dist(next_states).detach()

        next_actions = (self.model(next_states)).argmax(1)
        next_dist = next_dist[range(batch_size), next_actions]

        projected = self.projection_distribution(next_dist, rewards, dones)

        dist_a = dist[range(batch_size), actions]
        loss = -(projected * torch.log(dist_a + 1e-6)).sum(1)
        loss = (loss * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        new_prios = loss.detach().cpu().numpy() + 1e-5
        self.replay.update_priorities(idxs, new_prios)

        self.model.reset_noise()
        self.target.reset_noise()
##########################################
    def projection_distribution(self, next_dist, rewards, dones):
        batch_size = len(rewards)
        proj = torch.zeros(batch_size, self.atom_size).to(self.device)

        delta_z = (self.v_max - self.v_min) / (self.atom_size - 1)
        for i in range(self.atom_size):
            tz_j = rewards + (1 - dones) * (self.gamma ** self.n_step) * self.support[i]
            tz_j = tz_j.clamp(self.v_min, self.v_max)
            b_j = (tz_j - self.v_min) / delta_z
            l = b_j.floor().long()
            u = b_j.ceil().long()

            for idx in range(batch_size):
                proj[idx, l[idx]] += next_dist[idx, i] * (u[idx] - b_j[idx])
                proj[idx, u[idx]] += next_dist[idx, i] * (b_j[idx] - l[idx])

        return proj
####################################################################################
# -------------------------
# Training Loop
# -------------------------
def train_rainbow_dqn(env, agent, episodes=5000):
    for ep in range(episodes):
        s, _ = env.reset()
        total_reward = 0

        done = False
        while not done:
            a = agent.choose_action(s)
            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            agent.remember(s, a, r, s2, done)
            agent.learn_from_replay()

            s = s2
            total_reward += r

        if ep % 20 == 0:
            agent.target.load_state_dict(agent.model.state_dict())
            print(f"Episode: {ep}, Reward: {total_reward}")

    return agent
