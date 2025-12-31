import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


############################################
# DQN NETWORK (Atari-style CNN)
############################################
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        c, h, w = input_shape  # (4, 84, 84)

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Compute conv output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out_size = self.conv(dummy).view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = x / 255.0                # normalize Atari frames
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


############################################
# DQN AGENT
############################################
class DQN_Agent:
    def __init__(
        self,
        input_shape,
        num_actions,
        replay_buffer,
        lr=1e-4,
        gamma=0.99,
        batch_size=64,
        target_update_freq=1000,
        device="cuda"
    ):
        self.device = device
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.policy_net = DQN(input_shape, num_actions).to(device)
        self.target_net = DQN(input_shape, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.replay_buffer = replay_buffer

        self.epsilon = 1.0
        #self.epsilon_decay = 0.995 --> we disable epsilon decay for better training stability
        #added fixes 
        
        self.epsilon_decay_steps = 1_000_000  # Atari standard
        self.total_steps = 0
        self.epsilon_min = 0.01
        
        self.step_count = 0

    ########################################
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        state = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax(1).item()

    ########################################
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)

        #states = torch.as_tensor(batch["states"], device=self.device)
        #actions = torch.as_tensor(batch["actions"], device=self.device).unsqueeze(1)
        #rewards = torch.as_tensor(batch["rewards"], device=self.device)
        #next_states = torch.as_tensor(batch["next_states"], device=self.device)
        #dones = torch.as_tensor(batch["dones"], device=self.device)
        states = torch.from_numpy(batch["states"]).to(self.device).float()
        next_states = torch.from_numpy(batch["next_states"]).to(self.device).float()
        actions = torch.from_numpy(batch["actions"]).to(self.device).long().unsqueeze(1)
        rewards = torch.from_numpy(batch["rewards"]).to(self.device).float()
        dones = torch.from_numpy(batch["dones"]).to(self.device).float()
        # Q(s, a)
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # max Q'(s')
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * (1 - dones) * next_q_values

        loss = nn.functional.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Target network update
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target()

        # Epsilon decay --> an old way of doing it, we switched to linear decay as this was cusing colapses in training
        #if self.epsilon > self.epsilon_min:
        #    self.epsilon *= self.epsilon_decay
            
        # Linear epsilon decay (step-based)
        self.total_steps += 1
        self.epsilon = max(
            self.epsilon_min,
            1.0 - self.total_steps / self.epsilon_decay_steps
        )


    ########################################
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    ########################################
    def save_model(self, filepath):
        torch.save(self.policy_net.state_dict(), filepath)

    def load_model(self, filepath):
        self.policy_net.load_state_dict(torch.load(filepath))
        self.policy_net.to(self.device)
