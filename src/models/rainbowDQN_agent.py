import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.optim as optim
import numpy as np

from helper_classes.Rainbow_DQN_helper_classes.QNetwork import QNetwork
from helper_classes.Rainbow_DQN_helper_classes.PERbuffer import PERBuffer


class RainbowDQNAgent:
    def __init__(
        self,
        obs_shape,
        n_actions,
        gamma,
        lr,
        device,
        replay_size,
        batch_size,
        target_update_freq,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=1_000_000,
    ):
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # ---- C51 parameters ----
        self.n_actions = n_actions
        self.num_atoms = 51
        self.Vmin = -10.0
        self.Vmax = 10.0
        self.delta_z = (self.Vmax - self.Vmin) / (self.num_atoms - 1)
        self.support = torch.linspace(self.Vmin, self.Vmax, self.num_atoms).to(device)

        # ---- Networks ----
        self.q_net = QNetwork(obs_shape, n_actions).to(device)
        self.target_q_net = QNetwork(obs_shape, n_actions).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # ---- PER ----
        self.replay_buffer = PERBuffer(replay_size, alpha)
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame_idx = 0  # tracks how many learning steps have been taken

        # Optional: track total environment steps separately if needed
        self.total_steps = 0

    # =====================================================
    def _beta(self):
        # Anneal beta from beta_start to 1 over beta_frames frames
        return min(
            1.0,
            self.beta_start + self.frame_idx * (1.0 - self.beta_start) / self.beta_frames
        )

    # =====================================================
    def select_action(self, state):
        """
        Select action given state (greedy over expectation)
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device) / 255.0

        with torch.no_grad():
            log_probs = self.q_net(state)                # [1, A, atoms]
            probs = log_probs.exp()
            q_values = torch.sum(probs * self.support, dim=2)
            action = q_values.argmax(1).item()

        return action

    # =====================================================
    def store(self, s, a, r, s_next, done):
        """
        Store experience transition in replay buffer
        """
        self.replay_buffer.add(s, a, r, s_next, done)

    # =====================================================
    def learn(self):
        # --- FIX: Prevent learning if replay buffer not large enough ---
        if len(self.replay_buffer) < self.batch_size:
            return None  # Skip learning until enough samples

        # Optional: if you track total environment steps, add a "start learning after" threshold:
        # if self.total_steps < some_threshold:
        #     return None

        self.frame_idx += 1  # increment learning step count
        beta = self._beta()

        # --- FIX: sample() may return None if not enough valid data ---
        sample = self.replay_buffer.sample(self.batch_size, beta)
        if sample is None:
            return None  # Not enough data to sample full batch, skip this step

        states, actions, rewards, next_states, dones, idxs, weights = sample

        # Already tensors from PERBuffer → DO NOT re-wrap
        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        # =================================================
        # Current distribution
        log_probs = self.q_net(states)                   # [B,A,Z]
        actions = actions.clone().unsqueeze(1).unsqueeze(2)
        actions = actions.expand(-1, 1, self.num_atoms)

        chosen_log_probs = log_probs.gather(1, actions).squeeze(1)
        probs = chosen_log_probs.exp()
        q_values = torch.sum(probs * self.support, dim=1)

        # =================================================
        # Target distribution (Double DQN)
        with torch.no_grad():
            next_log_probs = self.q_net(next_states)
            next_probs = next_log_probs.exp()
            next_q = torch.sum(next_probs * self.support, dim=2)
            next_actions = next_q.argmax(1)

            next_actions = next_actions.clone().unsqueeze(1).unsqueeze(2)
            next_actions = next_actions.expand(-1, 1, self.num_atoms)

            target_log_probs = self.target_q_net(next_states)
            target_probs = target_log_probs.gather(1, next_actions).squeeze(1).exp()

            target_q = torch.sum(target_probs * self.support, dim=1)
            target = rewards + self.gamma * target_q * (1.0 - dones)

        # =================================================
        td_error = target - q_values
        loss = (weights * td_error.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # ---- PER priority update ----
        # FIX: Send raw tensor errors, PERBuffer will detach and convert internally
        self.replay_buffer.update_priorities(idxs, td_error)

        return loss.item()

    # =====================================================
    def update_target(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())
