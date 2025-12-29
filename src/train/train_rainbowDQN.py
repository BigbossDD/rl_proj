import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym
from collections import defaultdict
from helper_classes.Rainbow_DQN_helper_classes.PERbuffer import PERBuffer
from helper_classes.Rainbow_DQN_helper_classes.QNetwork import QNetwork
from helper_classes.Rainbow_DQN_helper_classes.noisy_linear import NoisyLinear
# helper classes (already implemented by you)
# from helpers.noisy_linear import NoisyLinear
# from helpers.per_buffer import PERBuffer
# from helpers.q_network import QNetwork


# ======================================================
# Rainbow DQN Agent (LOGIC ONLY)
# ======================================================
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

        self.q_net = QNetwork(obs_shape, n_actions).to(device)
        self.target_q_net = QNetwork(obs_shape, n_actions).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.replay_buffer = PERBuffer(replay_size, alpha)

        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame_idx = 0

        self.n_actions = n_actions

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.q_net(state)
        return q_vals.argmax(dim=1).item()

    def store(self, s, a, r, s_next, done):
        self.replay_buffer.add(s, a, r, s_next, done)

    def _beta(self):
        return min(
            1.0,
            self.beta_start + self.frame_idx * (1.0 - self.beta_start) / self.beta_frames
        )

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        self.frame_idx += 1
        beta = self._beta()

        s, a, r, s_next, d, idxs, w = self.replay_buffer.sample(
            self.batch_size, beta
        )

        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        s_next = torch.tensor(s_next, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.long, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)
        w = torch.tensor(w, dtype=torch.float32, device=self.device)

        q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            a_next = self.q_net(s_next).argmax(1)
            q_next = self.target_q_net(s_next).gather(
                1, a_next.unsqueeze(1)
            ).squeeze(1)
            target = r + self.gamma * q_next * (1 - d)

        td_error = target - q
        loss = (w * td_error.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.replay_buffer.update_priorities(
            idxs, td_error.abs().detach().cpu().numpy()
        )

        return loss.item()

    def update_target(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())


# ======================================================
# Training function (CALLED BY MAIN)
# ======================================================
def train_rainbowDQN(
    env_id,
    num_episodes,
    replay_size,
    batch_size,
    start_learning,
    train_freq,
    target_update_freq,
    gamma,
    lr,
    device,
):
    env = gym.make(env_id)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    agent = RainbowDQNAgent(
        obs_shape,
        n_actions,
        gamma,
        lr,
        device,
        replay_size,
        batch_size,
        target_update_freq,
    )

    stats = defaultdict(list)
    global_step = 0

    for ep in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            if global_step > start_learning and global_step % train_freq == 0:
                loss = agent.learn()
                if loss is not None:
                    stats["loss"].append(loss)

            if global_step % target_update_freq == 0:
                agent.update_target()

            global_step += 1

        stats["episode_rewards"].append(ep_reward)

    env.close()
    return stats
