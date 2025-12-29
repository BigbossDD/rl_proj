import numpy as np


class GAE:
    """
    Computes Generalized Advantage Estimation (GAE) for PPO.
    """

    @staticmethod
    def compute(values, rewards, dones, last_value, gamma, lam):
        """
        Args:
            values (np.ndarray): V(s_t)          shape: [T]
            rewards (np.ndarray): r_t           shape: [T]
            dones (np.ndarray): episode termination mask (0 or 1)
            last_value (float): V(s_T)
            gamma (float)
            lam (float): GAE λ parameter

        returns --> 
            advantages (np.ndarray)
            returns (np.ndarray)
        """
        T = len(rewards)

        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        # append last value for bootstrapping
        values = np.append(values, last_value)

        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae

        returns = advantages + values[:-1]

        return advantages, returns
