class GAE:
    """
    Computes Generalized Advantage Estimation (GAE) for PPO.
    """

    @staticmethod
    def compute(values, rewards, dones, last_value, gamma, lam):
        """
        Args:
            values (np.ndarray): V(s_t)
            rewards (np.ndarray): r_t
            dones (np.ndarray): episode termination mask
            last_value (float): V(s_T)
            gamma (float)
            lam (float): GAE λ parameter

        returns --> 
            advantages (np.ndarray)
            returns (np.ndarray)
        """
        pass
