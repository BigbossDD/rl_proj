import numpy as np

class RunningMeanStd:
    """
    Tracks running mean & variance using Welford's algorithm.
    Commonly used for:
      - state normalization
      - reward normalization
    """

    def __init__(self, shape, epsilon=1e-4):
        """
        Args:
            shape (tuple): shape of data being normalized
            epsilon (float): for numerical stability
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # start with small count to avoid div by zero

    def update(self, x):
        """
        Update mean & variance with a new batch.

        Args:
            x (np.ndarray): new data batch

        returns --> Nothing
        """
        x = np.asarray(x)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        """
        Normalize input using running mean & std.

        Args:
            x (np.ndarray)

        returns -->
            normalized_x (np.ndarray)
        """
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)
