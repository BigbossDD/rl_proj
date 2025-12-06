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
        pass

    def update(self, x):
        """
        Update mean & variance with a new batch.

        Args:
            x (np.ndarray)

        
        returns --> Nothing
            
        """
        pass

    def normalize(self, x):
        """
        Normalize input using running mean & std.

        Args:
            x (np.ndarray)

        
        returns -->
            normalized_x (np.ndarray)
        """
        pass
