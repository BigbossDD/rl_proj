import numpy as np

  
def minibatch_generator(batch_size, mini_batch_size, *arrays):
    """
    Yields shuffled minibatches.

    Args:
        batch_size (int): total rollout length
        mini_batch_size (int)
        *arrays: arrays to be batched together
                 (e.g. states, actions, advantages)

    returns -->
        iterator of tuple(array_slice1, array_slice2, ...)
    """

    assert batch_size % mini_batch_size == 0, \
        "batch_size must be divisible by mini_batch_size"

    indices = np.random.permutation(batch_size)

    for start in range(0, batch_size, mini_batch_size):
        end = start + mini_batch_size
        mb_indices = indices[start:end]

        yield tuple(arr[mb_indices] for arr in arrays)
