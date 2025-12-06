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
    pass
