import numpy as np

def batch(x, size, i):
    """
    :param x: Poplation; set of solutions intended to be fed to the net in the input
    :param size: Size of the batch desired
    :param i: Index of the last solution used in the last epoch
    :return: The index of the last solution in the batch (to be provided to this same
             function in the next epoch, the solutions in the actual batch, and their
             respective fitness scores
    """

    if i + size > x.shape[0]:  # In case there are not enough solutions before the end of the array

        index = i + size-x.shape[0]  # Select all the individuals until the end and restart
        return np.concatenate((x[i:, :], x[:index, :]))
    else:  # Easy case
        index = i+size
        return x[i:index, :]