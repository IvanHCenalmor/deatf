import numpy as np

def batch(data, n, i):
    """
    :param data: Set of solutions intended to be fed to the network.
    :param n: Size of the desired batch. 
    :param i: Index of the last solution used in the last epoch.
    :return: The batch of data form x with size n since the index i of the data.
    """

    if i + n > data.shape[0]:  
        # In case there are not enough solutions before the end of the array
        index = i + n-data.shape[0]  # Select all the individuals until the end and restart
        return np.concatenate((data[i:, :], data[:index, :]))
    else:  
        # Easy case
        index = i+n
        return data[i:index, :]