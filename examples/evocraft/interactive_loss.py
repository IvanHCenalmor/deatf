import tensorflow as tf
import numpy as np

from vectors_to_blocks import *
from visualize_networks import show_network

def interactive_loss(probabilities, wanted_blocks, bounds):
    
    probabilities = tf.math.sigmoid(probabilities)
    prob = tf.math.argmax(probabilities, 1)
    
    blocks = prob
    orientations = np.zeros(bounds, dtype=int)
    
    printed_blocks = blocks.numpy()
    for i, val in enumerate(wanted_blocks):
        printed_blocks = np.where(printed_blocks == i,val, printed_blocks)
        
    #print('Wanted_blocks: ', printed_blocks)
    # Build it in Minecraft
    reward = []
    for indx in range(10):
        random_indx = np.random.choice(list(range(len(printed_blocks))))

        puntuaction = show_network(printed_blocks[random_indx], bounds, [10*indx,10,0])
        reward.append(puntuaction)

    float_rewards = np.mean(reward)
    
    return float_rewards