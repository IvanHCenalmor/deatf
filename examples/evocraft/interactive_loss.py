import tensorflow as tf
import numpy as np

from vectors_to_blocks import *

def interactive_loss(probabilities, wanted_blocks, bounds):
    
    probabilities = tf.math.sigmoid(probabilities)
    prob = tf.math.argmax(probabilities, 1)
    
    blocks = prob[0]
    orientations = np.zeros(bounds, dtype=int)
    
    printed_blocks = blocks.numpy()
    for i, val in enumerate(wanted_blocks):
        printed_blocks = np.where(printed_blocks == i,val, printed_blocks)
        
    #print('Wanted_blocks: ', printed_blocks)
    # Build it in MInecraft
    build_zone(printed_blocks, [0, 10, 0], False,
           orientations, False, len(bounds)+1)
    # Ask Rating Human
    print("Rate the creation from 1 to 5:")
    reward = float(getch())
    print(reward)
    # Clean blocks function afterwards: clean all zone
    clean_zone([bounds[0]] + bounds, [0, 10, 0])

    scaled_reward = float(max(min(reward, 5), 1)*10)
    float_rewards = scaled_reward/5
    
    return float_rewards


def getch():
    """ Allows to input values without pressing enter """
    import termios
    import sys
    import tty

    def _getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    return _getch()