import numpy as np

from vectors_to_blocks import *

def evaluate_with_visualization(predictions, bounds, wanted_blocks, evaluate=True):
    
    # Transform probabilities into actual blocks
    predictions = np.argmax(predictions, axis=-1)
    predictions = np.reshape(predictions, [predictions.shape[0]]+bounds)
    
    for i, block in enumerate(wanted_blocks):   
        predictions = np.where(predictions == i, block, predictions)
    
    print(predictions)

    # current_ind is a counter of the individuals evaluations
    # this is needed to visualize each solution in a different position in the world
    # it would be more elegant not to have a global variable and get the index in the population
    # of the current  individual being evaluated
    current_ind = 0
    total_evaluations = []

    for printed_blocks in predictions:
                    
        # gap is the distance between the visualization of very solution (cube) in the population
        gap = 7
        cube_center = [gap*current_ind,10,0]
        
        current_ind = current_ind + 1
    
        ev = show_network(printed_blocks, bounds, cube_center, evaluate=evaluate)
        total_evaluations.append(ev)
        
    return np.mean(total_evaluations)

def show_network(printed_blocks, bounds, cube_center, evaluate=True):

    # Clean blocks function afterwards: clean all zone
    if cube_center[0]==0:
    	clean_zone([200,200,200] + bounds, cube_center)    
    #clean_zone([bounds[0]] + bounds, cube_center)
   
    
    orientations = np.zeros(bounds, dtype=int)
        
    print('Printed_blocks: ', printed_blocks.shape)

    for i in range(printed_blocks.shape[1]):
    	print(i,printed_blocks[:,i,:])
	

    # Build it in Minecraft
    build_zone(printed_blocks, cube_center, False,
           orientations, False, len(bounds)+1)
    # Ask Rating Human
    if evaluate:
        print("Rate the creation from 1 to 5:")
        reward = float(getch())
        print(reward)
    


        scaled_reward = float(max(min(reward, 5), 1)*10)
        float_rewards = scaled_reward/5
        
        return float_rewards
    else:
        return -1.


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