import numpy as np

import sys
sys.path.append('../')

from Network import MLPDescriptor, MLP

import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.models import Model
from vectors_to_blocks import *

from symmetry_loss import total_symmetry
from create_data import create_data

def evaluate_prediction(probabilities, wanted_blocks, verbose):
    
    probabilities = tf.math.sigmoid(probabilities)
    prob = tf.math.argmax(probabilities, 1)
    
    blocks = prob[0]
    orientations = np.zeros(bounds, dtype=int)
    
    if verbose:
        print('Probabilities: ', probabilities, ' Loss: ', evaluate(probabilities))
        print('Blocks: ',blocks.numpy())
    
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

def evaluate(probabilities):
    probabilities = tf.math.sigmoid(probabilities)
    loss = total_symmetry(probabilities[0])    
    return loss

def model_eval(network, training_iterations, bounds, wanted_blocks, verbose=0):    
    
    data = create_data(bounds)
    
    inp = Input(shape=data.shape[1:])
    out = network.building(inp)
    out = Reshape([len(wanted_blocks)]+bounds)(out)
    model = Model(inputs=inp, outputs=out)
    if verbose: 
        model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)

    @tf.function
    def train_step(x_train):
        
        with tf.GradientTape() as tape:   
            probabilities = model(x_train, training=True)
            #print(f'Out blocks: {out_blocks}')
            loss = evaluate(probabilities)
            if loss < 0.001:
                evaluate_prediction(probabilities)
            
        #print('Trainable variables: \n',model.trainable_variables)
        gradients_of_generator = tape.gradient(loss, model.trainable_variables)  
        #print(gradients_of_generator)
        opt.apply_gradients(zip(gradients_of_generator, model.trainable_variables))
        return loss
    
    for epoch in range(training_iterations):
        loss = train_step(data) 
        if epoch % 100 == 0 and verbose:            
            print("Epoch {:03d}: Loss: {}".format(epoch, loss))
        
    prediction = model.predict(data)
    ev = evaluate_prediction(prediction, wanted_blocks, verbose)
    return ev

if __name__ == '__main__':
    
    print('ADVISE: Remember to activate the server in order to see and evaluate the created structure after training.')
    
    #tf.config.run_functions_eagerly(True)

    bounds = [5,5,5]
    wanted_blocks = [23,45,64,22,33]
    
    desc = MLPDescriptor(10,3,len(wanted_blocks))
    desc.random_init(input_size=3, output_size=[len(wanted_blocks)]+bounds, 
                     nlayers=10, max_layer_size=200, 
                     dropout=True, batch_norm=True)
    network = MLP(desc)
    
    preds = model_eval(network, 1000, bounds, wanted_blocks, verbose=1)
    print(preds)
