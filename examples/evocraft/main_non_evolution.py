import sys
sys.path.append('../..')

from evoflow.network import MLPDescriptor, MLP

import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.models import Model
from vectors_to_blocks import *

from interactive_loss import interactive_loss
from symmetry_loss import total_symmetry
from create_data import create_data

def evaluate(probabilities):
    probabilities = tf.math.sigmoid(probabilities)
    loss = total_symmetry(probabilities[0])    
    return loss

def model_eval(network, training_iterations, bounds, wanted_blocks):    
    
    data = create_data(bounds)
    
    inp = Input(shape=data.shape[1:])
    out = network.building(inp)
    out = Reshape([len(wanted_blocks)]+bounds)(out)
    model = Model(inputs=inp, outputs=out)
    model.summary()
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)

    @tf.function
    def train_step(x_train):
        
        with tf.GradientTape() as tape:   
            probabilities = model(x_train, training=True)
            loss = evaluate(probabilities)
            
        gradients_of_generator = tape.gradient(loss, model.trainable_variables)  
        opt.apply_gradients(zip(gradients_of_generator, model.trainable_variables))
        return loss
    
    for epoch in range(training_iterations):
        loss = train_step(data)           
        print("Epoch {:03d}: Loss: {}".format(epoch, loss))
        
    prediction = model.predict(data)
    ev = interactive_loss(prediction, wanted_blocks, bounds)
    return ev

if __name__ == '__main__':
    
    print('ADVISE: Remember to activate the server in order to see and evaluate the created structure after training.')
    
    bounds = [5,5,5]
    wanted_blocks = [23,45,64,22,33]
    
    desc = MLPDescriptor(10,3,len(wanted_blocks))
    desc.random_init(input_size=3, output_size=[len(wanted_blocks)]+bounds, 
                     nlayers=10, max_layer_size=200, 
                     dropout=True, batch_norm=True)
    network = MLP(desc)
    
    preds = model_eval(network, 1000, bounds, wanted_blocks, verbose=1)
    print(preds)
