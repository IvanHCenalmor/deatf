import sys
sys.path.append('../..')

import tensorflow as tf
import numpy as np
import re

from deatf.network import MLPDescriptor, MLP

from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.models import Model

from interactive_loss import interactive_loss
from symmetry_loss import total_symmetry


def evaluate(probabilities):
    """
    Evaluation method for the networks during evolution process.
    
    :param probabilities: Data that will be shown, it contains the probability
                         distribution that describes the build.     
    :return: Value calculated with the symmetry of the received values.
    """
    probabilities = tf.math.sigmoid(probabilities)
    loss = total_symmetry(probabilities[0])    
    return loss

def model_eval(network, train_data, test_data, iters, bounds, wanted_blocks):    
    """
    The model that will evaluate the data is a simple MLP. Data passed to those 
    models in this case will be noise, models will have to predict a probability
    distribution with the size of the bounds of the build. The evaluation will
    measure the symmetry that is in each created build. Once the model is evaluated
    an example of its prediction will be created in Minecraft.
    
    :param network: Network that will be used to build the final model to be evaluated.
    :param train_data: Input data for training, this data will only be used to 
                       give it to the created networks and train them.
    :param test_data: Input data for testing, this data will only be used to 
                      give it to the created networks and test them. It can not be used during
                      training in order to get a real feedback.
    :param iters: Number of iterations that the model will be trained.
    :param bounds: Bounds of the build that will be created in Minecraft.
    :param wanted_blocks: List with the indexes of the desired blocks for the build.
    :return: Symmetry loss obtained with the test data that evaluates the true
             performance of the network.
    """
    #f.write(network.descriptor)
    
    inp = Input(shape=train_data.shape[1:])
    out = network.building(inp)
    out = Reshape(bounds+[len(wanted_blocks)])(out)
    model = Model(inputs=inp, outputs=out)
    
    #model.summary()
    #model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)

    @tf.function
    def train_step(x_train):
        
        with tf.GradientTape() as tape:   
            probabilities = model(x_train, training=True)
            loss = evaluate(probabilities)
            
        gradients_of_generator = tape.gradient(loss, model.trainable_variables)  
        opt.apply_gradients(zip(gradients_of_generator, model.trainable_variables))
        return loss
    
    for epoch in range(iters):
        loss = train_step(train_data)
        if epoch % 100 == 0:
            print("Epoch {:03d}: Loss: {}".format(epoch, loss))
            #f.write("Epoch {:03d}: Loss: {}".format(epoch, loss))
        
    probabilities = model(test_data)
    ev = evaluate(probabilities)

    index = [int(''.join(re.findall(r'\d', model.name))) if model.name != 'model' else 0, 0]    

    interactive_loss(probabilities, wanted_blocks, bounds, index, False)
    #f.write('Final ev: {}'.format(ev))

    return ev,


if __name__ == '__main__':
    
    print('ADVISE: Remember to activate the server in order to see and evaluate the created structure after training.')
    
    bounds = [9,9,9]
    wanted_blocks = [-1,164, 169, 173]
    
    train_noise = np.random.normal(size=(5000, np.array(bounds).prod()))
    test_noise = np.random.normal(size=(2000,  np.array(bounds).prod()))
    val_noise = np.random.normal(size=(2000,  np.array(bounds).prod()))
    
    desc = MLPDescriptor(10,3,len(wanted_blocks))
    desc.random_init(input_size=3, output_size=[len(wanted_blocks)]+bounds, 
                     max_num_layers=10, max_num_neurons=200, max_stride=None, max_filter=None,
                     dropout=True, batch_norm=True)
    network = MLP(desc)
    
    preds = model_eval(network, train_noise, val_noise, 500, bounds, wanted_blocks)
    print(preds)
