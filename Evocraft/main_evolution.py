import sys
sys.path.append('../')

import tensorflow as tf
import tensorflow.keras.optimizers as opt
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model

from evolution import Evolving, accuracy_error
from Network import MLPDescriptor

from symmetry_loss import total_symmetry
from create_data import create_data

optimizers = [opt.Adadelta, opt.Adagrad, opt.Adam]

def evaluate(probabilities):
    probabilities = tf.math.sigmoid(probabilities)
    loss = total_symmetry(probabilities[0])    
    return loss

def eval_model(nets, train_inputs, _, __, test_inputs, ___, hypers):  
    
    data = train_inputs['i0']
    training_iterations = 1000
    
    
    inp = Input(shape=data.shape[1:])
    out = nets['n0'].building(inp)
    out = Reshape([len(wanted_blocks)]+bounds)(out)
    model = Model(inputs=inp, outputs=out)
    
    #model.summary()
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)

    @tf.function
    def train_step(x_train):
        
        with tf.GradientTape() as tape:   
            probabilities = model(x_train, training=True)
            #print(f'Out blocks: {out_blocks}')
            loss = evaluate(probabilities)
            
        #print('Trainable variables: \n',model.trainable_variables)
        gradients_of_generator = tape.gradient(loss, model.trainable_variables)  
        #print(gradients_of_generator)
        opt.apply_gradients(zip(gradients_of_generator, model.trainable_variables))
        return loss
    
    for epoch in range(training_iterations):
        loss = train_step(data) 
        #if epoch % 100 == 0:            
            #print("Epoch {:03d}: Loss: {}".format(epoch, loss))
        
    #prediction = model.predict(data)
    #ev = evaluate_prediction(prediction, wanted_blocks, verbose)

    probabilities = model(test_inputs['i0'])
    loss = evaluate(probabilities)
    
    return loss.numpy(),


if __name__ == "__main__":
    
    bounds = [5,5,5]
    wanted_blocks = [23,45,64,22,33]

    data = create_data(bounds)
    
    e = Evolving(desc_list=[MLPDescriptor], x_trains=[data], y_trains=[data], 
                 x_tests=[data], y_tests=[data], evaluation=eval_model, 
                 batch_size=150, population=5, generations=10, 
                 n_inputs=[[data.shape[1:]]], n_outputs=[[len(wanted_blocks)]+bounds], 
                 cxp=0., mtp=1., hyperparameters={"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]}, 
                 batch_norm=True, dropout=True)
    a = e.evolve()

    print(a[-1])
