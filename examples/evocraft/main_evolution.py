import sys
sys.path.append('../..')
import tensorflow as tf

from deatf.evolution import Evolving
from deatf.network import MLPDescriptor

from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.models import Model
import tensorflow.keras.optimizers as opt

from interactive_loss import interactive_loss
from symmetry_loss import total_symmetry
from create_data import create_data

optimizers = [opt.Adadelta, opt.Adagrad, opt.Adam]

def evaluate(probabilities):
    probabilities = tf.math.sigmoid(probabilities)
    loss = total_symmetry(probabilities[0])    
    return loss

def eval_model(nets, train_inputs, _, __, iters, test_inputs, ___, hypers):  
    
    data = train_inputs['i0']
    
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
            loss = evaluate(probabilities)
            
        gradients_of_generator = tape.gradient(loss, model.trainable_variables)  
        opt.apply_gradients(zip(gradients_of_generator, model.trainable_variables))
        return loss
    
    for epoch in range(iters):
        loss = train_step(data)
        if epoch % 100 == 0:
            print("Epoch {:03d}: Loss: {}".format(epoch, loss))
        
    #loss = evaluate(probabilities)
    probabilities = model(data)
    ev = interactive_loss(probabilities, wanted_blocks, bounds)
   
    return ev,


if __name__ == "__main__":
    
    bounds = [9,9,9]
    wanted_blocks = [-1,164, 169, 173]

    data = create_data(bounds)
    
    e = Evolving(desc_list=[MLPDescriptor], x_trains=[data], y_trains=[data], 
                 x_tests=[data], y_tests=[data], evaluation=eval_model, 
                 batch_size=150, population=5, generations=10, iters=500, 
                 n_inputs=[[data.shape[1:]]], n_outputs=[[len(wanted_blocks)]+bounds], 
                 cxp=0., mtp=1., hyperparameters={"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]}, 
                 batch_norm=True, dropout=True)
    a = e.evolve()

    print(a[-1])
