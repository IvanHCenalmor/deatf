"""
This is a use case of EvoFlow
"""
from data import load_fashion
import tensorflow as tf
from evolution import Evolving, accuracy_error
from Network import RNNDescriptor

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

import tensorflow.keras.optimizers as opt


optimizers = [opt.Adadelta, opt.Adagrad, opt.Adam]


def eval_rnn(nets, train_inputs, train_outputs, batch_size, test_inputs, test_outputs, hypers):
    models = {}

    inp = Input(shape=train_inputs["i0"].shape[1:])
    out = nets["n0"].building(inp)
    model = Model(inputs=inp, outputs=out)
    #model.summary()
    opt = optimizers[hypers["optimizer"]](learning_rate=hypers["lrate"])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=[])
    
    # As the output has to be the same as the input, the input is passed twice
    model.fit(train_inputs['i0'], train_outputs['o0'], epochs=10, batch_size=batch_size, verbose=0)
            
    models["n0"] = model
                     
    preds = models["n0"].predict(test_inputs["i0"])
    
    res = tf.nn.softmax(preds)

    return accuracy_error(res, test_outputs["o0"]),

if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_fashion()

    # Normalize the input data
    x_train = x_train/255
    x_test = x_test/255

    '''
    OHEnc = OneHotEncoder()
    y_train = OHEnc.fit_transform(np.reshape(y_train, (-1, 1))).toarray()
    y_test = OHEnc.fit_transform(np.reshape(y_test, (-1, 1))).toarray()
    '''
    
    # Here we define a convolutional-transposed convolutional network combination
    e = Evolving(evaluation=eval_rnn, 
                 desc_list=[RNNDescriptor], 
                 x_trains=[x_train], y_trains=[y_train], 
                 x_tests=[x_test], y_tests=[y_test], 
                 batch_size=150, population=2, generations=10, 
                 n_inputs=[[28, 28]], n_outputs=[[10]], 
                 cxp=0, mtp=1, 
                 hyperparameters = {"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]}, 
                 batch_norm=True, dropout=True,
                 ev_alg='mu_plus_lambda')

    a = e.evolve()

    print(a[-1])
