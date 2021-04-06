"""
This is a use case of EvoFlow

Another example of a multinetwork model, a GAN. In order to give an automatic fitness fuction to each GAN, we use the Inception Score (IS, https://arxiv.org/pdf/1606.03498.pdf)
We use the MobileNet model instead of Inception because it gave better accuracy scores when training it.
"""
from data import load_fashion
import tensorflow as tf
import numpy as np
from keras.models import model_from_json
from evolution import Evolving, batch
from Network import MLPDescriptor
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model

def generator_loss(fake_out):
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_out,
                                                                    labels=tf.ones_like(fake_out)))
    return g_loss

def discriminator_loss(fake_out, real_out):
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_out, 
                                                                         labels=tf.ones_like(real_out)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_out,
                                                                         labels=tf.zeros_like(fake_out)))
    
    d_loss = d_loss_real + d_loss_fake
    return d_loss

def gan_train(nets, train_inputs, _, batch_size, __):
    
    models = {}
        
    noise = np.random.normal(size=(150, 10))
    
    g_inp = Input(shape=noise.shape[1:])
    g_out = nets["n1"].building(g_inp)
    g_out = Reshape(x_train.shape[1:])(g_out)
    
    g_model = Model(inputs=g_inp, outputs=g_out)

    d_inp = Input(shape=x_train.shape[1:])
    d_out = Flatten()(d_inp)
    d_out = nets["n0"].building(d_out)
    
    d_model = Model(inputs=d_inp, outputs=d_out)

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    @tf.function
    def train_step(x_train):
        
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
    
            generated_images = g_model(noise, training=True)
            fake_out = d_model(generated_images, training=True)
            
            real_out = d_model(x_train, training=True)
            
            g_loss = generator_loss(fake_out)
            d_loss = discriminator_loss(fake_out, real_out)
            
        gradients_of_generator = g_tape.gradient(g_loss, g_model.trainable_variables)
        gradients_of_discriminator = d_tape.gradient(d_loss, d_model.trainable_variables)
    
        generator_optimizer.apply_gradients(zip(gradients_of_generator, g_model.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, d_model.trainable_variables))
    
    aux_ind = 0        
    
    for epoch in range(100):

        image_batch = batch(train_inputs["i0"], batch_size, aux_ind)
        aux_ind = (aux_ind + batch_size) % train_inputs["i0"].shape[0]
        train_step(image_batch)
    

    models['n0'] = d_model
    models['n1'] = g_model
    
    return models

def gan_eval(models, _, __, ___):

    noise = np.random.normal(size=(150, 10))
    
    generated_images = models['n1'](noise, training=False)
    
    return generator_loss(generated_images).numpy(),
    
if __name__ == "__main__":

    x_train, _, x_test, _ = load_fashion()
    # The GAN evolutive process is a common 2-DNN evolution
    e = Evolving(loss=gan_train, desc_list=[MLPDescriptor, MLPDescriptor], 
                 x_trains=[x_train], y_trains=[x_train], 
                 x_tests=[x_test], y_tests=[x_test], 
                 evaluation=gan_eval, batch_size=150, 
                 population=10, generations=10, 
                 n_inputs=[[28, 28], [10]], n_outputs=[[1], [784]], 
                 cxp=0.5, mtp=0.5)
    res = e.evolve()

    print(res[0])
