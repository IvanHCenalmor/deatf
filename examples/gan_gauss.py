"""
This is a use case of EvoFlow

Another example of a multinetwork model, a GAN. In order to give an automatic fitness fuction to each GAN, we use the Inception Score (IS, https://arxiv.org/pdf/1606.03498.pdf)
We use the MobileNet model instead of Inception because it gave better accuracy scores when training it.
"""
import sys
sys.path.append('..')

import tensorflow as tf
import numpy as np
import sys

from evoflow.auxiliary_functions import batch
from evoflow.evolution import Evolving
from evoflow.network import MLPDescriptor

from evoflow.gaussians import mmd, plt_center_assignation, create_data
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

best_mmd = 28888888
eval_tot = 0

def generator_loss(fake_out):
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_out, labels=tf.ones_like(fake_out)))

    return g_loss

def discriminator_loss(fake_out, real_out):
    d_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=real_out, labels=tf.ones_like(real_out)) +
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_out, labels=tf.zeros_like(fake_out)))
    
    return d_loss


def gan_eval(nets, train_inputs, _, batch_size, __, test_outputs, ___):

    models = {}
    
    g_inp = Input(shape=z_size)
    g_out = nets["n1"].building(g_inp)
    g_out = tf.sigmoid(g_out)
    
    g_model = Model(inputs=g_inp, outputs=g_out)

    d_inp = Input(shape=2)
    d_out = nets["n0"].building(d_inp)
    d_out = tf.sigmoid(d_out)
    
    d_model = Model(inputs=d_inp, outputs=d_out)

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    @tf.function
    def train_step(x_train):
        
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:

            z_mb = np.random.uniform(size=(batch_size, z_size))
            generated_images = g_model(z_mb, training=True)
            z_mb = np.random.uniform(size=(batch_size, z_size))
            generated_images = g_model(z_mb, training=True)
            z_mb = np.random.uniform(size=(batch_size, z_size))
            generated_images = g_model(z_mb, training=True)
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

        x_mb = batch(train_inputs["i0"], batch_size, aux_ind)
        aux_ind = (aux_ind + batch_size) % train_inputs["i0"].shape[0]
        train_step(x_mb)
        
    models['n0'] = d_model
    models['n1'] = g_model
        
    global best_mmd
    global eval_tot
    
    noise = np.random.uniform(size=(n_samples, z_size))
    samples = models["n1"](noise)

    mmd_value, centers = mmd(candidate=samples, target=test_outputs["o0"])
    print(mmd_value)
    if mmd_value < best_mmd:
        best_mmd = mmd_value
        plt.plot(test_outputs["o0"][:, 0], test_outputs["o0"][:, 1], "o")
        plt.plot(samples[:, 0], samples[:, 1], "o")
        plt.savefig("gaussian_results/Evoflow_" + str(n_gauss) + "_" + str(seed) + "_" + str(eval_tot) + "_" + str(np.round(mmd_value, decimals=3)) + ".jpg")
        plt.clf()
        np.save("gaussian_results/Samples_" + str(n_gauss) + "_" + str(seed) + "_" + str(mmd_value), samples)
    eval_tot += 1

    return mmd_value,


if __name__ == "__main__":

    args = sys.argv[1:]
    seed = int(args[0])
    n_gauss = int(args[1])
    n_samples = int(args[2])
    population = int(args[3])
    generations = int(args[4])
    epochs = int(args[5])
    z_size = int(args[6])
    
    x_train = create_data(n_gauss, n_samples)
    x_train = x_train - np.min(x_train, axis=0)
    x_train = x_train / np.max(x_train, axis=0)

    x_test = create_data(n_gauss, n_samples)
    x_test = x_test - np.min(x_test, axis=0)
    x_test = x_test / np.max(x_test, axis=0)

    x_val = create_data(n_gauss, n_samples)
    x_val = x_val - np.min(x_val, axis=0)
    x_val = x_val / np.max(x_val, axis=0)

    # The GAN evolutive process is a common 2-DNN evolution
    e = Evolving(desc_list=[MLPDescriptor, MLPDescriptor], 
                 x_trains=[x_train], y_trains=[x_train], x_tests=[x_val], y_tests=[x_val], 
                 evaluation=gan_eval, batch_size=50, population=population, generations=generations, 
                 n_inputs=[[2], [z_size]], n_outputs=[[1], [2]], 
                 cxp=0.5, mtp=0.5, dropout=False, batch_norm=False)
    res = e.evolve()

    print(res[0])
