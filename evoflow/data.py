import tensorflow as tf

def load_fashion():

    fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()

    dividing_indx = int(fashion_mnist[0][0].shape[0] * 0.7)
    
    x_train = fashion_mnist[0][0][:dividing_indx]
    y_train = fashion_mnist[0][1][:dividing_indx]

    x_val = fashion_mnist[0][0][dividing_indx:]
    y_val = fashion_mnist[0][1][dividing_indx:]

    x_test = fashion_mnist[1][0]
    y_test = fashion_mnist[1][1]

    return x_train, y_train, x_test, y_test, x_val, y_val

def load_mnist():

    mnist = tf.keras.datasets.mnist.load_data()
 
    dividing_indx = int(mnist[0][0].shape[0] * 0.7)
    
    x_train = mnist[0][0][:dividing_indx]
    y_train = mnist[0][1][:dividing_indx]

    x_val = mnist[0][0][dividing_indx:]
    y_val = mnist[0][1][dividing_indx:]
    
    x_test = mnist[1][0]
    y_test = mnist[1][1]

    return x_train, y_train, x_test, y_test, x_val, y_val