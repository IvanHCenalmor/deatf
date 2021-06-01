import tensorflow as tf

def load_fashion():
    """
    :return: Data of fashion mnist dataset divided in train, test and validation
             (x_train, y_train, x_test, y_test, x_val, y_val).
    :rtype x_train: uint8 NumPy array of grayscale image data with shapes (42000, 28, 28), 
                    containing the training data.    
    :rtype y_train: uint8 NumPy array of labels (integers in range 0-9) with shape (42000,) 
                    for the training data.    
    :rtype x_test: uint8 NumPy array of grayscale image data with shapes (10000, 28, 28), 
                   containing the test data.
    :rtype y_test: uint8 NumPy array of labels (integers in range 0-9) with shape (10000,) 
                   for the test data.
    :rtype x_val: uint8 NumPy array of grayscale image data with shapes (18000, 28, 28), 
                  containing the validation data.
    :rtype y_val: uint8 NumPy array of labels (integers in range 0-9) with shape (18000,) 
                  for the validation data.    
    """
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
    """
    :return: Data of fashion mnist dataset divided in train, test and validation
             (x_train, y_train, x_test, y_test, x_val, y_val).
    :rtype x_train: uint8 NumPy array of grayscale image data with shapes (42000, 28, 28), 
                    containing the training data. Pixel values range from 0 to 255.   
    :rtype y_train: uint8 NumPy array of labels (integers in range 0-9) with shape (42000,) 
                    for the training data.    
    :rtype x_test: uint8 NumPy array of grayscale image data with shapes (10000, 28, 28), 
                   containing the test data. Pixel values range from 0 to 255.
    :rtype y_test: uint8 NumPy array of labels (integers in range 0-9) with shape (10000,)
                   for the test data.
    :rtype x_val: uint8 NumPy array of grayscale image data with shapes (18000, 28, 28), 
                  containing the validation data. Pixel values range from 0 to 255.
    :rtype y_val: uint8 NumPy array of labels (integers in range 0-9) with shape (18000,) 
                  for the validation data.    
    """
    
    mnist = tf.keras.datasets.mnist.load_data()
 
    dividing_indx = int(mnist[0][0].shape[0] * 0.7)
    
    x_train = mnist[0][0][:dividing_indx]
    y_train = mnist[0][1][:dividing_indx]

    x_val = mnist[0][0][dividing_indx:]
    y_val = mnist[0][1][dividing_indx:]
    
    x_test = mnist[1][0]
    y_test = mnist[1][1]

    return x_train, y_train, x_test, y_test, x_val, y_val