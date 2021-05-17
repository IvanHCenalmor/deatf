import tensorflow as tf

def symmetry_loss(objects, available_objects):
    symmetry = total_symmetry(objects)
    penalty = variance_pennalty(objects, len(available_objects))
    loss = 0.7*symmetry + 0.3*penalty
    return loss

def variance_pennalty(objects, num_total_objects):
    
    used_objects = tf.reshape(objects, shape=(-1,))
    used_objects, _ = tf.unique(used_objects)
    num_used_objects = len(used_objects)
    
    assert num_used_objects <= num_total_objects, 'Something has gone wrong in variance penalty.'
    if num_used_objects < num_total_objects*0.3:
        return 0.  
    else:
        return num_used_objects/num_total_objects

def total_symmetry(objects):
    num_axis = len(objects.shape)
    sym = 0
    for ax in range(num_axis):
        sym += symmetry(objects, ax)
        #print(f'Axis: {ax} \n Symmetry: {sym}')
    sym /= num_axis
    #print(f'Total symettry: {sym}')
    return sym

def symmetry(objects, axis):
    index = int(objects.shape[axis]/2)
    #print([index, objects.shape[axis]-index, -1])
    #print(objects)
    left, middle, right = tf.split(objects, [index, objects.shape[axis]-2*index, -1], axis=axis)
    #print('Left', left.shape, ' : \n', left)
    #print('Right', right.shape, ' : \n', right)
    right = tf.reverse(right,axis=[axis])
    #print('Reversed Right', right.shape, ' : \n', right)
    difference = tf.keras.losses.categorical_crossentropy(left, right)
    #print(difference)
    result = tf.reduce_mean(difference)
    #print(f'Result: {result}')
    return result