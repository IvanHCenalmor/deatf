import numpy as np
import math

def create_data(bounds, size):
    if len(bounds) == 2:
        return create_2d_data(bounds, size)
    elif len(bounds) == 3:
        return create_3d_data(bounds, size)
    else:
        raise ValueError('\'bounds\' is not well defined.')

def create_2d_data(bounds, size):

    data = []
    for _ in range(size):
        configuration = []
        for i in range(bounds[0]):
            for j in range(bounds[1]):
                position_norm = [-1 + 2 * i/bounds[0], -1 + 2*j/bounds[1]]
                dist = np.linalg.norm(position_norm)
                
                position_norm.append(dist)
                configuration.append(position_norm)
        configuration = np.array(configuration).flatten()
        configuration = np.array([configuration])
        configuration = configuration*(0.5 + np.random.rand())
        data.append(configuration)
        
    data = np.array(data)
    return data

def create_3d_data(bounds, size):

    data = []
    for _ in range(size):
        configuration = []
        for i in range(bounds[0]):
            for j in range(bounds[1]):
                for z in range(bounds[2]):
                    position_norm = [-1 + 2 * i/bounds[0], -1 + 2*j/bounds[1], -1 + 2*z/bounds[2]]
                    dist = np.linalg.norm(position_norm)
                    
                    position_norm.append(dist)
                    configuration.append(position_norm)
        configuration = np.array(configuration).flatten()
        configuration = np.array([configuration])
        configuration = configuration*(0.5 + np.random.rand())
        data.append(configuration)

    data = np.array(data)
    return data

x = create_data([5,5,5], 100)

"""        
def create_2d_data(bounds):
    radial_bound = False
    max_radius = 0.9
    input_symmetry = 'sin'

    data = []
    for i in range(bounds[0]):
        for j in range(bounds[1]):
            position_norm = [-1 + 2 * i/bounds[0], -1 + 2*j/bounds[1]]
            dist = np.linalg.norm(position_norm)
            
            # Limit to a radius if decide so
            if (radial_bound and dist < max_radius) or not radial_bound:
                # only for x and z HERE ! but that may slow down..
                if input_symmetry == 'sin':
                    position_norm = [np.sin(math.pi*position_norm[0]), position_norm[1]]
                elif input_symmetry == 'abs':
                    position_norm = np.abs(position_norm)
                position_norm.append(dist)
                data.append(position_norm)
    data = np.array(data).flatten()
    data = np.array([data])

    return data

def create_3d_data(bounds, size):
    radial_bound = False
    max_radius = 0.9
    input_symmetry = 'sin'

    data = []
    for i in range(bounds[0]):
        for j in range(bounds[1]):
            for z in range(bounds[2]):
                position_norm = [-1 + 2 * i/bounds[0], -1 + 2*j/bounds[1], -1 + 2*z/bounds[2]]
                dist = np.linalg.norm(position_norm)
                
                # Limit to a radius if decide so
                if (radial_bound and dist < max_radius) or not radial_bound:
                    # only for x and z HERE ! but that may slow down..
                    if input_symmetry == 'sin':
                        position_norm = [np.sin(math.pi*position_norm[0]), position_norm[1]]
                    elif input_symmetry == 'abs':
                        position_norm = np.abs(position_norm)
                    position_norm.append(dist)
                    data.append(position_norm)
    data = np.array(data).flatten()
    data = np.array([data])

    return data
"""