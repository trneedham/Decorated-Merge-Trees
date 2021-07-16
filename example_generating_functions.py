import numpy as np

"""
This file contains functions for generating toy point cloud data
"""

def noisy_circle(n_samples, noise_level, center_x, center_y ,radius):

    t = np.linspace(0,2*np.pi,n_samples)
    x = center_x + radius*np.cos(t)
    y = center_y + radius*np.sin(t)
    noise = np.random.rand(n_samples,2)
    data = np.array([x,y]).T + noise_level*radius*noise

    return data

def noisy_disk(n_samples,noise_level,center_x,center_y,radius):

    t = np.linspace(0,2*np.pi,n_samples)
    r = radius*np.random.rand(n_samples)
    x = center_x + np.multiply(r,np.cos(t))
    y = center_y + np.multiply(r,np.sin(t))
    noise = np.random.rand(n_samples,2)
    data = np.array([x,y]).T + noise_level*radius*noise

    return data

def one_disk_two_circles(radii, separation, n_samples_per_shape = 50, noise_level = 0.5, centersy = None):

    """
    This example produces one disk and two circles or radii r1,r2,r3. The circles are clustered together and
    the disk is separated in the x-direction by `separation`
    In: triple of radii = [r1,r2,r3], separation distance
    """

    data = np.zeros([n_samples_per_shape*3,2])

    shapes = ['disk','circle','circle']
    r1 = radii[0]
    r2 = radii[1]
    r3 = radii[2]
    centersx = [0,r1+separation+r2,r1+separation+2*r2+r3]

    if centersy is None:
        centersy = [0,0,0]

    num_shapes = len(shapes)

    for j in range(num_shapes):
        if shapes[j] == 'disk':
            shape = noisy_disk(n_samples_per_shape,noise_level,centersx[j],centersy[j],radii[j])
        elif shapes[j] == 'circle':
            shape = noisy_circle(n_samples_per_shape, noise_level,centersx[j],centersy[j],radii[j])
        data[j*n_samples_per_shape:(j+1)*n_samples_per_shape,:] = shape

    return data

def two_circles(radii, separation, n_samples_per_shape = 50, noise_level = 0.5, centersy = None):

    """
    This example produces two and two circles or radii r1,r2.
    The circles are separated in the x-direction by `separation`
    In: triple of radii = [r1,r2], separation distance
    """

    data = np.zeros([n_samples_per_shape*2,2])

    shapes = ['circle','circle']
    r1 = radii[0]
    r2 = radii[1]

    centersx = [0,r1+separation+r2]

    if centersy is None:
        centersy = [0,0]

    num_shapes = len(shapes)

    for j in range(num_shapes):
        if shapes[j] == 'disk':
            shape = noisy_disk(n_samples_per_shape,noise_level,centersx[j],centersy[j],radii[j])
        elif shapes[j] == 'circle':
            shape = noisy_circle(n_samples_per_shape, noise_level,centersx[j],centersy[j],radii[j])
        data[j*n_samples_per_shape:(j+1)*n_samples_per_shape,:] = shape

    return data

def three_circles_two_clusters(radii, separation, n_samples_per_shape = 50, noise_level = 0.5, centersy = None):

    """
    This example produces two and two circles or radii r1,r2.
    The circles are separated in the x-direction by `separation`
    In: triple of radii = [r1,r2], separation distance
    """

    data = np.zeros([n_samples_per_shape*3,2])

    shapes = ['circle','circle','circle']
    r1 = radii[0]
    r2 = radii[1]
    r3 = radii[2]
    centersx = [0,r1+separation+r2,r1+separation+2*r2+r3]

    if centersy is None:
        centersy = [0,0,0]

    num_shapes = len(shapes)

    for j in range(num_shapes):
        if shapes[j] == 'disk':
            shape = noisy_disk(n_samples_per_shape,noise_level,centersx[j],centersy[j],radii[j])
        elif shapes[j] == 'circle':
            shape = noisy_circle(n_samples_per_shape, noise_level,centersx[j],centersy[j],radii[j])
        data[j*n_samples_per_shape:(j+1)*n_samples_per_shape,:] = shape

    return data

def three_circles(radii, separation1, separation2, n_samples_per_shape = 50, noise_level = 0.5, centersy = None):


    data = np.zeros([n_samples_per_shape*3,2])

    shapes = ['circle','circle','circle']
    r1 = radii[0]
    r2 = radii[1]
    r3 = radii[2]
    centersx = [0,r1+separation1+r2,r1+separation1+r2]

    if centersy is None:
        centersy = [0,0,r2+separation2+r3]

    num_shapes = len(shapes)

    for j in range(num_shapes):
        if shapes[j] == 'disk':
            shape = noisy_disk(n_samples_per_shape,noise_level,centersx[j],centersy[j],radii[j])
        elif shapes[j] == 'circle':
            shape = noisy_circle(n_samples_per_shape, noise_level,centersx[j],centersy[j],radii[j])
        data[j*n_samples_per_shape:(j+1)*n_samples_per_shape,:] = shape

    return data

def three_clusters_variable_circles(shapes, radii, separation1, separation2, n_samples_per_shape = 50, noise_level = 0.5, centersy = None):

    data = np.zeros([n_samples_per_shape*4,2])

    r1 = radii[0]
    r2 = radii[1]
    r3 = radii[2]
    r4 = radii[3]
    centersx = [0, r1+r2, r1+r2+separation1+r3, r1+r2+separation1+r3]

    if centersy is None:
        centersy = [0,0,0,r3+separation2+r4]

    num_shapes = len(shapes)

    for j in range(num_shapes):
        if shapes[j] == 'disk':
            shape = noisy_disk(n_samples_per_shape,noise_level,centersx[j],centersy[j],radii[j])
        elif shapes[j] == 'circle':
            shape = noisy_circle(n_samples_per_shape, noise_level,centersx[j],centersy[j],radii[j])
        data[j*n_samples_per_shape:(j+1)*n_samples_per_shape,:] = shape

    return data

def two_clusters_variable_circles(shapes, radii, separation1, n_samples_per_shape = 50, noise_level = 0.5, centersy = None):
    
    data = np.zeros([n_samples_per_shape*3,2])

    r1 = radii[0]
    r2 = radii[1]
    r3 = radii[2]
    centersx = [0, r1+r2, r1+r2+separation1+r3]

    if centersy is None:
        centersy = [0,0,0]

    num_shapes = len(shapes)

    for j in range(num_shapes):
        if shapes[j] == 'disk':
            shape = noisy_disk(n_samples_per_shape,noise_level,centersx[j],centersy[j],radii[j])
        elif shapes[j] == 'circle':
            shape = noisy_circle(n_samples_per_shape, noise_level,centersx[j],centersy[j],radii[j])
        data[j*n_samples_per_shape:(j+1)*n_samples_per_shape,:] = shape

    return data
