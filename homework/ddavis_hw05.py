import os

import numpy as np
import tensorflow as tf


#fixed seed for now so runs are repeatable (testable)
np.random.seed(1138)

def pix_dist(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def make_true_image(dim_x, dim_y, num_points):
    # force no adjacent point sources
    #todo: sanity check num_points vs dim_x, y s|t the configuration is possible
    #a bit simplistic, but sufficient and safe (each point claims the 3x3 elements centered on it)
    assert(dim_x*dim_y > num_points*9)

    image = np.zeros((dim_x,dim_y))
    points = []

    #todo: if relax the restriction on dimensions and num of points, could add a safety here
    #todo: to detect no-solution cases and restart
    for i in range(num_points):
        x = np.random.choice(range(dim_x))
        y = np.random.choice(range(dim_y))

        okay = True
        for (xp,yp) in points:
            if pix_dist(x,y,xp,yp) < 2.:
                okay = False
                break

        if okay:
            points.append((x,y)) #for tracking
            image[x,y] = 1

    return image, points


def make_psf(dim_x, dim_y):
    psf = np.random.random((dim_x, dim_y))
    psf /= np.sum(psf)
    return psf

def convolve(image,kernel):
    #for now, at least, image is 2D with one channel

    # num_batches, x,y, input channels
    # if RGB would be 3 input channles?
    # if passing in multiple 2D matrices, num_batches would equal that number?
    x = image.reshape(1,image.shape[0],image.shape[1],1).astype('float32')

    # x,y,num input channels, num output channels ... inputs need to match vs the true_image
    w = kernel.reshape(kernel.shape[0],kernel.shape[1],1,1).astype('float32')
    return tf.nn.convolution(x, w, "SAME")



def main():

    true_image, true_points = make_true_image(10,10,3)
    true_psf = make_psf(3,3)

    #convolved_image = tf.nn.convolution(true_image,true_psf,"SAME")
    convolved_image = convolve(true_image,true_psf)

    print(convolved_image)


if __name__ == '__main__':
    main()