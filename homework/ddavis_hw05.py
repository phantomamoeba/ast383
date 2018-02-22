import os

import numpy as np
import tensorflow as tf


#fixed seed for now so runs are repeatable (testable)
SEED = 1138
np.random.seed(SEED)

poisson_avg_photons_per_pix = 3
poisson_avg_value_per_photon = 0.1

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

    return image.reshape(1,image.shape[0],image.shape[1],1).astype('float32'), points


def make_psf(dim_x, dim_y):
    psf = np.random.random((dim_x, dim_y))
    psf /= np.sum(psf)
    return psf.reshape(psf.shape[0],psf.shape[1],1,1).astype('float32')


def build_observation(true_image, true_psf):
    ph_image = tf.placeholder(dtype=tf.float32, shape=true_image.shape)
    ph_psf = tf.placeholder(dtype=tf.float32, shape=true_psf.shape)

    out = None

    with tf.Session() as sess:
        # convolve with the psf
        convolved_image = tf.nn.convolution(ph_image, ph_psf, "SAME")  # convolve(ph_image,ph_psf)

        # add poisson noise
        convolved_image += poisson_avg_value_per_photon * \
                           tf.random_poisson(lam=poisson_avg_photons_per_pix, shape=true_image.shape, seed=SEED)

        # run the graph
        init = tf.global_variables_initializer()
        sess.run(init)
        out = sess.run(convolved_image, feed_dict={ph_image: true_image, ph_psf: true_psf})

        #print(observation)

    return out

def main():

    #start with the true data
    true_image, true_points = make_true_image(10,10,3)
    true_psf = make_psf(3,3)

    #step 5
    observation = build_observation(true_image,true_psf)

    ph_obs = tf.placeholder(dtype=tf.float32, shape=observation.shape)
    ph_psf = tf.placeholder(dtype=tf.float32, shape=true_psf.shape)

    var_model = tf.get_variable(name='model',
                                shape=observation.shape,
                                dtype=tf.float32,
                                trainable=True,
                                constraint=lambda x: tf.clip_by_value(x, 0, np.inf), #force positive only
                                initializer=None,
                                regularizer= tf.contrib.layers.l1_regularizer(scale=1.0) #todo: make scale tunable
                                )



    with tf.Session() as sess:

        #convolve with the true PSF
        var_model = tf.nn.convolution(var_model, ph_psf, "SAME")

        obj = tf.nn.l2_loss(ph_obs - var_model) #not adding a bias term (yet)
        optimizer = tf.train.AdamOptimizer().minimize(obj) #use defaults for now



        # run the graph
        init = tf.global_variables_initializer()
        sess.run(init)

        out = sess.run(optimizer,feed_dict={ph_obs: observation,ph_psf:true_psf})



        print(out,sess.graph.get_tensor_by_name('model:0').eval())



   # print(observation)

if __name__ == '__main__':
    main()