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

def make_image(dim_x, dim_y, num_points):
    # force no adjacent point sources
    #todo: sanity check num_points vs dim_x, y s|t the configuration is possible
    #a bit simplistic, but sufficient and safe (each point claims the 3x3 elements centered on it)
    assert(dim_x*dim_y > num_points*9)

    image = np.zeros((dim_x,dim_y))
    points = []

    #todo: if relax the restriction on dimensions and num of points, could add a safety here
    #todo: to detect no-solution cases and restart
    for _ in range(num_points):
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
    #test with no psf
    #psf = np.zeros((dim_x,dim_y))
    return psf.reshape(psf.shape[0],psf.shape[1],1,1).astype('float32')


def build_observation(image, psf):
    out = None #will be the convolution of the image and psf

    #build the graph (use named scope to prevent any collisions)
    with tf.variable_scope("observation"):
        #inputs
        ph_image = tf.placeholder(dtype=tf.float32, shape=image.shape)
        ph_psf = tf.placeholder(dtype=tf.float32, shape=psf.shape)

        # convolve with the psf
        convolved_image = tf.nn.convolution(ph_image, ph_psf, "SAME")  # convolve(ph_image,ph_psf)

        # add poisson noise
        convolved_image += poisson_avg_value_per_photon * \
                           tf.random_poisson(lam=poisson_avg_photons_per_pix, shape=image.shape, seed=SEED)

        with tf.Session() as sess:
            # run the graph
            sess.run(tf.global_variables_initializer())
            out = sess.run(convolved_image, feed_dict={ph_image: image, ph_psf: psf})

            #print("Observation:\n ", out)

    return out


def fit_model(observation,psf,iterations=1000,learning_rate=0.01, l1_reg_scale=0.0001):
    # step 5
    best_model = None
    with tf.variable_scope("fit"):
        ph_obs = tf.placeholder(dtype=tf.float32, shape=observation.shape)
        ph_psf = tf.placeholder(dtype=tf.float32, shape=psf.shape)
        # ph_test = tf.placeholder(dtype=tf.float32, shape=observation.shape)

        test_image, test_points = make_image(observation.shape[1], observation.shape[2], 0) #start all zeroes

        var_model = tf.get_variable(name='model',
                                    # shape=observation.shape,
                                    dtype=tf.float32,
                                    trainable=True,
                                    constraint=lambda x: tf.clip_by_value(x, 0, np.inf),  # force positive only
                                    #initializer=build_observation(test_image, psf),
                                    initializer=test_image,
                                    regularizer=tf.contrib.layers.l1_regularizer(scale=l1_reg_scale)  # todo: make scale tunable
                                    )

        var_convolved = tf.get_variable(name='convolved_model',
                                    shape=observation.shape,
                                    dtype=tf.float32,
                                    trainable=False,
                                    constraint=None,  # force positive only
                                    # initializer=build_observation(test_image, psf),
                                    initializer=None,
                                    regularizer=None
                                    # todo: make scale tunable
                                    )

        #regularizer = tf.contrib.layers.l1_regularizer(scale=l1_reg_scale)
       # weights = tf.trainable_variables()
       # l1_reg_penalty = tf.contrib.layers.apply_regularization(tf.contrib.layers.l1_regularizer(scale=l1_reg_scale),weights)
        #var_convolved = tf.nn.convolution(var_model, ph_psf, "SAME")#build_observation(var_model,psf)
        loss = tf.nn.l2_loss(var_model - ph_obs) + sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) # not adding a bias term (yet)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.minimize(loss)  # use defaults for now

        # could have just done train = tf.train.AdamOptimizer(0.5).minimize(loss), but want to be more explicitly clear

        with tf.Session() as sess:
            # run the graph
            sess.run(tf.global_variables_initializer())

            #todo: make the iteration length variable (base on input size or stabilization of loss?)
            for _ in range(iterations):
                sess.run(train, feed_dict={ph_obs: observation, ph_psf: psf})

                # see how we are progressing
                print(sess.run(loss, feed_dict={ph_obs: observation, ph_psf: psf}))
                # print(sess.graph.get_tensor_by_name('fit/model:0').eval())

            best_model = sess.run(var_model)

            print("test",best_model )

    return best_model


def chi_sqr(obs, exp, error=None):

    obs = np.array(obs).flatten()
    exp = np.array(exp).flatten()

    if error is not None:
        error = np.array(error)

    #trim non-values
    if 0:
        remove = []
        for i in range(len(obs)):
            if obs[i] > 98.0:
                remove.append(i)

        obs = np.delete(obs, remove)
        exp = np.delete(exp, remove)
        if error is not None:
            error = np.delete(error, remove)

    if error is not None:
        c = np.sum((obs*exp)/(error*error)) / np.sum((exp*exp)/(error*error))
    else:
        c = 1.0

    chisqr = 0
    if error is None:
        error=np.zeros(len(obs))
        error += 1.0

    for i in range(len(obs)):
        chisqr = chisqr + ((obs[i]-c*exp[i])**2)/(error[i])

    return chisqr,c

def fidelity(true_image,model_image):
    #say, chi2 over all pixels??
    assert true_image.shape == model_image.shape

    chi2, c = chi_sqr(true_image,model_image)
    return chi2

def main():

    #start with the true data
    true_image, true_points = make_image(100,100,30)
    true_psf = make_psf(3,3)

    observation = build_observation(true_image, true_psf)
    model = fit_model(observation,true_psf)

    print("true", observation)

    print("\n\ndiff", observation-model)

    print("chi2 between truth and model",fidelity(observation,model))

   # print(observation)

if __name__ == '__main__':
    main()