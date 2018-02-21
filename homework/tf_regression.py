import os

import numpy as np
import tensorflow as tf

__all__ = ['linear_regression']


def linear_regression(x, y, gamma_w, gamma_b, minibatch_size, learning_rate,
                      training_epochs, architecture, method):
    assert x.shape[0] == y.shape[0]
    assert architecture in ['neural', 'explicit']
    assert method in ['sgd', 'adam']

    x_dim = x.shape[1]
    y_dim = y.shape[1]

    #placeholder ... something that will later be used as an input to the graph
    #shape[None,x_dim] ... # of rows not specified ... will later supply one row or several or all
    #   with SGD normally feed in some subset
    x_tf = tf.placeholder(dtype=tf.float32, shape=[None, x_dim])
    y_tf = tf.placeholder(dtype=tf.float32, shape=[None, y_dim])

    regularizer = tf.contrib.layers.l2_regularizer
    initializer = tf.contrib.layers.xavier_initializer() #random initial guesses for w and b ... but with small dispersion

    #random initial guesses for w and b ... but with small dispersion
#    initializer = tf.glorot_uniform_initializer()

    if architecture == 'neural':
        with tf.variable_scope('neural') as scope:

            #dense ... every input is connected to outputs? ... fully connected or dense layer
            #same, mathematically, as matrix multiplication (where each edge is an element in the output matrix)

            y_net = tf.layers.dense(inputs=x_tf, units=y_dim,
                                    kernel_initializer=initializer,
                                    bias_initializer=initializer,
                                    name='dense') #here 'dense' is just the name or label

            #convolution layers are NOT dense ... have some constraint that some edges must be the same
            # and there is a finite extent s|t as nodes get farther apart, they are NOT connected


            #TRAINABLE ... eg. fitable like w and b
            def apply_regularization(param, gamma):
                reg_vars = [var for var in
                            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              scope=scope.name)
                            if param in var.name]
                return tf.contrib.layers.apply_regularization(regularizer=regularizer(scale=gamma),
                                                              weights_list=reg_vars)

            obj = tf.nn.l2_loss(y_net - y_tf)
            obj += apply_regularization('kernel', gamma_w)
            obj += apply_regularization('bias', gamma_b)

            def w_extractor(sess): return sess.graph.get_tensor_by_name('neural/dense/kernel:0').eval().T

            def b_extractor(sess): return sess.graph.get_tensor_by_name('neural/dense/bias:0').eval()

    elif architecture == 'explicit':
        with tf.variable_scope('explicit'):

            def create_variable(name, shape, gamma):
                return tf.get_variable(name=name,
                                       shape=shape,
                                       dtype=tf.float32,
                                       initializer=initializer,
                                       regularizer=regularizer(scale=gamma))

            w = create_variable(name='weights', shape=[y_dim, x_dim], gamma=gamma_w)
            b = create_variable(name='biases', shape=[y_dim], gamma=gamma_b)

            #this is NOT an assignment ... more of a declaration ... is a part of the graph
            y_exp = tf.matmul(x_tf, tf.transpose(w)) + b

            #the objective
            obj = tf.nn.l2_loss(y_exp - y_tf) + sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            def w_extractor(sess): return sess.run(w)

            def b_extractor(sess): return sess.run(b)

    else:
        obj = w_extractor = b_extractor = None

    optimizer = {'sgd': tf.train.GradientDescentOptimizer, 'adam': tf.train.AdamOptimizer}
    optimizer = optimizer[method]
    optimizer = optimizer(learning_rate).minimize(obj)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init_op)

        #each epoch is fitting one mini-batch
        for epoch in range(training_epochs):

            def minibatch_dict():
                count = x.shape[0]
                #select a random mini-batch
                minibatch = np.random.choice(count, size=minibatch_size, replace=False)
                return {x_tf: x[minibatch], y_tf: y[minibatch]}

            #each of these is different (called the function twice) but might have some overlap
            train_dict = minibatch_dict()
            test_dict = minibatch_dict()

            sess.run(optimizer, feed_dict=train_dict)

            def is_power2():
                return not epoch & (epoch - 1)

            #feed_dictionary is how to feed data in tf into placeholders
            #keys are placeholders
            #values are minibatches (matrices)
            if is_power2():
                #see if objective is decreasingg
                obj_val = sess.run(obj, feed_dict=test_dict)
                print('epoch {} obj = {}'.format(epoch, obj_val))

        w_fit = w_extractor(sess)
        b_fit = b_extractor(sess)

    return w_fit, b_fit


def main(argv):
    assert (len(argv) == 1)

    x_dim = 4
    y_dim = 8
    nb_obs = 128

    w_true = np.random.randn(y_dim, x_dim)
    b_true = np.random.randn(y_dim)

    x = np.random.randn(nb_obs, x_dim)
    y = x @ w_true.T + b_true  #broadcasting (add vector to every row), @ is matrix mult. (python 3.5+)

    #want to recover w_true and b_true
    #gamma_w and _b are regularizers (could have different values)
    w_fit, b_fit = linear_regression(x, y, gamma_w=1e-4, gamma_b=1e-4,
                                     minibatch_size=16,
                                     learning_rate=1e-1,
                                     training_epochs=1000,
                                     architecture='explicit',
                                     method='adam')

    #tf can also calculate (don't have to use numpy)
    def error(a_fit, a_true):
        return np.max(np.absolute(a_fit - a_true) /
                      (0.5 * (np.absolute(a_fit) + np.absolute(a_true))))

    b_error = error(b_fit, b_true)
    w_error = error(w_fit, w_true)

    print('maximum relative error in b = {}'.format(b_error))
    print('maximum relative error in w = {}'.format(w_error))

    #produces the graph (need to install TensorBoard to view this)
    with tf.Session() as sess:
        _ = tf.summary.FileWriter(os.getcwd(), sess.graph)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
