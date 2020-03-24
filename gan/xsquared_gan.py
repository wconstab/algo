"""
Following:
https://blog.paperspace.com/implementing-gans-in-tensorflow/

"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from plot_util import BokehPlotter, CostPlotHandler, PeriodicDataframeHandler

def get_y(x):
    return 10 + x*x

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def sample_data(n=10000, scale=100):
    data = []

    x = scale*(np.random.random_sample((n,))-0.5)

    for i in range(n):
        yi = get_y(x[i])
        data.append([x[i], yi])

    return np.array(data)

def generator(Z,hsize=[16, 16],reuse=False):
    with tf.compat.v1.variable_scope("GAN/Generator",reuse=reuse):
        h1 = tf.compat.v1.layers.dense(Z,hsize[0],activation=tf.nn.leaky_relu)
        h2 = tf.compat.v1.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
        out = tf.compat.v1.layers.dense(h2,2)

    return out

def discriminator(X,hsize=[16, 16],reuse=False):
    with tf.compat.v1.variable_scope("GAN/Discriminator",reuse=reuse):
        h1 = tf.compat.v1.layers.dense(X,hsize[0],activation=tf.nn.leaky_relu)
        h2 = tf.compat.v1.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
        h3 = tf.compat.v1.layers.dense(h2,2)
        out = tf.compat.v1.layers.dense(h3,1)

    return out, h3

def train():
    tf.compat.v1.disable_eager_execution()

    X = tf.compat.v1.placeholder(tf.float32,[None,2])
    Z = tf.compat.v1.placeholder(tf.float32,[None,2])

    G_sample = generator(Z)
    r_logits, r_rep = discriminator(X)
    f_logits, g_rep = discriminator(G_sample,reuse=True)

    disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits,labels=tf.ones_like(r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.zeros_like(f_logits)))
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.ones_like(f_logits)))

    gen_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
    disc_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

    gen_step = tf.compat.v1.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss,var_list = gen_vars) # G Train step
    disc_step = tf.compat.v1.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss,var_list = disc_vars) # D Train step

    batch_size = 64
    sess = tf.compat.v1.Session()
    tf.compat.v1.global_variables_initializer().run(session=sess)

    for i in range(100001):
        X_batch = sample_data(n=batch_size)
        Z_batch = sample_Z(batch_size, 2)
        _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})
        _, gloss, gsamp = sess.run([gen_step, gen_loss, G_sample], feed_dict={Z: Z_batch})
        if i % 500 == 0:
            scatter_handler.add_data(dict(x=gsamp[:,0], y=gsamp[:,1]))
        if i % 100 == 0:
            cost_handler.add_data(dict(iter=i, generator=[gloss], discriminator=[dloss]))
        # print("Iterations: {}\t Discriminator loss: {}\t Generator loss: {}".format(i,dloss,gloss))

if __name__ == "__main__":
    try:
        cost_handler = CostPlotHandler(costs=['generator', 'discriminator'])
        scatter_handler = PeriodicDataframeHandler()

        plotter = BokehPlotter([cost_handler, scatter_handler])
        plotter.start()
        train()
        plotter.stop()
    except KeyboardInterrupt:
        plotter.stop()
        print("Exit")