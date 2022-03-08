import os
# import json
# import time
# import math
import numpy as np
# import pandas as pd
import tensorflow as tf
from scipy.stats import sem
import tensorflow.keras as tfk
import tensorflow_probability as tfp
import tensorflow.keras.layers as tfkl
tfd,tfpl = tfp.distributions,tfp.layers
import tensorflow.keras.backend as tfkb

activation_global = 'elu'
nh_global = 3
h_global = 100

def fc_net(input_shape, layers, out_layers = [], activation = activation_global, lamba = 1e-4):
    net = tfk.Sequential([tfkl.InputLayer([input_shape])])
    for hidden in layers:
        net.add(tfkl.Dense(
            hidden, 
            activation = activation,
            kernel_regularizer = tf.keras.regularizers.l2(lamba),
            kernel_initializer='RandomNormal',
            )
        )
    if len(out_layers) > 0:
        [outdim, activation_out] = out_layers
        net.add(tfkl.Dense(outdim, activation = activation_out))
    return net

class p_x_z(tf.keras.Model):
    def __init__(self, x_bin_dim, x_num_dim, z_dim, nh, h):
        super(p_x_z, self).__init__()
        ########################################
        self.p_x_z_shared = fc_net(z_dim, (nh-1) * [h])
        self.p_x_z_bin = fc_net(h, [h], [x_bin_dim, None])
        self.p_x_z_num_shared = fc_net(h, [h], [])
        self.p_x_z_num_mu = tfkl.Dense(x_num_dim, activation = None)
        self.p_x_z_num_sigma = tfkl.Dense(x_num_dim, activation = "softplus")
        
    def call(self, z_input, training=False, serving=False):
        shared = self.p_x_z_shared(z_input)
        logits = self.p_x_z_bin(shared)
        p_x_z_bin = tfd.Bernoulli(logits = logits)
        num_shared = self.p_x_z_num_shared(shared)
        mu = self.p_x_z_num_mu(num_shared)
        sigma = self.p_x_z_num_sigma(num_shared)
        p_x_z_num = tfd.Normal(loc = mu, scale = sigma)
        return p_x_z_num, p_x_z_bin

class p_t_z(tf.keras.Model):
    def __init__(self, t_dim, z_dim, nh, h):
        super(p_t_z, self).__init__()
        self.p_t_z = fc_net(z_dim, nh * [h], [t_dim, None])

    def call(self, z_input, training=False, serving=False):
        logits = self.p_t_z(z_input)
        p_t_z = tfd.Bernoulli(logits = logits)
        return p_t_z

class p_y_tz(tf.keras.Model):
    def __init__(self, y_dim, t_dim, z_dim, nh, h):
        super(p_y_tz, self).__init__()
        self.t_dim, self.z_dim = t_dim, z_dim
        self.p_y_t0z_mu = fc_net(z_dim, nh * [h], [y_dim, None])
        self.p_y_t1z_mu = fc_net(z_dim, nh * [h], [y_dim, None])

    def call(self, tz_input, training=False, serving=False):
        t,z = tz_input[...,:self.t_dim],tz_input[...,self.t_dim:]
        p_y_t0z_mu = self.p_y_t0z_mu(z)
        p_y_t1z_mu = self.p_y_t1z_mu(z)
        y_loc = t * p_y_t1z_mu + (1-t) * p_y_t0z_mu
        return tfd.Normal(
            loc = y_loc,
            scale = tf.ones_like(y_loc),
        )

class q_t_x(tf.keras.Model):
    def __init__(self, x_bin_dim, x_num_dim, t_dim, nh, h):
        super(q_t_x, self).__init__()
        self.logits_t = fc_net(x_bin_dim+x_num_dim, nh * [h], [t_dim, None])

    def call(self, x_input, training=False, serving=False):
        logits = self.logits_t(x_input)
        q_t_x = tfd.Bernoulli(logits = logits)
        return q_t_x

class q_y_tx(tf.keras.Model):
    def __init__(self, x_bin_dim, x_num_dim, y_dim, t_dim, nh, h):
        super(q_y_tx, self).__init__()
        self.t_dim = t_dim
        self.q_y_xt_shared_hqy = fc_net(x_bin_dim + x_num_dim, (nh - 1) * [h], [])
        self.q_y_xt0_mu = fc_net(h, [h], [y_dim, None])
        self.q_y_xt1_mu = fc_net(h, [h], [y_dim, None])

    def call(self, data, training=False, serving=False):
        hqy = self.q_y_xt_shared_hqy(data)
        qy_t0_mu = self.q_y_xt0_mu(hqy)
        qy_t1_mu = self.q_y_xt1_mu(hqy)

        yt0 = tfd.Normal(loc = qy_t0_mu, scale = tf.ones([tf.shape(data)[0],1]),)
        yt1 = tfd.Normal(loc = qy_t1_mu, scale = tf.ones([tf.shape(data)[0],1]))
        return yt0, yt1

class q_z_txy(tf.keras.Model):
    def __init__(self, x_bin_dim, x_num_dim, y_dim, t_dim, z_dim, nh,h):
        super(q_z_txy, self).__init__()
        self.t_dim = t_dim
        xy_input_shape = x_bin_dim + x_num_dim + y_dim
        self.q_z_xty_shared = fc_net(xy_input_shape, (nh-1) * [h], [])
        
        self.q_z_xty_shared_t0 = tfkl.Dense(h, activation = 'elu')
        self.q_z_xt0y_mu = tfkl.Dense(z_dim, activation = None)
        self.q_z_xt0y_sigma = tfkl.Dense(z_dim, activation = 'softplus')

        self.q_z_xty_shared_t1 = tfkl.Dense(h, activation = 'elu')
        self.q_z_xt1y_mu = tfkl.Dense(z_dim, activation = None)
        self.q_z_xt1y_sigma = tfkl.Dense(z_dim, activation = 'softplus')

    def call(self, txy_input, training=False, serving=False):
        t_input = txy_input[...,:self.t_dim]
        xy_input = txy_input[...,self.t_dim:]
        hqz = self.q_z_xty_shared(xy_input)
        
        q_z_xty_shared_t0 = self.q_z_xty_shared_t0(hqz)
        q_z_xt0y_mu = self.q_z_xt0y_mu(q_z_xty_shared_t0)
        q_z_xt0y_sigma = self.q_z_xt0y_sigma(q_z_xty_shared_t0)

        q_z_xty_shared_t1 = self.q_z_xty_shared_t1(hqz)
        q_z_xt1y_mu = self.q_z_xt1y_mu(q_z_xty_shared_t1)
        q_z_xt1y_sigma = self.q_z_xt1y_sigma(q_z_xty_shared_t1)

        z_loc = t_input * q_z_xt1y_mu + (1-t_input) * q_z_xt0y_mu
        z_scale = t_input * q_z_xt1y_sigma + (1-t_input) * q_z_xt0y_sigma

        return tfd.Normal(
                    loc = z_loc , 
                    scale = z_scale,
                    )