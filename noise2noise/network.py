# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import tensorflow as tf
import numpy as np
import ipdb


#----------------------------------------------------------------------------
# Get/create weight tensor for a convolutional or fully-connected layer.

def get_weight(shape, gain=np.sqrt(2)):
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    std = gain / np.sqrt(fan_in) # He init
    w = tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))
    return w

#----------------------------------------------------------------------------
# Convolutional layer.

def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    return x + tf.reshape(b, [1, -1, 1, 1])

def conv2d_bias(x, fmaps, kernel, gain=np.sqrt(2)):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain)
    w = tf.cast(w, x.dtype)
    return apply_bias(tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW'))

def maxpool2d(x, k=2):
    ksize = [1, 1, k, k]
    return tf.nn.max_pool(x, ksize=ksize, strides=ksize, padding='SAME', data_format='NCHW')

# TODO use fused upscale+conv2d from gan2
def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x

def conv_lr(name, x, fmaps):
    with tf.variable_scope(name):
        return tf.nn.leaky_relu(conv2d_bias(x, fmaps, 3), alpha=0.1)

def conv(name, x, fmaps, gain):
    with tf.variable_scope(name):
        return conv2d_bias(x, fmaps, 3, gain)

def autoencoder(x, width=256, height=256, **_kwargs):
    #x.set_shape([None, 3, height, width])    ### Normal 3 RGB channels with standard width.
    #x.set_shape([None, 6, height, width//2]) ### Half-width and 6 channels for the rfft
    #x.set_shape([None, 2, height, width//2]) ### Half-width and 2 channels for the B/W rfft
    x.set_shape([None, 1, height, width])     ### Normal B/W image
    skips = [x]

    n = x   #?, 6, 256, 128
    
    #ipdb.set_trace()
    
    n = conv_lr('enc_conv0', n, 48) #?, 48, 256, 128
    n = conv_lr('enc_conv1', n, 48) #?, 48, 256, 128
    n = maxpool2d(n) #?, 48, 128, 64
    skips.append(n) # [[?, 6, 256, 128], [?, 48, 128, 64]]

    n = conv_lr('enc_conv2', n, 48) #?, 48, 128, 64
    n = maxpool2d(n) #?, 48, 64, 32
    skips.append(n) # [[?, 6, 256, 128], [?, 48, 128, 64], [?, 48, 64, 32]]

    n = conv_lr('enc_conv3', n, 48) #?, 48, 64, 32
    n = maxpool2d(n) #?, 48, 32, 16
    skips.append(n) # [[?, 6, 256, 128], [?, 48, 128, 64], [?, 48, 64, 32], [?, 48, 32, 16]]

    n = conv_lr('enc_conv4', n, 48) #?, 48, 32, 16
    n = maxpool2d(n) #?, 48, 16, 8
    skips.append(n) # [[?, 6, 256, 128], [?, 48, 128, 64], [?, 48, 64, 32], [?, 48, 32, 16], [?, 48, 16, 8]]

    n = conv_lr('enc_conv5', n, 48) #?, 48, 16, 8
    n = maxpool2d(n) #?, 48, 8, 4
    n = conv_lr('enc_conv6', n, 48) #?, 48, 8, 4

    #-----------------------------------------------
    n = upscale2d(n) #?, 48, 16, 8
    n = tf.concat([n, skips.pop()], axis=1) #?, 96, 16, 8
    n = conv_lr('dec_conv5', n, 96) #?, 96, 16, 8
    n = conv_lr('dec_conv5b', n, 96) #?, 96, 16, 8

    n = upscale2d(n) #?, 96, 32, 16
    n = tf.concat([n, skips.pop()], axis=1) #?, 144, 32, 16
    n = conv_lr('dec_conv4', n, 96) #?, 96, 32, 16
    n = conv_lr('dec_conv4b', n, 96) #?, 96, 32, 16

    n = upscale2d(n) #?, 96, 64, 32
    n = tf.concat([n, skips.pop()], axis=1) #?, 144, 64, 32
    n = conv_lr('dec_conv3', n, 96) #?, 96, 64, 32
    n = conv_lr('dec_conv3b', n, 96) #?, 96, 64, 32

    n = upscale2d(n) #?, 96, 128, 64
    n = tf.concat([n, skips.pop()], axis=1) #?, 144, 128, 64
    n = conv_lr('dec_conv2', n, 96) #?, 96, 128, 64
    n = conv_lr('dec_conv2b', n, 96) #?, 96, 128, 64

    n = upscale2d(n)  #?, 96, 256, 128
    n = tf.concat([n, skips.pop()], axis=1) #?, 102, 256, 128
    n = conv_lr('dec_conv1a', n, 64) #?, 64, 256, 128
    n = conv_lr('dec_conv1b', n, 32) #?, 32, 256, 128

    #n = conv('dec_conv1', n, 3, gain=1.0)  ### normal 3 RGB channels
    #n = conv('dec_conv1', n, 6, gain=1.0)   ### 6 channels for the rfft
    #n = conv('dec_conv1', n, 2, gain=1.0)   ### 2 channels for the B/W rfft
    n = conv('dec_conv1', n, 1, gain=1.0) 

    return n #?, 6, 256, 128
