# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

from venv import create
import tensorflow as tf
import numpy as np

import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary
import dnnlib.tflib.tfutil as tfutil
import dnnlib.util as util

import config

from util import save_image, save_snapshot, invert_fft_abs_ang, loop_fft, save_clean_image, create_circle
from validation import ValidationSet
from dataset import create_dataset
import numpy as np
import scipy.signal as signal
import scipy.fft as fftpack
import sys
import PIL.Image
import ipdb

def gkern(size_kernel, stddev):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    #ax = tf.linspace(-(size_kernel - 1) / 2., (size_kernel - 1) / 2., size_kernel)
    #gauss = tf.exp(-0.5 * tf.square(ax) / tf.square(stddev))
    #kernel = tf.tensordot(gauss, gauss, axes=0)
    '''kernel = (tf.random_normal(shape = [size_kernel, size_kernel]))*stddev
    kernel[size_kernel//2, size_kernel//2] += 1
    #kernel = tf.nn.relu(kernel)
    kernel, _ = tf.linalg.normalize(kernel, ord=1)
    return kernel
    #tf.constant(kernel / tf.reduce_sum(kernel), dtype=tf.float32)'''
    kernel = npgkern(size_kernel, stddev)
    return tf.constant(kernel, dtype=tf.float32)

def npgkern(size_kernel, stddev):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    #ax = np.linspace(-(size_kernel - 1) / 2., (size_kernel - 1) / 2., size_kernel)
    #gauss = np.exp(-0.5 * np.square(ax) / np.square(stddev))
    #kernel = np.outer(gauss, gauss)
    
    kernel = (np.random.normal(size = [size_kernel, size_kernel]))*stddev
    kernel[size_kernel//2, size_kernel//2] = kernel[size_kernel//2, size_kernel//2] + 1
    #kernel = kernel * (kernel > 0)
    #kernel = kernel/np.sum(kernel)
    '''
    kernel = np.random.lognormal(sigma = stddev, size = [size_kernel, size_kernel//2 + 1]) * np.exp(1j * stddev * np.random.normal(size = [size_kernel, size_kernel//2 + 1]))
    kernel = fftpack.irfft2(kernel, [size_kernel, size_kernel])
    '''
    return kernel
    #tf.constant(kernel / tf.reduce_sum(kernel), dtype=tf.float32)


class AugmentBlur:
    def __init__(self, size_kernel, validation_stddev, train_stddev_rng_range):
        self.validation_stddev = validation_stddev
        self.train_stddev_range = train_stddev_rng_range
        self.size_kernel = size_kernel
        self.dc_removed = None
    
    def access_dc(self):
        return self.dc_removed

    def add_train_noise_tf(self, x):
        #ipdb.set_trace()

        (minval,maxval) = self.train_stddev_range
        rng_stddev = np.random.uniform(low = minval, high = maxval)
        gkernel = gkern(self.size_kernel, rng_stddev)
        x = x[:, :, :, tf.newaxis] 
        gkernel = gkernel[:-1, :-1, tf.newaxis, tf.newaxis]
        convolved = tf.squeeze(tf.nn.conv2d(x, gkernel, strides=[1,1,1,1], padding='SAME'))
        
        #ipdb.set_trace()
        convolved = convolved[tf.newaxis, :, :] #### <------- B/W 
        ft = tf.signal.rfft2d(convolved)
        #ft = tf.log(ft)

        low_pass = create_circle(ft.get_shape().as_list(), 10)
        low_pass = tf.constant(low_pass, dtype = tf.complex64)
        ft = ft * low_pass
        out = tf.signal.irfft2d(ft)

        return out


    def add_validation_noise_np(self, x):

        #ipdb.set_trace()

        #img = invert_fft_abs_ang(x)

        gkernel = npgkern(self.size_kernel, self.validation_stddev)

        real_x = signal.convolve2d(x.squeeze(), gkernel, mode='same')
        real_x = real_x[None, :, :]

        ft = fftpack.rfft2(real_x)
        #ft = np.log(ft)
        
        low_pass = create_circle(ft.shape, 10)
        self.dc_removed = ft*(1- low_pass)
        ft = ft * low_pass

        out = fftpack.irfft2(ft, s = [real_x.shape[1], real_x.shape[2]])
        #ipdb.set_trace()
        return out


class AugmentGaussianBlur:
    def __init__(self, size_kernel, validation_stddev, train_stddev_rng_range):
        self.validation_stddev = validation_stddev
        self.train_stddev_range = train_stddev_rng_range
        self.size_kernel = size_kernel

    def add_train_noise_tf(self, x):
        (minval,maxval) = self.train_stddev_range
        #shape = tf.shape(x)
        #rng_stddev = tf.random_uniform(shape=[1], minval=minval, maxval=maxval)
        rng_stddev = np.random.uniform(low = minval, high = maxval)
        gkernel = gkern(self.size_kernel, rng_stddev)
        x = x[:, :, :, tf.newaxis] 
        gkernel = gkernel[:, :, tf.newaxis, tf.newaxis]
        convolved = tf.squeeze(tf.nn.conv2d(x, gkernel, strides=[1,1,1,1], padding='SAME'))
        
        convolved = convolved[tf.newaxis, :, :] #### <------- B/W 

        return convolved


    def add_validation_noise_np(self, x):

        gkernel = gkern(self.size_kernel, self.validation_stddev)
        x = tf.constant(x, dtype = tf.float32)
        
        x = x[:, :, :, tf.newaxis]
        gkernel = gkernel[:, :, tf.newaxis, tf.newaxis]
        convolved = tf.squeeze(tf.nn.conv2d(x, gkernel, strides=[1,1,1,1], padding='SAME'))
        convolved = convolved[tf.newaxis, :, :]

        return convolved.eval()

class AugmentSuper:
    def __init__(self, validation_stddev, train_stddev_rng_range):
        self.validation_stddev = validation_stddev
        self.train_stddev_range = train_stddev_rng_range

    def add_train_noise_tf(self, x):
        (minval,maxval) = self.train_stddev_range
        shape = x.get_shape().as_list()
        #ipdb.set_trace()
        #print(shape)
        x = tf.transpose(x, perm=[1, 2, 0])
        x = x[tf.newaxis, :, :, :]
        x = tf.image.resize(x, size = [shape[1]//2, shape[2]//2], method = tf.image.ResizeMethod.BICUBIC)
        x = tf.image.resize(x, size = [shape[1], shape[2]], method = tf.image.ResizeMethod.BICUBIC)
        noise = tf.tile(tf.constant([[[0, 1],[1, 0]], [[0, 1],[1, 0]], [[0, 1],[1, 0]]], dtype = tf.float32), [1, shape[1]//2, shape[2]//2])
        #noise = tf.tile(tf.constant([[0,1], [1,0]], dtype = tf.float32), [1, shape[1]//2, shape[2]//2]) # <----- B/W
        #x = (1 - noise)*x
        #print(noise.get_shape())
        #input()
        rng_stddev = tf.random_uniform(shape=[1, 1, 1], minval=minval/255.0, maxval=maxval/255.0)
        
        x = tf.squeeze(x)
        #x = x[:, :, tf.newaxis] ## <----- B/W
        x = tf.transpose(x, perm=[2, 0, 1])
        x = x + tf.random_normal(shape) * rng_stddev * noise
        return x

    def add_validation_noise_np(self, x):

        #ipdb.set_trace()

        shape = x.shape
        #print(shape)
        
        x = tf.constant(x, dtype = tf.float32)
        x = tf.transpose(x, perm=[1, 2, 0])
        x = x[tf.newaxis, :, :, :]
        x = tf.image.resize(x, size = [shape[1]//2, shape[2]//2], method = tf.image.ResizeMethod.BICUBIC)
        x = tf.image.resize(x, size = [shape[1], shape[2]], method = tf.image.ResizeMethod.BICUBIC)
        x = tf.squeeze(x)
        x = tf.transpose(x, perm=[2, 0, 1])
        x = x.eval()

        noise = np.tile(np.array([[[0, 1],[1, 0]], [[0, 1],[1, 0]], [[0, 1],[1, 0]]], dtype = np.float32), [1, shape[1]//2, shape[2]//2])
        x = (1.0 - noise)*x
        kernel = 1/4*np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        kernel = kernel[None, :, :]
        x = x + noise * signal.convolve(x, kernel, mode='same')
        #input()
        #print(noise.get_shape())
        #input()
        x = x +  np.random.normal(size=shape) * (self.validation_stddev/255.0) * noise
        return x

class AugmentGaussian:
    def __init__(self, validation_stddev, train_stddev_rng_range):
        self.validation_stddev = validation_stddev
        self.train_stddev_range = train_stddev_rng_range

    def add_train_noise_tf(self, x):
        (minval,maxval) = self.train_stddev_range
        shape = tf.shape(x)
        ft = tf.signal.rfft2d(x)
        shape_ft = ft.get_shape()
        rng_stddev = tf.random_uniform(shape=[1, 1, 1], minval=minval/255.0, maxval=maxval/255.0)
        #rng_stddev_theta = tf.random_uniform(shape=[1, 1, 1], minval=minval/255.0, maxval=maxval/255.0)
        rng_stddev = tf.cast(rng_stddev, dtype = tf.complex64)
        #rng_stddev_theta = tf.cast(rng_stddev_theta, dtype = tf.complex64)
        #return x * (1 + tf.random_normal(shape) * rng_stddev)
        #noised_ft = ft * (1 + tf.cast(tf.random_normal(shape_ft), dtype = tf.complex64) * rng_stddev_r) # * 
                    #tf.exp(1j * tf.cast(tf.random_normal(shape_ft), dtype = tf.complex64) * rng_stddev_theta))
        #np.random.lognormal(sigma = stddev, size = [size_kernel, size_kernel//2 + 1])
        #std = np.random.uniform(minval/255.0, maxval/255.0)
        #noise = np.random.lognormal(mean = - std**2/2, sigma = std, size = shape_ft.as_list())
        #noise = tf.constant(noise, dtype = tf.complex64)
        #noised_ft = ft * noise
        noise = 1 + tf.cast(tf.random_normal(shape_ft), dtype = tf.complex64) * rng_stddev + 1j * tf.cast(tf.random_normal(shape_ft), dtype = tf.complex64) * rng_stddev
        noised_ft = ft * noise
        x = tf.signal.irfft2d(noised_ft)
        return x

    def add_validation_noise_np(self, x):
        #return x * (1 + np.random.normal(size=x.shape)*(self.validation_stddev/255.0))
        ft = fftpack.rfft2(x)
        std = self.validation_stddev/255.0
        noise = 1 + np.random.normal(size = ft.shape)*std + 1j * np.random.normal(size = ft.shape)*std
        #noise = np.random.lognormal(mean = -std**2/2, sigma = std, size = ft.shape)
        noised_ft = ft * noise
        x = fftpack.irfft2(noised_ft, s = [x.shape[1], x.shape[2]])
        return x

class AugmentPoisson:
    def __init__(self, lam_max):
        self.lam_max = lam_max

    def add_train_noise_tf(self, x):
        chi_rng = tf.random_uniform(shape=[1, 1, 1], minval=0.001, maxval=self.lam_max)
        return tf.random_poisson(chi_rng*(x+0.5), shape=[])/chi_rng - 0.5

    def add_validation_noise_np(self, x):
        chi = 30.0
        return np.random.poisson(chi*(x+0.5))/chi - 0.5

def compute_ramped_down_lrate(i, iteration_count, ramp_down_perc, learning_rate):
    ramp_down_start_iter = iteration_count * (1 - ramp_down_perc)
    if i >= ramp_down_start_iter:
        t = ((i - ramp_down_start_iter) / ramp_down_perc) / iteration_count
        smooth = (0.5+np.cos(t * np.pi)/2)**2
        return learning_rate * smooth
    return learning_rate

def train(
    submit_config: dnnlib.SubmitConfig,
    iteration_count: int,
    eval_interval: int,
    minibatch_size: int,
    learning_rate: float,
    ramp_down_perc: float,
    noise: dict,
    validation_config: dict,
    train_tfrecords: str,
    noise2noise: bool):
    noise_augmenter = dnnlib.util.call_func_by_name(**noise)
    validation_set = ValidationSet(submit_config)
    validation_set.load(**validation_config)

    # Create a run context (hides low level details, exposes simple API to manage the run)
    ctx = dnnlib.RunContext(submit_config, config)

    # Initialize TensorFlow graph and session using good default settings
    tfutil.init_tf(config.tf_config)

    dataset_iter = create_dataset(train_tfrecords, minibatch_size, noise_augmenter.add_train_noise_tf)

    # Construct the network using the Network helper class and a function defined in config.net_config
    with tf.device("/gpu:0"):
        net = tflib.Network(**config.net_config)

    # Optionally print layer information
    net.print_layers()

    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'), tf.device("/cpu:0"):
        lrate_in        = tf.placeholder(tf.float32, name='lrate_in', shape=[])

        noisy_input, noisy_target, clean_target = dataset_iter.get_next()
        noisy_input_split = tf.split(noisy_input, submit_config.num_gpus)
        noisy_target_split = tf.split(noisy_target, submit_config.num_gpus)
        clean_target_split = tf.split(clean_target, submit_config.num_gpus)

    # Define the loss function using the Optimizer helper class, this will take care of multi GPU
    opt = tflib.Optimizer(learning_rate=lrate_in, **config.optimizer_config)

    for gpu in range(submit_config.num_gpus):
        with tf.device("/gpu:%d" % gpu):
            net_gpu = net if gpu == 0 else net.clone()

            denoised = net_gpu.get_output_for(noisy_input_split[gpu])

            if noise2noise:
                meansq_error = tf.reduce_mean(tf.square(noisy_target_split[gpu] - denoised))
            else:
                meansq_error = tf.reduce_mean(tf.square(clean_target_split[gpu] - denoised))
            # Create an autosummary that will average over all GPUs
            with tf.control_dependencies([autosummary("Loss", meansq_error)]):
                opt.register_gradients(meansq_error, net_gpu.trainables)

    train_step = opt.apply_updates()

    # Create a log file for Tensorboard
    summary_log = tf.summary.FileWriter(submit_config.run_dir)
    summary_log.add_graph(tf.get_default_graph())

    print('Training...')
    time_maintenance = ctx.get_time_since_last_update()
    ctx.update(loss='run %d' % submit_config.run_id, cur_epoch=0, max_epoch=iteration_count)

    # The actual training loop
    for i in range(iteration_count):
        # Whether to stop the training or not should be asked from the context
        if ctx.should_stop():
            break

        # Dump training status
        if i % eval_interval == 0:

            time_train = ctx.get_time_since_last_update()
            time_total = ctx.get_time_since_start()

            # Evaluate 'x' to draw a batch of inputs
            [source_mb, target_mb] = tfutil.run([noisy_input, clean_target])
            denoised = net.run(source_mb)
            save_image(submit_config, denoised[0], "img_{0}_y_pred.png".format(i))
            save_clean_image(submit_config, target_mb[0], "img_{0}_y.png".format(i))
            save_image(submit_config, source_mb[0], "img_{0}_x_aug.png".format(i))

            validation_set.evaluate(net, i, noise_augmenter.add_validation_noise_np)#, noise_augmenter.access_dc)

            print('iter %-10d time %-12s sec/eval %-7.1f sec/iter %-7.2f maintenance %-6.1f' % (
                autosummary('Timing/iter', i),
                dnnlib.util.format_time(autosummary('Timing/total_sec', time_total)),
                autosummary('Timing/sec_per_eval', time_train),
                autosummary('Timing/sec_per_iter', time_train / eval_interval),
                autosummary('Timing/maintenance_sec', time_maintenance)))

            dnnlib.tflib.autosummary.save_summaries(summary_log, i)
            ctx.update(loss='run %d' % submit_config.run_id, cur_epoch=i, max_epoch=iteration_count)
            time_maintenance = ctx.get_last_update_interval() - time_train

        lrate =  compute_ramped_down_lrate(i, iteration_count, ramp_down_perc, learning_rate)
        tfutil.run([train_step], {lrate_in: lrate})

    print("Elapsed time: {0}".format(util.format_time(ctx.get_time_since_start())))
    save_snapshot(submit_config, net, 'final')

    # Summary log and context should be closed at the end
    summary_log.close()
    ctx.close()
