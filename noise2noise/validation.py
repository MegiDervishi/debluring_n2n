# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import sys
import numpy as np
import PIL.Image
import tensorflow as tf

import dnnlib
import dnnlib.submission.submit as submit
import dnnlib.tflib.tfutil as tfutil
from dnnlib.tflib.autosummary import autosummary

import util
import config
import ipdb
import scipy.fft as fftpack

class ValidationSet:
    def __init__(self, submit_config):
        self.images = None
        self.submit_config = submit_config
        return

    def load(self, dataset_dir):
        import glob

        abs_dirname = os.path.join(submit.get_path_from_template(dataset_dir), '*')
        fnames = sorted(glob.glob(abs_dirname))
        if len(fnames) == 0:
            print ('\nERROR: No files found using the following glob pattern:', abs_dirname, '\n')
            sys.exit(1)

        images = []
        for fname in fnames:
            try:
                #im = PIL.Image.open(fname).convert('RGB') #<--------------- Color
                im = PIL.Image.open(fname).convert('L') #<--------------- B/W
                arr = np.array(im, dtype=np.float32)
                arr = arr[:, :, None] #<----------- B/W
                reshaped = arr.transpose([2, 0, 1]) / 255.0 - 0.5
                ### ---------- perform fft --------------
                '''
                convolved = reshaped
                shape = convolved.shape
                ft = fftpack.rfft2(convolved, s = [shape[1], shape[2]-2])
                ft = np.log(ft)

                ft = np.concatenate([np.abs(ft), np.angle(ft)], 0)

                reshaped = ft
                '''
                # -------------- end of fft --------------
                images.append(reshaped)
            except OSError as e:
                print ('Skipping file', fname, 'due to error: ', e)
        self.images = images

    '''
    def evaluate(self, net, iteration, noise_func, dc_func):
        #ipdb.set_trace()
        
        avg_psnr = 0.0
        for idx in range(len(self.images)):
            orig_img = self.images[idx]
            w = orig_img.shape[2] 
            h = orig_img.shape[1]
            orig255 = util.clip_to_uint8(orig_img)

            noisy_img = noise_func(orig_img)
            #ipdb.set_trace()
            dc_removed = dc_func()
            pred = util.infer_image(net, noisy_img)
            predfft = fftpack.rfft2(pred)
            predfft = predfft * util.create_circle(predfft.shape, 10)   
            predfft = predfft + dc_removed
            #predfft = np.exp(predfft)
            ## predfft = np.exp(predfft) * low_pass + np.exp(dc_removed) * (1 - low_pass)
            predifft = fftpack.irfft2(predfft, s = [orig_img.shape[1], orig_img.shape[2]])
            pred255 = util.clip_to_uint8(predifft) 

            ####
            noisy_img = fftpack.irfft2(fftpack.rfft2(noisy_img) * util.create_circle(predfft.shape, 10) + dc_removed,
                            s = [noisy_img.shape[1], noisy_img.shape[2]])
            ####

            assert (pred255.shape[2] == w and pred255.shape[1] == h)

            sqerr = np.square(orig255.astype(np.float32) - pred255.astype(np.float32))
            s = np.sum(sqerr)
            cur_psnr = 10.0 * np.log10((255*255)/(s / (w*h*3)))
            avg_psnr += cur_psnr

            util.save_clean_image(self.submit_config, pred255, "img_{0}_val_{1}_pred.png".format(iteration, idx))

            if iteration == 0:
                util.save_clean_image(self.submit_config, orig_img, "img_{0}_val_{1}_orig.png".format(iteration, idx))
                util.save_image(self.submit_config, noisy_img, "img_{0}_val_{1}_noisy.png".format(iteration, idx))
        avg_psnr /= len(self.images)
        print ('Average PSNR: %.2f' % autosummary('PSNR_avg_psnr', avg_psnr))
    '''
    def evaluate(self, net, iteration, noise_func):
        #ipdb.set_trace()
        
        avg_psnr = 0.0
        for idx in range(len(self.images)):
            #ipdb.set_trace()
            orig_img = self.images[idx]
            w = orig_img.shape[2] 
            h = orig_img.shape[1]

            noisy_img = noise_func(orig_img)
            pred = util.infer_image(net, noisy_img)

            orig255 = util.clip_to_uint8(orig_img)
            pred255 = util.clip_to_uint8(pred)

            assert (pred255.shape[2] == w and pred255.shape[1] == h)

            sqerr = np.square(orig255.astype(np.float32) - pred255.astype(np.float32))
            s = np.sum(sqerr)
            cur_psnr = 10.0 * np.log10((255*255)/(s / (w*h*3)))
            avg_psnr += cur_psnr

            util.save_clean_image(self.submit_config, pred255, "img_{0}_val_{1}_pred.png".format(iteration, idx))

            if iteration == 0:
                util.save_clean_image(self.submit_config, orig_img, "img_{0}_val_{1}_orig.png".format(iteration, idx))
                util.save_image(self.submit_config, noisy_img, "img_{0}_val_{1}_noisy.png".format(iteration, idx))
        avg_psnr /= len(self.images)
        print ('Average PSNR: %.2f' % autosummary('PSNR_avg_psnr', avg_psnr))


def validate(submit_config: dnnlib.SubmitConfig, noise: dict, dataset: dict, network_snapshot: str):
    noise_augmenter = dnnlib.util.call_func_by_name(**noise)
    validation_set = ValidationSet(submit_config)
    validation_set.load(**dataset)

    ctx = dnnlib.RunContext(submit_config, config)

    tfutil.init_tf(config.tf_config)

    with tf.device("/gpu:0"):
        net = util.load_snapshot(network_snapshot)
        validation_set.evaluate(net, 0, noise_augmenter.add_validation_noise_np)#, noise_augmenter.access_dc)
    ctx.close()

def infer_image(network_snapshot: str, image: str, out_image: str):
    tfutil.init_tf(config.tf_config)
    net = util.load_snapshot(network_snapshot)
    im = PIL.Image.open(image).convert('RGB')
    arr = np.array(im, dtype=np.float32)
    reshaped = arr.transpose([2, 0, 1]) / 255.0 - 0.5
    pred255 = util.infer_image(net, reshaped)
    t = pred255.transpose([1, 2, 0])  # [RGB, H, W] -> [H, W, RGB]
    PIL.Image.fromarray(t, 'RGB').save(os.path.join(out_image))
    print ('Inferred image saved in', out_image)
