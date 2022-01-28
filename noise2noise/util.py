# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import numpy as np
import pickle
import PIL.Image
import scipy.fft as fftpack
import ipdb

import dnnlib.submission.submit as submit

# save_pkl, load_pkl are used by the mri code to save datasets
def save_pkl(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_pkl(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# save_snapshot, load_snapshot are used save/restore trained networks
def save_snapshot(submit_config, net, fname_postfix):
    dump_fname = os.path.join(submit_config.run_dir, "network_%s.pickle" % fname_postfix)
    with open(dump_fname, "wb") as f:
        pickle.dump(net, f)

def load_snapshot(fname):
    fname = os.path.join(submit.get_path_from_template(fname))
    with open(fname, "rb") as f:
        return pickle.load(f)


def invert_fft_re_im(img):
    ft = img

    img_real = ft[:, :, :ft.shape[2]//2]
    img_imag = ft[:, :, ft.shape[2]//2:]

    ft = img_real + 1j*img_imag
    #ft = np.exp(ft)

    real_img = fftpack.irfft2(ft, s=[ft.shape[1], ft.shape[2]*2])
    return real_img

def invert_fft_abs_ang(img):
    ft = img

    #img_abs = ft[:3, :, :]
    #img_ang = ft[3:, :, :]  <-------- Color
    img_abs = ft[0, :, :]
    img_ang = ft[1, :, :]


    ft = img_abs * np.exp(1j*img_ang) 
    #ft = np.exp(ft)
    ft = ft[None, :, :]

    real_img = fftpack.irfft2(ft, s=[ft.shape[1], ft.shape[2]*2])
    return real_img

def loop_fft(img, dir = 1):
    ft = fftpack.rfft2(img)
    #if dir == 1:
    #    ft = np.exp(ft)
    #else:
    #    ft = np.log(ft)
    out = fftpack.irfft2(ft)
    return out

def save_image(submit_config, img_t, filename):
    #img_t = invert_fft_abs_ang(img_t)
    #img_t = loop_fft(img_t)
    img_t = clip_to_uint8(img_t)
    t = img_t.transpose([1, 2, 0])  # [RGB, H, W] -> [H, W, RGB]
    if t.dtype in [np.float32, np.float64]:
        t = clip_to_uint8(t)
    else:
        assert t.dtype == np.uint8
    #PIL.Image.fromarray(t, 'RGB').save(os.path.join(submit_config.run_dir, filename)) #<--------- Color
    PIL.Image.fromarray(t.squeeze(), 'L').save(os.path.join(submit_config.run_dir, filename))  # <-------- B/W
    '''
    filename = os.path.join(submit_config.run_dir, filename)
    output = open(filename, 'wb')
    pickle.dump(img_t, output)
    output.close()
    '''
    
def save_clean_image(submit_config, img_t, filename):
    t = img_t.transpose([1, 2, 0])  # [RGB, H, W] -> [H, W, RGB]
    if t.dtype in [np.float32, np.float64]:
        t = clip_to_uint8(t)
    else:
        assert t.dtype == np.uint8
    #PIL.Image.fromarray(t, 'RGB').save(os.path.join(submit_config.run_dir, filename)) # <--------- Color
    PIL.Image.fromarray(t.squeeze(), 'L').save(os.path.join(submit_config.run_dir, filename))  # <-------- B/W

def clip_to_uint8(arr):
    return np.clip((arr + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)

def crop_np(img, x, y, w, h):
    return img[:, y:h, x:w]

# Run an image through the network (apply reflect padding when needed
# and crop back to original dimensions.)
def infer_image(net, img):
    #ipdb.set_trace()
    w = img.shape[2]
    h = img.shape[1]
    pw, ph = (w+31)//32*32-w, (h+31)//32*32-h
    padded_img = img
    if pw!=0 or ph!=0:
        padded_img  = np.pad(img, ((0,0),(0,ph),(0,pw)), 'reflect')
    #inferred = net.run(np.expand_dims(padded_img, axis=0), width=(w+pw)*2, height=h+ph)
    inferred = net.run(np.expand_dims(padded_img, axis=0), width=(w+pw), height=h+ph) 
    #return clip_to_uint8(crop_np(inferred[0], 0, 0, w, h))
    return crop_np(inferred[0], 0, 0, w, h)
    #return crop_np(inferred[0], 0, 0, w, h)

def create_circle(shape, radius):
    #ipdb.set_trace()
    out = np.ones(shape)
    for i in range(-radius, radius+1):
        for j in range(-radius, radius+1):
            if i**2 + j**2 <= radius**2:
                out[:, i, j] = 0
    return out

def remove_DC(img, rad):
    ft = fftpack.rfft2(img)
    low_pass = create_circle(ft.shape, rad)
    ft = ft * low_pass
    return fftpack.irfft2(ft, s = [img.shape[1], img.shape[2]])

def restore_DC(orig_img, img, rad):
    #ipdb.set_trace()
    orig_ft = fftpack.rfft2(orig_img)
    ft = fftpack.rfft2(img)
    low_pass = create_circle(ft.shape, rad)
    return fftpack.irfft2(ft + (1 - low_pass) * orig_ft, s = [img.shape[1], img.shape[2]])

