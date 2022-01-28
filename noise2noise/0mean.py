from audioop import avg
import scipy.fft as fftpack
import scipy.signal as signal
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
#import tensorflow as tf
import ipdb

def npgkern(size_kernel, stddev, img_len):
    kernel = (np.random.normal(size = [size_kernel, size_kernel]))*stddev
    dpad = (img_len - size_kernel)//2
    kernel = np.pad(kernel, ((dpad, dpad), (dpad, dpad)))
    kernel = fftpack.fftshift(kernel)
    #kernel = np.zeros((size_kernel, size_kernel))
    #kernel[size_kernel//2, size_kernel//2] = kernel[size_kernel//2, size_kernel//2] + 1 
    kernel[0, 0] = kernel[0, 0] + 1
    return kernel
    #kernel = np.random.lognormal(sigma = stddev, size = [size_kernel, size_kernel//2 + 1]) * np.exp(1j * stddev * np.random.normal(size = [size_kernel, size_kernel//2 + 1]))
    #kernel = 1 + stddev * np.random.normal(size = [size_kernel, size_kernel//2 + 1]) + 1j * stddev * np.random.normal(size = [size_kernel, size_kernel//2 + 1]) 
    #kernel_real = fftpack.irfft2(kernel, [size_kernel, size_kernel])
    #kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    #kernel = kernel/np.sum(kernel)
    #return kernel_real, kernel
    return kernel_real

def transform(img, std):
    ft = fftpack.rfft2(img)
    #noise = np.random.lognormal(mean = -std**2/2, sigma = std, size = ft.shape)
    r = np.random.normal(size = ft.shape) * std
    #theta = np.exp(1j * np.random.normal(size = ft.shape) * std)
    im = np.random.normal(size = ft.shape) * std
    noise = (1 + r) + 1j * im
    noised_ft = ft * noise
    x = fftpack.irfft2(noised_ft, s = [img.shape[1], img.shape[2]])
    return x

img_len = 5
img = np.random.uniform(size = (1, img_len, img_len))
#tf_img = tf.constant(img, dtype = tf.float32)
avg_img = np.zeros(img.shape)
avg_img2 = np.zeros(img.shape)
avg_img3 = np.zeros(img.shape)
#tf_avg_img2 = tf.constant(avg_img2, dtype = tf.float32) 
#tf_img = tf_img[:, :, :, tf.newaxis]
std = 1

N = 1000000
kernel_size = 3
dists = []
dists2 = []
dists3 = []

#with tf.Session() as sess:
for i in tqdm(range(N)):
    mod = transform(img, std)
    avg_img = avg_img + mod
    dist = np.sum((avg_img/(i+1) - img)**2)
    dists.append(dist)
    
    kernel = npgkern(kernel_size, std, img_len)
    mod2 = signal.convolve2d(img[0, :, :], kernel, mode = 'same')
    avg_img3 = avg_img3 + mod2
    dist = np.sum((avg_img3/(i+1) - img)**2)
    dists3.append(dist)

    #ipdb.set_trace()

    '''
    tf_kernel = tf.constant(kernel, dtype = tf.float32)
    tf_kernel = tf_kernel[:, :, tf.newaxis, tf.newaxis]

    convolved = tf.squeeze(tf.nn.conv2d(tf_img, tf_kernel, strides=[1,1,1,1], padding='SAME'))
    convolved = convolved[tf.newaxis, :, :]

    tf_avg_img2 = tf_avg_img2 + convolved

    dist = np.sum((tf_avg_img2.eval()/(i+1) - img)**2)
    dists2.append(dist)
    '''
    

plt.semilogy(dists, label = 'Fourier multiplicative noise')
plt.semilogy(dists2, label = 'Tensorflow convolution')
plt.semilogy(dists3, label = 'Scipy convolution')
plt.legend()
plt.show()

ipdb.set_trace()


'''
#kernel, ft_kernel = npgkern(3, 0.01)
#ft = fftpack.rfft2(kernel)
#ft = np.log(ft)
#avg_kernel = np.zeros(kernel.shape)
#avg_ft_kernel = np.zeros(ft_kernel.shape)
#total = np.zeros(ft.shape, dtype = np.complex128)
#dist = []

N = 1000000
for i in tqdm(range(N)):
    #ipdb.set_trace()
    kernel, ft_kernel = npgkern(3, 0.01)
    ft = fftpack.rfft2(kernel)
    #print(ft)
    #input()
    ft = np.log(ft)
    avg_kernel = avg_kernel + kernel
    avg_ft_kernel = avg_ft_kernel + np.log(ft_kernel)
    total = total + ft

total = total/N
avg_kernel = avg_kernel/N
avg_ft_kernel = avg_ft_kernel/N
print(avg_ft_kernel)
print(total)
print(avg_kernel)
#plt.plot(dist)
#plt.show()
ipdb.set_trace()
'''
