from dataset_tool_tf import *
import matplotlib.pyplot as plt
import scipy.fft as fftpack
import scipy.signal as signal
import pickle

def npgkern(size_kernel, stddev):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    #ax = np.linspace(-(size_kernel - 1) / 2., (size_kernel - 1) / 2., size_kernel)
    #gauss = np.exp(-0.5 * np.square(ax) / np.square(stddev))
    #kernel = np.outer(gauss, gauss)

    kernel = np.random.uniform(size = [size_kernel, size_kernel], low = 0, high = stddev)

    kernel = kernel/np.sum(kernel)
    return kernel

def clip_to_uint8(arr):
    return np.clip((arr + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)

fname = "C:\\Users\\Marco\\Desktop\\n2n_deblur\\noise2noise\\results\\00025-autoencoder'-test'-n2n\\img_1900_y_pred.png"

pkl_file = open(fname, 'rb')

img = pickle.load(pkl_file)
#print(img)

pkl_file.close()

plt.imshow(clip_to_uint8(np.transpose(img, axes=[1, 2, 0])))
plt.show()

print(img.shape)

'''
gkernel = npgkern(7, 25)
gkernel = gkernel[None, :, :]

#convolved = np.zeros(shape=x.shape)

#for i in range(3):
#    convolved[i] = signal.convolve2d(x[i], gkernel, mode='same')

plt.imshow(clip_to_uint8(np.transpose(img, axes=[1, 2, 0])))
plt.show()

convolved = signal.convolve(img, gkernel, mode='same')

plt.imshow(clip_to_uint8(np.transpose(convolved, axes=[1, 2, 0])))
plt.show()

#print(convolved.shape)

ft = fftpack.rfft2(convolved, s = (convolved.shape[1], convolved.shape[2]-2))
ft = np.log(ft)

#print(ft.shape)
#input()

ft = np.concatenate([ft.real, ft.imag], axis = -1)
print(ft.real)

#print(ft.shape)
#print(ft.dtype)
#input()

plt.imshow(np.transpose(ft, [1, 2, 0]))
plt.show()
'''

ft = img

img_real = ft[:, :, :ft.shape[2]//2]
img_imag = ft[:, :, ft.shape[2]//2:]

print(img_real.shape, img_imag.shape)
#input()

ft = img_real + 1j*img_imag
ft = np.exp(ft)

real_img = fftpack.irfft2(ft, s=[ft.shape[1], ft.shape[2]*2])

print(real_img.shape)

real_img = np.transpose(real_img, axes=[1, 2, 0])

plt.imshow(clip_to_uint8(real_img))
plt.show()