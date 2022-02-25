# Median filtering without OpenCV
import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt


#--------------------------------- WRITE YOUR CODE HERE ---------------------------------#
def median_filter_gray(image, w_size):
    height = len(image)
    width = len(image[0])
    kl = int((w_size - 1) / 2)
    ker_len = 2 * kl + 1

    ## start dealing margin
    im = np.zeros((height + kl * 2, width + kl * 2), dtype=np.int)  # create an empty ndarray
    for i in range(kl):
        for j in range(kl):
            im[i, j] = image[kl + 1 - j, kl + 1 - i]  # up left
            im[i, width + kl + j] = image[kl + 1 - j, width - 1 - i]  # up right
            im[height + kl + i, j] = image[height - 1 - j, kl + 1 - i]  # down left
            im[height + kl + i, width + kl + j] = image[height - 1 - j, width - 1 - i]  # down right
    for i in range(kl):
        im[i, kl: width + kl] = image[kl + 1 - i, :]  # up
        im[height + kl + i, kl: width + kl] = image[height - 1 - i, :]  # down
        im[kl: height + kl, i] = image[:, kl + 1 - i]  # left
        im[kl: height + kl, width + kl + i] = image[:, width - 1 - i]  # right
    im[kl:height + kl, kl:width + kl] = image[:, :]  # center
    ## finish dealing margin

    image = np.zeros((height, width), dtype=np.float)

    # convolve the image with given kernel
    # the pixel in the image
    for i in range(len(image)):
        for j in range(len(image[0])):
            image[i, j] = np.median(im[i:i + ker_len, j:j + ker_len])

    return image



##########################################################################################


im_gray = cv2.imread('../inputs/lena.jpg',0)
im_gray = cv2.resize(im_gray, (256,256))

gaussian_noise = np.zeros((im_gray.shape[0], im_gray.shape[1]),dtype=np.uint8)#
gaussian_noise = cv2.randn(gaussian_noise, 128, 20)

uniform_noise = np.zeros((im_gray.shape[0], im_gray.shape[1]),dtype=np.uint8)
uniform_noise = cv2.randu(uniform_noise,0,255)
ret, impulse_noise = cv2.threshold(uniform_noise,220,255,cv2.THRESH_BINARY)

gaussian_noise = (gaussian_noise*0.5).astype(np.uint8)
impulse_noise = impulse_noise.astype(np.uint8)

imnoise_gaussian = cv2.add(im_gray, gaussian_noise)
imnoise_impulse = cv2.add(im_gray, impulse_noise)


cv2.imwrite('../results/ex2c_gnoise.jpg', np.uint8(imnoise_gaussian))
cv2.imwrite('../results/ex2c_inoise.jpg', np.uint8(imnoise_impulse))


result_original_mf = median_filter_gray(im_gray, 5)
result_gaussian_mf = median_filter_gray(imnoise_gaussian, 5)
result_impulse_mf = median_filter_gray(imnoise_impulse, 5)

cv2.imwrite('../results/ex2c_original_median_5.jpg', np.uint8(result_original_mf))
cv2.imwrite('../results/ex2c_gnoise_median_5.jpg', np.uint8(imnoise_gaussian))
cv2.imwrite('../results/ex2c_inoise_median_5.jpg', np.uint8(result_impulse_mf))


result_original_mf = median_filter_gray(im_gray, 11)
result_gaussian_mf = median_filter_gray(imnoise_gaussian, 11)
result_impulse_mf = median_filter_gray(imnoise_impulse, 11)

cv2.imwrite('../results/ex2c_original_median_11.jpg', np.uint8(result_original_mf))
cv2.imwrite('../results/ex2c_gnoise_median_11.jpg', np.uint8(result_gaussian_mf))
cv2.imwrite('../results/ex2c_inoise_median_11.jpg', np.uint8(result_impulse_mf))

