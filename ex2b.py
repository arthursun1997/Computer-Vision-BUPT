# Bilateral filtering without OpenCV
import numpy as np
import cv2
import sys
import math


# --------------------------------- WRITE YOUR CODE HERE ---------------------------------#
def bilateral_filter_gray(image, w_size, int_std, spa_std):
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
    kernel = np.zeros((ker_len, ker_len), dtype=np.float)

    # convolve the image with given kernel
    # the pixel in the image
    for i in range(len(image)):
        for j in range(len(image[0])):
            # the point of kernel
            for k in range(-kl, kl + 1):
                for l in range(-kl, kl + 1):
                    kernel[k + kl, l + kl] = np.exp(-1 * ((k * k + l * l) / (2 * spa_std * spa_std) + (
                                im[i + kl, j + kl] - im[i + kl + k, j + kl + l]) ** 2 / (2 * int_std * int_std)))
            kernel = kernel / np.sum(kernel)
            image[i, j] = np.sum(im[i:i + ker_len, j:j + ker_len] * kernel)

    return image


##########################################################################################


im_gray = cv2.imread('../inputs/cat.png', 0)

result_bf1 = bilateral_filter_gray(im_gray, 10, 30.0, 3.0)
result_bf2 = bilateral_filter_gray(im_gray, 10, 30.0, 30.0)
result_bf3 = bilateral_filter_gray(im_gray, 10, 100.0, 3.0)
result_bf4 = bilateral_filter_gray(im_gray, 10, 100.0, 30.0)
result_bf5 = bilateral_filter_gray(im_gray, 5, 100.0, 30.0)

result_bf1 = np.uint8(result_bf1)
result_bf2 = np.uint8(result_bf2)
result_bf3 = np.uint8(result_bf3)
result_bf4 = np.uint8(result_bf4)
result_bf5 = np.uint8(result_bf5)

cv2.imwrite('../results/ex2b_bf_10_30_3.jpg', result_bf1)
cv2.imwrite('../results/ex2b_bf_10_30_30.jpg', result_bf2)
cv2.imwrite('../results/ex2b_bf_10_100_3.jpg', result_bf3)
cv2.imwrite('../results/ex2b_bf_10_100_30.jpg', result_bf4)
cv2.imwrite('../results/ex2b_bf_5_100_30.jpg', result_bf5)
