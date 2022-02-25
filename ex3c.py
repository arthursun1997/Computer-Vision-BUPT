# Adaptive Thresholding
import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt


def adaptive_thres(input, n, b_value):
    # --------------------------------- WRITE YOUR CODE HERE ---------------------------------#
    height = len(input)
    width = len(input[0])
    ker_len = 2 * n + 1

    ## start dealing margin
    im = np.zeros((height + n * 2, width + n * 2), dtype=np.int)  # create an empty ndarray
    for i in range(n):
        for j in range(n):
            im[i, j] = input[n + 1 - j, n + 1 - i]  # up left
            im[i, width + n + j] = input[n + 1 - j, width - 1 - i]  # up right
            im[height + n + i, j] = input[height - 1 - j, n + 1 - i]  # down left
            im[height + n + i, width + n + j] = input[height - 1 - j, width - 1 - i]  # down right
    for i in range(n):
        im[i, n: width + n] = input[n + 1 - i, :]  # up
        im[height + n + i, n: width + n] = input[height - 1 - i, :]  # down
        im[n: height + n, i] = input[:, n + 1 - i]  # left
        im[n: height + n, width + n + i] = input[:, width - 1 - i]  # right
    im[n:height + n, n:width + n] = input[:, :]  # center
    ## finish dealing margin

    output = np.zeros((height, width), dtype=np.dtype('uint8'))

    # convolve the input with given kernel
    # the pixel in the input
    for i in range(len(input)):
        for j in range(len(input[0])):
            T = np.sum(im[i:i + ker_len, j:j + ker_len]) * b_value / ker_len ** 2
            if input[i, j] > T:
                output[i, j] = 255
            else:
                output[i, j] = 0

    return output


    ##########################################################################################



im = cv2.imread('../inputs/writing_ebu7240.png')
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

output = adaptive_thres(im_gray, 2, 0.4)
cv2.imwrite('../results/ex3c_thres_0.4.jpg', output)
output = adaptive_thres(im_gray, 2, 0.6)
cv2.imwrite('../results/ex3c_thres_0.6.jpg', output)
output = adaptive_thres(im_gray, 2, 0.8)
cv2.imwrite('../results/ex3c_thres_0.8.jpg', output)

