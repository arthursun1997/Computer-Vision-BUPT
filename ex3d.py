# K-Means Clustering
import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt

im = cv2.imread('../inputs/baboon.jpg')
#--------------------------------- WRITE YOUR CODE HERE ---------------------------------#
def my_kmeans_rgb(im, K):
    height = len(im)
    width = len(im[0])

    # reshape and format the input image
    im = im.reshape((-1, 3))
    im = np.float32(im)


    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS, 0, 1.0)
    ret, label, center = cv2.kmeans(im, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]

    # reshape the image
    res = res.reshape((height, width, 3))

    return res


def my_kmeans_rgbxy(im, K, siegma):
    height = len(im)
    width = len(im[0])

    # reshape and format the input image
    im = im.reshape((-1, 3))
    im = np.float32(im)

    # the array of position parameter
    height_arr = np.float32(range(0, height)) / siegma
    height_arr = np.array([height_arr] * width).reshape(-1)
    width_arr = np.float32(range(0, width)) / siegma
    width_arr = np.array([width_arr] * height).reshape(-1)

    # create the sample array
    im_pos = np.zeros((height * width, 5), dtype=np.dtype('float32'))
    im_pos[:, 0:3] = im
    im_pos[:, 3] = height_arr
    im_pos[:, 4] = width_arr

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS, 0, 1.0)
    ret, label, center = cv2.kmeans(im_pos, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]

    # remove position information and reshape the image
    res = res[:, 0:3]
    res = res.reshape((height, width, 3))

    return res

siegma = 1.2

result_rgb_2 = my_kmeans_rgb(im, 2)
result_rgbxy_2 = my_kmeans_rgbxy(im, 2, siegma)
result_rgb_4 = my_kmeans_rgb(im, 4)
result_rgbxy_4 = my_kmeans_rgbxy(im, 4, siegma)
result_rgb_8 = my_kmeans_rgb(im, 8)
result_rgbxy_8 = my_kmeans_rgbxy(im, 8, siegma)


##########################################################################################
cv2.imwrite('../results/ex3d_rgb_2.jpg', result_rgb_2)
cv2.imwrite('../results/ex3d_rgbxy_2.jpg', result_rgbxy_2)
cv2.imwrite('../results/ex3d_rgb_4.jpg', result_rgb_4)
cv2.imwrite('../results/ex3d_rgbxy_4.jpg', result_rgbxy_4)
cv2.imwrite('../results/ex3d_rgb_8.jpg', result_rgb_8)
cv2.imwrite('../results/ex3d_rgbxy_8.jpg', result_rgbxy_8)