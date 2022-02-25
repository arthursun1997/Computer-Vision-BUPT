# SIFT matching using OpenCV
import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt

im_gray1 = cv2.imread('../inputs/sift_input1.jpg', 0)
im_gray2 = cv2.imread('../inputs/sift_input2.jpg', 0)

# --------------------------------- WRITE YOUR CODE HERE ---------------------------------#
sift = cv2.SIFT_create()
kp_1, des_1 = sift.detectAndCompute(im_gray1, None)
img_sift_kp_1 = cv2.drawKeypoints(im_gray1, kp_1, im_gray1)

sift = cv2.SIFT_create()
kp_2, des_2 = sift.detectAndCompute(im_gray2, None)
img_sift_kp_2 = cv2.drawKeypoints(im_gray2, kp_2, im_gray2)

# BFMatcher with default params with using threshold ratio
bf = cv2.BFMatcher()
matches = bf.knnMatch(des_1, des_2, k=2)

# Refine matched points using threshold ratio of nearest to 2nd nearest descriptor
threshold = 0.75  # best performance in Low's study
true_matches = []
for first_near, second_near in matches:
    if first_near.distance < threshold * second_near.distance:
        true_matches.append([first_near])

# sort the true_matches list with the increase order of distance
true_matches.sort(key=lambda m: m[0].distance)

img_least50 = cv2.drawMatchesKnn(im_gray1, kp_1, im_gray2, kp_2, true_matches[len(true_matches) - 50: len(true_matches)], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_most50 = cv2.drawMatchesKnn(im_gray1, kp_1, im_gray2, kp_2, true_matches[0:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


##########################################################################################

# Keypoint maps
cv2.imwrite('../results/ex2d_sift_input1.jpg', np.uint8(img_sift_kp_1))
cv2.imwrite('../results/ex2d_sift_input2.jpg', np.uint8(img_sift_kp_2))

# Feature Matching outputs
cv2.imwrite('../results/ex2d_matches_least50.jpg', np.uint8(img_least50))
cv2.imwrite('../results/ex2d_matches_most50.jpg', np.uint8(img_most50))
