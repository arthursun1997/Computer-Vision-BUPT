# Image stitching using affine transform
import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt

im1 = cv2.imread('../inputs/Img01.jpg')
im2 = cv2.imread('../inputs/Img02.jpg')


im_gray1 = cv2.imread('../inputs/Img01.jpg', 0)
im_gray2 = cv2.imread('../inputs/Img02.jpg', 0)

#--------------------------------- WRITE YOUR CODE HERE ---------------------------------#
sift = cv2.SIFT_create()
kp_1, des_1 = sift.detectAndCompute(im_gray1, None)

sift = cv2.SIFT_create()
kp_2, des_2 = sift.detectAndCompute(im_gray2, None)

# BFMatcher with default params with using threshold ratio
bf = cv2.BFMatcher()
matches = bf.knnMatch(des_2, des_1, k=2)

# Refine matched points using threshold ratio of nearest to 2nd nearest descriptor
threshold = 0.75  # best performance in Low's study
true_matches = []
for first_near, second_near in matches:
    if first_near.distance < threshold * second_near.distance:
        true_matches.append([first_near])

# transport the true_matches list to an array
true_matches = np.asarray(true_matches)

# store the corresponding matched pairs of keypoint into src_pt and des_pt respectively, each keypoint is a (1, 2) array
src_pt = np.asarray([kp_2[m.queryIdx].pt for m in true_matches[:, 0]])
des_pt = np.asarray([kp_1[m.trainIdx].pt for m in true_matches[:, 0]])

# find the homography matrix (requires at least 4 matched pairs to performance the transformation)
H_no, masked_no = cv2.findHomography(src_pt, des_pt, 0)
H, masked = cv2.findHomography(src_pt, des_pt, cv2.RANSAC)

# affine transform
panorama_noRANSAC = cv2.warpPerspective(im2, H_no, (im1.shape[1] + im2.shape[1], im1.shape[0]))
panorama_RANSAC = cv2.warpPerspective(im2, H, (im1.shape[1] + im2.shape[1], im1.shape[0]))

# stitch images
panorama_noRANSAC[:, 0:im1.shape[1], :] = im1
panorama_RANSAC[:, 0:im1.shape[1], :] = im1

##########################################################################################

cv2.imwrite('../results/ex3a_stitched_noRANSAC.jpg', panorama_noRANSAC)
cv2.imwrite('../results/ex3a_stitched_RANSAC.jpg', panorama_RANSAC)