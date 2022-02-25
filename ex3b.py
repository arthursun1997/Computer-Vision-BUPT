import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt

im1 = cv2.imread('../inputs/building.jpg')
#--------------------------------- WRITE YOUR CODE HERE ---------------------------------#
im_result = im1

# find edges using canny edge detector
edges_1 = cv2.Canny(im1, 50, 200, 3)

# detect lines points
lines_1 = cv2.HoughLinesP(edges_1, 1, np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)

# Draw lines on the image
for line in lines_1:
    x1, y1, x2, y2 = line[0]
    cv2.line(im_result, (x1, y1), (x2, y2), (0, 0, 255), 2, 8)


# YOUR_OWN.jpg
im2 = cv2.imread('../inputs/lanes.jpg')
im_res = im2

# find edges using canny edge detector
edges_2 = cv2.Canny(im2, 50, 200, 3)

# detect lines points
lines_2 = cv2.HoughLinesP(edges_2, 1, np.pi / 180, threshold=80, minLineLength=10, maxLineGap=200)

# Draw lines on the image
for line in lines_2:
    x1, y1, x2, y2 = line[0]
    cv2.line(im_res, (x1, y1), (x2, y2), (0, 0, 255), 2, 8)

cv2.imwrite('../results/ex3b_lanes_hough.jpg', im_res)

##########################################################################################

cv2.imwrite('../results/ex3b_building_hough.jpg', im_result)
