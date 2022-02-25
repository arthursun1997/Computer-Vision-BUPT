# Gaussian filtering without OpenCV
import numpy as np
import cv2


# --------------------------------- WRITE YOUR CODE HERE ---------------------------------#
def gaussian_filter_gray(image, w_size, std):

    height = len(image)
    width = len(image[0])

    ## start: calculate the gaussian kernel
    siz = round((w_size - 1) / 2)
    x, y = np.meshgrid(np.arange(-siz, siz + 1), np.arange(-siz, siz + 1))  # gaussian distribution exponent parameter array
    para = -(x * x + y * y) / (2 * std * std)  # the exponent parameter of gaussian
    kernel = np.exp(para)  # compute gaussian
    kernel = kernel / np.sum(kernel)  # normalize
    ## finish: calculate the gaussian kernel

    kl = int((len(kernel) - 1) / 2)

    ## start dealing margin
    im = np.zeros((height + kl * 2, width + kl * 2), dtype=np.int) # create an empty ndarray
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

    ker_len = len(kernel)

    image = np.zeros((height, width), dtype=np.float)

    # convolve the image with given kernel
    for i in range(len(image)):
        for j in range(len(image[0])):
            image[i, j] = np.sum(kernel * im[i: i + ker_len, j: j + ker_len])

    return image


##########################################################################################


im_gray = cv2.imread('../inputs/lena.jpg', 0)
im_gray = cv2.resize(im_gray, (256, 256))

result_gf1 = gaussian_filter_gray(im_gray, 5, 1.0)
result_gf2 = gaussian_filter_gray(im_gray, 5, 10.0)
result_gf3 = gaussian_filter_gray(im_gray, 10, 1.0)
result_gf4 = gaussian_filter_gray(im_gray, 10, 10.0)

result_gf1 = np.uint8(result_gf1)
result_gf2 = np.uint8(result_gf2)
result_gf3 = np.uint8(result_gf3)
result_gf4 = np.uint8(result_gf4)

cv2.imwrite('../results/ex2a_gf_5_1.jpg', result_gf1)
cv2.imwrite('../results/ex2a_gf_5_10.jpg', result_gf2)
cv2.imwrite('../results/ex2a_gf_10_1.jpg', result_gf3)
cv2.imwrite('../results/ex2a_gf_10_10.jpg', result_gf4)
