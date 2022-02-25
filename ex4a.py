# Object tracking with your image

import numpy as np
import cv2

# --------------------------------- WRITE YOUR CODE HERE ---------------------------------#
size = (360, 640)
fps = 30


def paper_func(r, c, h, w):
    cap = cv2.VideoCapture('../inputs/ebu7240_hand.mp4')

    i = 0
    has_next, frame = cap.read()

    # setup initial location of window, the region of interest
    window = (c, r, w, h)

    # set up the ROI for tracking
    roi = frame[r:r + h, c:c + w]  # region of interest
    # cv2.imshow("", roi)
    # cv2.waitKey()
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # convert roi color space from rgb into hsv

    mask = cv2.inRange(hsv_roi, np.array((0., 0., 0.)),
                       np.array((180., 255., 255.)))  # set the mask the pixel to be ignore by hsv
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])  # the histogram of roi
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)  # normalize the histogram of roi

    # Setup the termination criteria, either 10 iteration or move by at least 10 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 0, 50)

    img_arr = []  # list to store the image

    while has_next:

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # convert this frame color space from rgb into hsv
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)  # backproject this frame into a possibility of roi

        # apply meanshift to get the new location
        ret, window = cv2.meanShift(dst, window, term_crit)

        # Draw it on image
        x, y, w, h = window
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        img_arr.append(frame)

        # store particular frame
        # if i == 0:
        #     cv2.imwrite('../results/ex4a_p_f1.png', frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        # elif i == 19:
        #     cv2.imwrite('../results/ex4a_p_f20.png', frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        # elif i == 39:
        #     cv2.imwrite('../results/ex4a_p_f40.png', frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        # elif i == 59:
        #     cv2.imwrite('../results/ex4a_p_f60.png', frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        # elif i == 89:
        #     cv2.imwrite('../results/ex4a_p_f90.png', frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

        has_next, frame = cap.read()
        i += 1

    cap.release()
    return img_arr


def id_func(r, c, h, w):
    cap = cv2.VideoCapture('../inputs/ebu7240_hand.mp4')

    i = 0
    has_next, frame = cap.read()

    # setup initial location of window, the region of interest
    window = (c, r, w, h)

    # set up the ROI for tracking
    roi = frame[r:r + h, c:c + w]  # region of interest
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # convert roi color space from rgb into hsv

    mask = cv2.inRange(hsv_roi, np.array((0., 0., 0.)),
                       np.array((180., 255., 100.)))  # set the mask the pixel to be ignore by hsv
    roi_hist = cv2.calcHist([hsv_roi], [2], mask, [255], [0, 255])  # the histogram of roi
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)  # normalize the histogram of roi

    # Setup the termination criteria, either 10 iteration or move by at least 10 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 0, 50)

    img_arr = []  # list to store the image

    while has_next:

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # convert this frame color space from rgb into hsv
        dst = cv2.calcBackProject([hsv], [2], roi_hist, [0, 255], 1)  # backproject this frame into a possibility of roi

        # apply meanshift to get the new location
        ret, window = cv2.meanShift(dst, window, term_crit)

        # Draw it on image
        x, y, w, h = window
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        img_arr.append(frame)

        # store particular frame
        # if i == 0:
        #     cv2.imwrite('../results/ex4a_i_f1.png', frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        # elif i == 19:
        #     cv2.imwrite('../results/ex4a_i_f20.png', frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        # elif i == 39:
        #     cv2.imwrite('../results/ex4a_i_f40.png', frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        # elif i == 59:
        #     cv2.imwrite('../results/ex4a_i_f60.png', frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        # elif i == 89:
        #     cv2.imwrite('../results/ex4a_i_f90.png', frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

        has_next, frame = cap.read()
        i += 1

    cap.release()
    return img_arr


# paper
c, r, w, h = 39, 229, 263, 183  # window position: row, column, height, width
paper_arr = paper_func(r, c, h, w)

# id
c, r, w, h = 95, 325, 152, 22  # window position: row, column, height, width
id_arr = id_func(r, c, h, w)

##########################################################################################


out_paper = cv2.VideoWriter('../results/ex4a_meanshift_track_paper.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
out_id = cv2.VideoWriter('../results/ex4a_meanshift_track_id.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

for i in range(len(paper_arr)):
    out_paper.write(paper_arr[i])
out_paper.release()

for i in range(len(id_arr)):
    out_id.write(id_arr[i])
out_id.release()
