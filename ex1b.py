import cv2


cap = cv2.VideoCapture('../inputs/ebu7240_hand.mp4')

img_array = []

if (cap.isOpened() == False):
    print("Error opening video stream or file")

im_myname = cv2.imread('../inputs/my_name.png')

#--------------------------------- WRITE YOUR CODE HERE ---------------------------------#

size = (360, 640)
fps = 30

i = 0

while True:
    hasNext, frame = cap.read()

    if hasNext:
        # print(i)
        pos = i * 2
        frame[550:640, pos:pos+180] = im_myname

        img_array.append(frame)

        # store particular frame
        # if i == 0:
        #     cv2.imwrite('../results/ex1_a_hand_composition_f1.png', frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        # elif i == 20:
        #     cv2.imwrite('../results/ex1_a_hand_composition_f21.png', frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        # elif i == 30:
        #     cv2.imwrite('../results/ex1_a_hand_composition_f31.png', frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        # elif i == 60:
        #     cv2.imwrite('../results/ex1_a_hand_composition_f61.png', frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        # elif i == 89:
        #     cv2.imwrite('../results/ex1_a_hand_composition_f90.png', frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

        i += 1

    else:
        print('end')
        break

##########################################################################################


out = cv2.VideoWriter('../results/ex1_b_hand_composition.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
