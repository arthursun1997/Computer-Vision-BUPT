# image classification
import numpy as np
import cv2


# --------------------------------- WRITE YOUR CODE HERE ---------------------------------#

# read data
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
dict = unpickle('../inputs/data_batch_1')
data = np.matrix(dict.get('data'.encode('utf-8')), dtype=np.float32)
labels = np.array(dict.get('labels'.encode('utf-8')), dtype=np.int)

# hog
hog = cv2.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 9)
data_hog = []
im = np.zeros([1024, 3], dtype=np.float32)
for i in data[0:2200]:
    i = np.asarray(i)[0]
    im[:, 0] = i[0:1024]
    im[:, 1] = i[1024:2048]
    im[:, 2] = i[2048:3072]
    im = im.reshape(32, 32, 3)
    im = im.astype(np.uint8)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h = hog.compute(gray).reshape(-1)
    data_hog.append(h)
    im = np.zeros([1024, 3], dtype=np.float32)
data_hog = np.matrix(data_hog, dtype=np.float32)

# svm para
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
# rgb
svm.train(data[0:2000], cv2.ml.ROW_SAMPLE, labels[0:2000])
response_rgb = svm.predict(data[2000:2200])
# HOG
svm.train(data_hog[0:2000], cv2.ml.ROW_SAMPLE, labels[0:2000])
response_gra = svm.predict(data_hog[2000:2200])

count = 0
count_rgb = 0
count_gra = 0
for i in range(200):
    # label 0 is airplane
    if labels[i + 2000] == 0:
        im = np.zeros([1024, 3], dtype=np.float32)
        arr = np.asarray(data[i + 2000])[0]
        im[:, 0] = arr[0:1024]
        im[:, 1] = arr[1024:2048]
        im[:, 2] = arr[2048:3072]
        im = im.reshape(32, 32, 3)
        im = im.astype(np.uint8)
        num = "{}".format(i)
        count += 1
        # rgb
        if response_rgb[1][i] == labels[i + 2000]:
            count_rgb += 1
            cv2.imwrite('../results/ex4b_rgb_True_' + num + '.png', im, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        else:
            cv2.imwrite('../results/ex4b_rgb_False_' + num + '.png', im, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        # HOG
        if response_gra[1][i] == labels[i + 2000]:
            count_gra += 1
            cv2.imwrite('../results/ex4b_gradient_True_' + num + '.png', im, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        else:
            cv2.imwrite('../results/ex4b_gradient_False_' + num + '.png', im, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

print("rgb: ", count_rgb / count, "gradient: ", count_gra / count)

##########################################################################################
