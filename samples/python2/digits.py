import numpy as np
import cv2
import itertools as it

'''
from scipy.io import loadmat

m = loadmat('ex4data1.mat')
X = m['X'].reshape(-1, 20, 20)
X = np.transpose(X, (0, 2, 1))
img = np.vstack(map(np.hstack, X.reshape(-1, 100, 20, 20)))
img = np.uint8(np.clip(img, 0, 1)*255)
cv2.imwrite('digits.png', img)
'''

def unroll_responses(responses, class_n):
    sample_n = len(responses)
    new_responses = np.zeros((sample_n, class_n), np.float32)
    new_responses[np.arange(sample_n), responses] = 1
    return new_responses
    

SZ = 20
digits_img = cv2.imread('digits.png', 0)

h, w = digits_img.shape
digits = [np.hsplit(row, w/SZ) for row in np.vsplit(digits_img, h/SZ)]
digits = np.float32(digits).reshape(-1, SZ*SZ)
N = len(digits)
labels = np.repeat(np.arange(10), N/10)

shuffle = np.random.permutation(N)
train_n = int(0.9*N)

digits_train, digits_test = np.split(digits[shuffle], [train_n])
labels_train, labels_test = np.split(labels[shuffle], [train_n])

labels_train_unrolled = unroll_responses(labels_train, 10)

model = cv2.ANN_MLP()
layer_sizes = np.int32([SZ*SZ, 25, 10])
model.create(layer_sizes)
        
# CvANN_MLP_TrainParams::BACKPROP,0.001
params = dict( term_crit = (cv2.TERM_CRITERIA_COUNT, 300, 0.01),
               train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,
               bp_dw_scale = 0.001,
               bp_moment_scale = 0.0 )
print 'training...'
model.train(digits_train, labels_train_unrolled, None, params=params)
model.save('dig_nn.dat')
model.load('dig_nn.dat')

ret, resp = model.predict(digits_test)
resp = resp.argmax(-1)
error_mask = (resp == labels_test)
print error_mask.mean()

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return it.izip_longest(fillvalue=fillvalue, *args)

def mosaic(w, imgs):
    imgs = iter(imgs)
    img0 = imgs.next()
    pad = np.zeros_like(img0)
    imgs = it.chain([img0], imgs)
    rows = grouper(w, imgs, pad)
    return np.vstack(map(np.hstack, rows))

test_img = np.uint8(digits_test).reshape(-1, SZ, SZ)

def vis_resp(img, flag):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if not flag:
        img[...,:2] = 0
    return img

test_img = mosaic(25, it.starmap(vis_resp, it.izip(test_img, error_mask)))
cv2.imshow('test', test_img)
cv2.waitKey()







