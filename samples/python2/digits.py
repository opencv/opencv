'''
Neural network digit recognition sample.
Usage:
   digits.py

   Sample loads a dataset of handwritten digits from 'digits.png'.
   Then it trains a neural network classifier on it and evaluates
   its classification accuracy.
'''

import numpy as np
import cv2
from common import mosaic

def unroll_responses(responses, class_n):
    '''[1, 0, 2, ...] -> [[0, 1, 0], [1, 0, 0], [0, 0, 1], ...]'''
    sample_n = len(responses)
    new_responses = np.zeros((sample_n, class_n), np.float32)
    new_responses[np.arange(sample_n), responses] = 1
    return new_responses
    

SZ = 20 # size of each digit is SZ x SZ
CLASS_N = 10
digits_img = cv2.imread('digits.png', 0)

# prepare dataset
h, w = digits_img.shape
digits = [np.hsplit(row, w/SZ) for row in np.vsplit(digits_img, h/SZ)]
digits = np.float32(digits).reshape(-1, SZ*SZ)
N = len(digits)
labels = np.repeat(np.arange(CLASS_N), N/CLASS_N)

# split it onto train and test subsets
shuffle = np.random.permutation(N)
train_n = int(0.9*N)
digits_train, digits_test = np.split(digits[shuffle], [train_n])
labels_train, labels_test = np.split(labels[shuffle], [train_n])

# train model
model = cv2.ANN_MLP()
layer_sizes = np.int32([SZ*SZ, 25, CLASS_N])
model.create(layer_sizes)
params = dict( term_crit = (cv2.TERM_CRITERIA_COUNT, 100, 0.01),
               train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,
               bp_dw_scale = 0.001,
               bp_moment_scale = 0.0 )
print 'training...'
labels_train_unrolled = unroll_responses(labels_train, CLASS_N)
model.train(digits_train, labels_train_unrolled, None, params=params)
model.save('dig_nn.dat')
model.load('dig_nn.dat')

def evaluate(model, samples, labels):
    '''Evaluates classifier preformance on a given labeled samples set.'''
    ret, resp = model.predict(samples)
    resp = resp.argmax(-1)
    error_mask = (resp == labels)
    accuracy = error_mask.mean()
    return accuracy, error_mask

# evaluate model
train_accuracy, _ = evaluate(model, digits_train, labels_train)
print 'train accuracy: ', train_accuracy
test_accuracy, test_error_mask = evaluate(model, digits_test, labels_test)
print 'test accuracy: ', test_accuracy

# visualize test results
vis = []
for img, flag in zip(digits_test, test_error_mask):
    img = np.uint8(img).reshape(SZ, SZ)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if not flag:
        img[...,:2] = 0
    vis.append(img)
vis = mosaic(25, vis)
cv2.imshow('test', vis)
cv2.waitKey()
