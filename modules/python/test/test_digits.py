#!/usr/bin/env python

'''
SVM and KNearest digit recognition.

Sample loads a dataset of handwritten digits from '../data/digits.png'.
Then it trains a SVM and KNearest classifiers on it and evaluates
their accuracy.

Following preprocessing is applied to the dataset:
 - Moment-based image deskew (see deskew())
 - Digit images are split into 4 10x10 cells and 16-bin
   histogram of oriented gradients is computed for each
   cell
 - Transform histograms to space with Hellinger metric (see [1] (RootSIFT))


[1] R. Arandjelovic, A. Zisserman
    "Three things everyone should know to improve object retrieval"
    http://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf

'''


# Python 2/3 compatibility
from __future__ import print_function

# built-in modules
from multiprocessing.pool import ThreadPool

import cv2

import numpy as np
from numpy.linalg import norm


SZ = 20 # size of each digit is SZ x SZ
CLASS_N = 10
DIGITS_FN = 'samples/python2/data/digits.png'

def split2d(img, cell_size, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  # Known bug: https://github.com/Itseez/opencv/issues/4969
    def save(self, fn):
        self.model.save(fn)

class KNearest(StatModel):
    def __init__(self, k = 3):
        self.k = k
        self.model = cv2.KNearest()

    def train(self, samples, responses):
        self.model.train(samples, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.find_nearest(samples, self.k)
        return results.ravel()

class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.params = dict( kernel_type = cv2.SVM_RBF,
                            svm_type = cv2.SVM_C_SVC,
                            C = C,
                            gamma = gamma )
        self.model = cv2.SVM()

    def train(self, samples, responses):
        self.model.train(samples, responses, params = self.params)

    def predict(self, samples):
        return self.model.predict_all(samples).ravel()


def evaluate_model(model, digits, samples, labels):
    resp = model.predict(samples)
    err = (labels != resp).mean()

    confusion = np.zeros((10, 10), np.int32)
    for i, j in zip(labels, resp):
        confusion[int(i), int(j)] += 1

    return err, confusion

def preprocess_simple(digits):
    return np.float32(digits).reshape(-1, SZ*SZ) / 255.0

def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)

from tests_common import NewOpenCVTests

class digits_test(NewOpenCVTests):

    def load_digits(self, fn):
        digits_img = self.get_sample(fn, 0)
        digits = split2d(digits_img, (SZ, SZ))
        labels = np.repeat(np.arange(CLASS_N), len(digits)/CLASS_N)
        return digits, labels

    def test_digits(self):

        digits, labels = self.load_digits(DIGITS_FN)

        # shuffle digits
        rand = np.random.RandomState(321)
        shuffle = rand.permutation(len(digits))
        digits, labels = digits[shuffle], labels[shuffle]

        digits2 = list(map(deskew, digits))
        samples = preprocess_hog(digits2)

        train_n = int(0.9*len(samples))
        digits_train, digits_test = np.split(digits2, [train_n])
        samples_train, samples_test = np.split(samples, [train_n])
        labels_train, labels_test = np.split(labels, [train_n])
        errors = list()
        confusionMatrixes = list()

        model = KNearest(k=4)
        model.train(samples_train, labels_train)
        error, confusion = evaluate_model(model, digits_test, samples_test, labels_test)
        errors.append(error)
        confusionMatrixes.append(confusion)

        model = SVM(C=2.67, gamma=5.383)
        model.train(samples_train, labels_train)
        error, confusion = evaluate_model(model, digits_test, samples_test, labels_test)
        errors.append(error)
        confusionMatrixes.append(confusion)

        eps = 0.001
        normEps = len(samples_test) * 0.02

        confusionKNN = [[45,  0,  0,  0,  0,  0,  0,  0,  0,  0],
         [ 0, 57,  0,  0,  0,  0,  0,  0,  0,  0],
         [ 0,  0, 59,  1,  0,  0,  0,  0,  1,  0],
         [ 0,  0,  0, 43,  0,  0,  0,  1,  0,  0],
         [ 0,  0,  0,  0, 38,  0,  2,  0,  0,  0],
         [ 0,  0,  0,  2,  0, 48,  0,  0,  1,  0],
         [ 0,  1,  0,  0,  0,  0, 51,  0,  0,  0],
         [ 0,  0,  1,  0,  0,  0,  0, 54,  0,  0],
         [ 0,  0,  0,  0,  0,  1,  0,  0, 46,  0],
         [ 1,  1,  0,  1,  1,  0,  0,  0,  2, 42]]

        confusionSVM = [[45,  0,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0, 57,  0,  0,  0,  0,  0,  0,  0,  0],
          [ 0,  0, 59,  2,  0,  0,  0,  0,  0,  0],
          [ 0,  0,  0, 43,  0,  0,  0,  1,  0,  0],
          [ 0,  0,  0,  0, 40,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  1,  0, 50,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  1,  0,  51, 0,  0,  0],
          [ 0,  0,  1,  0,  0,  0,  0,  54, 0,  0],
          [ 0,  0,  0,  0,  0,  0,  0,  0, 47,  0],
          [ 0,  1,  0,  1,  0,  0,  0,  0,  1, 45]]

        self.assertLess(cv2.norm(confusionMatrixes[0] - confusionKNN, cv2.NORM_L1), normEps)
        self.assertLess(cv2.norm(confusionMatrixes[1] - confusionSVM, cv2.NORM_L1), normEps)

        self.assertLess(errors[0] - 0.034, eps)
        self.assertLess(errors[1] - 0.018, eps)