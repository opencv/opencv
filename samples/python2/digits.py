'''
SVN and KNearest digit recognition.

Sample loads a dataset of handwritten digits from 'digits.png'.
Then it trains a SVN and KNearest classifiers on it and evaluates
their accuracy. Moment-based image deskew is used to improve 
the recognition accuracy.

Usage:
   digits.py
'''

import numpy as np
import cv2
from multiprocessing.pool import ThreadPool
from common import clock, mosaic

SZ = 20 # size of each digit is SZ x SZ
CLASS_N = 10

def load_digits(fn):
    print 'loading "%s" ...' % fn
    digits_img = cv2.imread(fn, 0)
    h, w = digits_img.shape
    digits = [np.hsplit(row, w/SZ) for row in np.vsplit(digits_img, h/SZ)]
    digits = np.array(digits).reshape(-1, SZ, SZ)
    labels = np.repeat(np.arange(CLASS_N), len(digits)/CLASS_N)
    return digits, labels

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
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class KNearest(StatModel):
    def __init__(self, k = 3):
        self.k = k
        self.model = cv2.KNearest()

    def train(self, samples, responses):
        self.model = cv2.KNearest()
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
        self.model = cv2.SVM()
        self.model.train(samples, responses, params = self.params)

    def predict(self, samples):
        return self.model.predict_all(samples).ravel()


def evaluate_model(model, digits, samples, labels):
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print 'error: %.2f %%' % (err*100)
    
    confusion = np.zeros((10, 10), np.int32)
    for i, j in zip(labels, resp):
        confusion[i, j] += 1
    print 'confusion matrix:'
    print confusion
    print

    vis = []
    for img, flag in zip(digits, resp == labels):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[...,:2] = 0
        vis.append(img)
    return mosaic(25, vis)


if __name__ == '__main__':
    print __doc__

    digits, labels = load_digits('digits.png')
    
    print 'preprocessing...'
    # shuffle digits
    rand = np.random.RandomState(12345)
    shuffle = rand.permutation(len(digits))
    digits, labels = digits[shuffle], labels[shuffle]
    
    digits2 = map(deskew, digits)
    samples = np.float32(digits2).reshape(-1, SZ*SZ) / 255.0

    train_n = int(0.9*len(samples))
    cv2.imshow('test set', mosaic(25, digits[train_n:]))
    digits_train, digits_test = np.split(digits2, [train_n])
    samples_train, samples_test = np.split(samples, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])

    
    print 'training KNearest...'
    model = KNearest(k=1)
    model.train(samples_train, labels_train)
    vis = evaluate_model(model, digits_test, samples_test, labels_test)
    cv2.imshow('KNearest test', vis)

    print 'training SVM...'
    model = SVM(C=4.66, gamma=0.08)
    model.train(samples_train, labels_train)
    vis = evaluate_model(model, digits_test, samples_test, labels_test)
    cv2.imshow('SVM test', vis)
    print 'saving SVM as "digits_svm.dat"...'
    model.save('digits_svm.dat')

    cv2.waitKey(0)
