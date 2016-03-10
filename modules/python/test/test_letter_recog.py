#!/usr/bin/env python

'''
The sample demonstrates how to train Random Trees classifier
(or Boosting classifier, or MLP, or Knearest, or Support Vector Machines) using the provided dataset.

We use the sample database letter-recognition.data
from UCI Repository, here is the link:

Newman, D.J. & Hettich, S. & Blake, C.L. & Merz, C.J. (1998).
UCI Repository of machine learning databases
[http://www.ics.uci.edu/~mlearn/MLRepository.html].
Irvine, CA: University of California, Department of Information and Computer Science.

The dataset consists of 20000 feature vectors along with the
responses - capital latin letters A..Z.
The first 10000 samples are used for training
and the remaining 10000 - to test the classifier.
======================================================
  Models: RTrees, KNearest, Boost, SVM, MLP
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

def load_base(fn):
    a = np.loadtxt(fn, np.float32, delimiter=',', converters={ 0 : lambda ch : ord(ch)-ord('A') })
    samples, responses = a[:,1:], a[:,0]
    return samples, responses

class LetterStatModel(object):
    class_n = 26
    train_ratio = 0.5

    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

    def unroll_samples(self, samples):
        sample_n, var_n = samples.shape
        new_samples = np.zeros((sample_n * self.class_n, var_n+1), np.float32)
        new_samples[:,:-1] = np.repeat(samples, self.class_n, axis=0)
        new_samples[:,-1] = np.tile(np.arange(self.class_n), sample_n)
        return new_samples

    def unroll_responses(self, responses):
        sample_n = len(responses)
        new_responses = np.zeros(sample_n*self.class_n, np.int32)
        resp_idx = np.int32( responses + np.arange(sample_n)*self.class_n )
        new_responses[resp_idx] = 1
        return new_responses

class RTrees(LetterStatModel):
    def __init__(self):
        self.model = cv2.RTrees()

    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        params = dict(max_depth=20 )
        self.model.train(samples, cv2.CV_ROW_SAMPLE, responses.astype(int), params = params)

    def predict(self, samples):
        return np.float32( [self.model.predict(s) for s in samples] )


class KNearest(LetterStatModel):
    def __init__(self):
        self.model = cv2.KNearest()

    def train(self, samples, responses):
        self.model.train(samples, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.find_nearest(samples, k = 10)
        return results.ravel()


class Boost(LetterStatModel):
    def __init__(self):
        self.model = cv2.Boost()

    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        new_samples = self.unroll_samples(samples)
        new_responses = self.unroll_responses(responses)
        var_types = np.array([cv2.CV_VAR_NUMERICAL] * var_n + [cv2.CV_VAR_CATEGORICAL, cv2.CV_VAR_CATEGORICAL], np.uint8)
        params = dict(max_depth=10, weak_count=15)
        self.model.train(new_samples, cv2.CV_ROW_SAMPLE, new_responses.astype(int), varType = var_types, params=params)

    def predict(self, samples):
        new_samples = self.unroll_samples(samples)
        pred = np.array( [self.model.predict(s) for s in new_samples] )
        pred = pred.reshape(-1, self.class_n).argmax(1)
        return pred


class SVM(LetterStatModel):
    def __init__(self):
        self.model = cv2.SVM()

    def train(self, samples, responses):
        params = dict( kernel_type = cv2.SVM_RBF,
                       svm_type = cv2.SVM_C_SVC,
                       C = 1,
                       gamma = .1 )
        self.model.train(samples, responses.astype(int), params = params)

    def predict(self, samples):
        return self.model.predict_all(samples).ravel()


class MLP(LetterStatModel):
    def __init__(self):
        self.model = cv2.ANN_MLP()

    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        new_responses = self.unroll_responses(responses).reshape(-1, self.class_n)
        layer_sizes = np.int32([var_n, 100, 100, self.class_n])

        self.model.create(layer_sizes, cv2.ANN_MLP_SIGMOID_SYM, 2, 1)
        params = dict( train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,
                       bp_moment_scale = 0.0,
                       bp_dw_scale = 0.001,
                       term_crit = (cv2.TERM_CRITERIA_COUNT, 20, 0.01) )
        self.model.train(samples, np.float32(new_responses), None, params = params)

    def predict(self, samples):
        ret, resp = self.model.predict(samples)
        return resp.argmax(-1)

from tests_common import NewOpenCVTests

class letter_recog_test(NewOpenCVTests):

    def test_letter_recog(self):

        eps = 0.01

        models = [RTrees, KNearest, Boost, SVM, MLP]
        models = dict( [(cls.__name__.lower(), cls) for cls in models] )
        testErrors = {RTrees: (98.930000, 92.390000), KNearest: (94.960000, 92.010000),
         Boost: (85.970000, 74.920000), SVM: (99.780000, 95.680000), MLP: (90.060000, 87.410000)}

        for model in models:
            Model = models[model]
            classifier = Model()

            samples, responses = load_base(self.repoPath + '/samples/cpp/letter-recognition.data')
            train_n = int(len(samples)*classifier.train_ratio)

            classifier.train(samples[:train_n], responses[:train_n])
            train_rate = np.mean(classifier.predict(samples[:train_n]) == responses[:train_n].astype(int))
            test_rate  = np.mean(classifier.predict(samples[train_n:]) == responses[train_n:].astype(int))

            self.assertLess(train_rate - testErrors[Model][0], eps)
            self.assertLess(test_rate - testErrors[Model][1], eps)