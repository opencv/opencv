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
        self.model = cv2.ml.RTrees_create()

    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        self.model.setMaxDepth(20)
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses.astype(int))

    def predict(self, samples):
        ret, resp = self.model.predict(samples)
        return resp.ravel()


class KNearest(LetterStatModel):
    def __init__(self):
        self.model = cv2.ml.KNearest_create()

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.findNearest(samples, k = 10)
        return results.ravel()


class Boost(LetterStatModel):
    def __init__(self):
        self.model = cv2.ml.Boost_create()

    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        new_samples = self.unroll_samples(samples)
        new_responses = self.unroll_responses(responses)
        var_types = np.array([cv2.ml.VAR_NUMERICAL] * var_n + [cv2.ml.VAR_CATEGORICAL, cv2.ml.VAR_CATEGORICAL], np.uint8)

        self.model.setWeakCount(15)
        self.model.setMaxDepth(10)
        self.model.train(cv2.ml.TrainData_create(new_samples, cv2.ml.ROW_SAMPLE, new_responses.astype(int), varType = var_types))

    def predict(self, samples):
        new_samples = self.unroll_samples(samples)
        ret, resp = self.model.predict(new_samples)

        return resp.ravel().reshape(-1, self.class_n).argmax(1)


class SVM(LetterStatModel):
    def __init__(self):
        self.model = cv2.ml.SVM_create()

    def train(self, samples, responses):
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setC(1)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setGamma(.1)
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses.astype(int))

    def predict(self, samples):
        ret, resp = self.model.predict(samples)
        return resp.ravel()


class MLP(LetterStatModel):
    def __init__(self):
        self.model = cv2.ml.ANN_MLP_create()

    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        new_responses = self.unroll_responses(responses).reshape(-1, self.class_n)
        layer_sizes = np.int32([var_n, 100, 100, self.class_n])

        self.model.setLayerSizes(layer_sizes)
        self.model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
        self.model.setBackpropMomentumScale(0)
        self.model.setBackpropWeightScale(0.001)
        self.model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 20, 0.01))
        self.model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)

        self.model.train(samples, cv2.ml.ROW_SAMPLE, np.float32(new_responses))

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

            samples, responses = load_base(self.repoPath + '/samples/data/letter-recognition.data')
            train_n = int(len(samples)*classifier.train_ratio)

            classifier.train(samples[:train_n], responses[:train_n])
            train_rate = np.mean(classifier.predict(samples[:train_n]) == responses[:train_n].astype(int))
            test_rate  = np.mean(classifier.predict(samples[train_n:]) == responses[train_n:].astype(int))

            self.assertLess(train_rate - testErrors[Model][0], eps)
            self.assertLess(test_rate - testErrors[Model][1], eps)