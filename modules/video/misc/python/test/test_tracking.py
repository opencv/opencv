#!/usr/bin/env python
import os
import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests, unittest

class tracking_test(NewOpenCVTests):

    def test_createMILTracker(self):
        t = cv.TrackerMIL.create()
        self.assertTrue(t is not None)

    def test_createGoturnTracker(self):
        proto = self.find_file("dnn/gsoc2016-goturn/goturn.prototxt");
        weights = self.find_file("dnn/gsoc2016-goturn/goturn.caffemodel");
        net = cv.dnn.readNet(proto, weights)
        t = cv.TrackerGOTURN.create(net)
        self.assertTrue(t is not None)

    def test_createNanoTracker(self):
        backbone_path = self.find_file("dnn/onnx/models/nanotrack_backbone_sim_v2.onnx");
        neckhead_path = self.find_file("dnn/onnx/models/nanotrack_head_sim_v2.onnx");
        backbone = cv.dnn.readNet(backbone_path)
        neckhead = cv.dnn.readNet(neckhead_path)
        t = cv.TrackerNano.create(backbone, neckhead)
        self.assertTrue(t is not None)

    def test_createVitTracker(self):
        model_path = self.find_file("dnn/onnx/models/vitTracker.onnx");
        model = cv.dnn.readNet(model_path)
        t = cv.TrackerVit.create(model)
        self.assertTrue(t is not None)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
