#!/usr/bin/env python
import os
import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests, unittest

class tracking_test(NewOpenCVTests):

    def test_createMILTracker(self):
        t = cv.TrackerMIL.create()
        self.assertTrue(t is not None)

    def test_createNanoTracker(self):
        backbone_path = self.find_file("dnn/onnx/models/nanotrack_backbone_sim_v2.onnx", required=False);
        neckhead_path = self.find_file("dnn/onnx/models/nanotrack_head_sim_v2.onnx", required=False);
        backbone = cv.dnn.readNet(backbone_path)
        neckhead = cv.dnn.readNet(neckhead_path)
        t = cv.TrackerNano.create(backbone, neckhead)
        self.assertTrue(t is not None)

    def test_createVitTracker(self):
        model_path = self.find_file("dnn/onnx/models/vitTracker.onnx", required=False);
        model = cv.dnn.readNet(model_path)
        t = cv.TrackerVit.create(model)
        self.assertTrue(t is not None)

    def test_createMultiTracker(self):
        t = cv.MultiTracker.create()
        self.assertTrue(t is not None)

    def test_multiTrackerParams(self):
        p = cv.MultiTracker_Params()
        self.assertEqual(p.minHits, 3)
        self.assertEqual(p.maxAge, 30)
        self.assertTrue(p.classAware)
        p.minHits = 1
        t = cv.MultiTracker.create(p)
        self.assertTrue(t is not None)

    def test_multiTrackerTracksAnObject(self):
        p = cv.MultiTracker_Params()
        p.minHits = 2
        t = cv.MultiTracker.create(p)

        boxes = [(100.0, 100.0, 40.0, 80.0)]
        scores = [0.9]
        classes = [0]

        # withheld until seen minHits times, then reported with a stable id
        ids, _, _ = t.update(boxes, scores, classes)
        self.assertEqual(len(ids), 0)
        ids, out_boxes, out_classes = t.update(boxes, scores, classes)
        self.assertEqual(len(ids), 1)
        self.assertEqual(len(out_boxes), 1)
        self.assertEqual(len(out_classes), 1)

        first = ids[0]
        for _ in range(3):
            ids, _, _ = t.update(boxes, scores, classes)
        self.assertEqual(list(ids), [first])

        t.reset()
        ids, _, _ = t.update([], [], [])
        self.assertEqual(len(ids), 0)

    def test_multiTrackerEmbeddingOverload(self):
        p = cv.MultiTracker_Params()
        p.minHits = 1
        p.embeddingWeight = 0.5
        t = cv.MultiTracker.create(p)

        boxes = [(100.0, 100.0, 40.0, 80.0)]
        scores = [0.9]
        classes = [0]
        emb = np.array([[1, 0, 0, 0]], dtype=np.float32)

        ids, _, _ = t.update(boxes, scores, classes, emb)
        self.assertEqual(len(ids), 1)
        first = ids[0]
        ids, _, _ = t.update(boxes, scores, classes, emb)
        self.assertEqual(list(ids), [first])


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
