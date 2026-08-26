#!/usr/bin/env python
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
# Copyright (C) 2026, BigVision LLC, all rights reserved.
# Third party copyrights are property of their respective owners.

from __future__ import print_function
import os
import re
import glob

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests, unittest
from test_onnx_conformance_list import CONFORMANCE_TESTS, cpp_denylisted

# Mirrors modules/dnn/test/test_onnx_conformance.cpp through cv2.dnn bindings, since that C++ suite never calls into Python and so can't catch binding-layer regressions.
# The case list and the denylist lookup live in test_onnx_conformance_list.py.
# OpenCV/CPU only: the bindings are backend-agnostic, so iterating targets would only re-test C++ math.

TOLERANCE_OVERRIDES = {
    "test_attention_4d_fp16": (0.0002, 0.001),
    "test_attention_4d_fp16_expanded": (0.0002, 0.001),
    "test_attention_4d_gqa_with_past_and_present_fp16": (0.0002, 0.001),
    "test_attention_4d_gqa_with_past_and_present_fp16_expanded": (0.0002, 0.001),
    "test_causal_conv_with_state_fp16": (0.0002, 0.002),
    "test_causal_conv_with_state_silu_fp16": (0.0002, 0.002),
    "test_flexattention_fp16": (0.0002, 0.001),
    "test_flexattention_fp16_expanded_ver26": (0.0002, 0.001),
    "test_gelu_tanh_1": (0.00011, 0.00016),
    "test_gelu_tanh_2": (9e-05, 0.0005),
    "test_linear_attention_fp16": (0.0002, 0.001),
    "test_linear_attention_fp16_expanded": (0.0002, 0.001),
    "test_nllloss_NCd1d2_reduction_sum_expanded": (2e-05, 0.0001),
    "test_nllloss_NCd1d2d3d4d5_mean_weight_expanded": (2e-05, 0.0001),
    "test_reduce_prod_default_axes_keepdims_random": (0.002, 0.002),
    "test_reduce_sum_square_default_axes_keepdims_random": (2e-05, 0.0001),
    "test_reduce_sum_square_default_axes_keepdims_random_expanded": (2e-05, 0.0001),
    "test_roialign_aligned_false": (3e-05, 0.0001),
    "test_roialign_aligned_true": (3e-05, 0.0001),
}

L1_DEFAULT = 1e-5
LINF_DEFAULT = 1e-4

def _pbIndex(path):
    """Sort key for input_N.pb / output_N.pb, matching the C++ test's extractIndex."""
    m = re.search(r'_(\d+)\.pb$', os.path.basename(path))
    return int(m.group(1)) if m else 0


def normAssert(test, ref, actual, msg="", l1=L1_DEFAULT, lInf=LINF_DEFAULT):
    """Python port of cv::dnn normAssert (test_common.impl.hpp): FP16->FP32 upcast, scalar/1-element equivalence, exact NaN/Inf matching, then L1/Linf tolerance."""
    ref = np.asarray(ref)
    actual = np.asarray(actual)

    if ref.dtype == np.float16:
        ref = ref.astype(np.float32)
    if actual.dtype == np.float16:
        actual = actual.astype(np.float32)

    if ref.ndim == 0 and actual.shape == (1,):
        ref = ref.reshape(1)
    elif actual.ndim == 0 and ref.shape == (1,):
        actual = actual.reshape(1)

    if ref.size == 0 or actual.size == 0:
        test.assertEqual(ref.size, actual.size, msg)
        test.assertEqual(ref.shape, actual.shape, msg)
        return

    test.assertEqual(ref.shape, actual.shape, msg)

    ref64 = ref.astype(np.float64)
    act64 = actual.astype(np.float64)

    # NaN/Inf must match exactly, and NaN never equals itself (array_equal(equal_nan=) needs NumPy 1.19).
    nonfinite = ~np.isfinite(ref64) | ~np.isfinite(act64)
    if np.any(nonfinite):
        ref_nf, act_nf = ref64[nonfinite], act64[nonfinite]
        both_nan = np.isnan(ref_nf) & np.isnan(act_nf)
        test.assertTrue(
            bool(np.all((ref_nf == act_nf) | both_nan)),
            '%s: non-finite value mismatch (NaN/Inf) at %d position(s)' % (msg, int(nonfinite.sum())))

    # Subtract only where both sides are finite: inf - inf would warn and produce NaN.
    diff = np.zeros(ref64.shape, dtype=np.float64)
    finite = ~nonfinite
    diff[finite] = np.abs(ref64[finite] - act64[finite])
    normL1 = float(diff.sum()) / diff.size
    normInf = float(diff.max())
    test.assertLessEqual(normL1, l1, '%s: normL1=%r (l1=%r)' % (msg, normL1, l1))
    test.assertLessEqual(normInf, lInf, '%s: normInf=%r (lInf=%r)' % (msg, normInf, lInf))


class onnx_conformance_test(NewOpenCVTests):

    _cppDenylisted = None  # lazily built once per process, class-level cache

    def setUp(self):
        super(onnx_conformance_test, self).setUp()
        if onnx_conformance_test._cppDenylisted is None:
            onnx_conformance_test._cppDenylisted = cpp_denylisted(self.repoPath)

    def find_conformance_model(self, name, required=True):
        relpath = '/'.join(['dnn', 'onnx', 'conformance', 'node', name, 'model.onnx'])
        return self.find_file(relpath, [self.extraTestDataPath], required=required)

    def runConformanceCase(self, name):
        if name in self._cppDenylisted:
            self.skipTest('excluded by OpenCV\'s C++ ONNX conformance denylist')

        model_path = self.find_conformance_model(name, required=False)
        node_dir = os.path.dirname(model_path)
        dataset_dir = os.path.join(node_dir, 'test_data_set_0')

        inputFiles = sorted(glob.glob(os.path.join(dataset_dir, 'input_*.pb')), key=_pbIndex)
        outputFiles = sorted(glob.glob(os.path.join(dataset_dir, 'output_*.pb')), key=_pbIndex)
        if not outputFiles:
            self.skipTest('No reference data in %s (opencv_extra out of date?)' % dataset_dir)

        inputs = [cv.dnn.readTensorFromONNX(f) for f in inputFiles]
        refs = [cv.dnn.readTensorFromONNX(f) for f in outputFiles]

        net = cv.dnn.readNetFromONNX(model_path)
        self.assertFalse(net.empty(), 'failed to parse %s' % model_path)

        inputNames = [str(i) for i in range(len(inputs))]
        net.setInputsNames(inputNames)
        for inputName, inp in zip(inputNames, inputs):
            net.setInput(inp, inputName)

        outputs = net.forward(net.getUnconnectedOutLayersNames())
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        self.assertGreaterEqual(len(outputs), len(refs),
                                 '%s: expected >= %d outputs, got %d' % (name, len(refs), len(outputs)))

        l1, lInf = TOLERANCE_OVERRIDES.get(name, (L1_DEFAULT, LINF_DEFAULT))
        for i in range(len(refs)):
            normAssert(self, refs[i], outputs[i], msg=name, l1=l1, lInf=lInf)


def _makeConformanceTest(caseName):
    def test(self):
        self.runConformanceCase(caseName)
    test.__name__ = str(caseName)
    return test


for _case_name in CONFORMANCE_TESTS:
    setattr(onnx_conformance_test, _case_name, _makeConformanceTest(_case_name))
del _case_name


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
