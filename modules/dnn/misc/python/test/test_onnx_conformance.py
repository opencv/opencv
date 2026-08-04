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
from test_onnx_conformance_list import CONFORMANCE_TESTS

# Mirrors modules/dnn/test/test_onnx_conformance.cpp through cv2.dnn bindings, since that C++ suite never calls into Python and so can't catch binding-layer regressions.
# The case list lives in test_onnx_conformance_list.py; OpenCV/CPU only, since bindings are backend-agnostic.
# Denylisted cases are read from OpenCV's own C++ .inl.hpp files at test time (reused, not duplicated).

TOLERANCE_OVERRIDES = {
    "test_attention_4d_fp16": (0.0002, 0.001),
    "test_attention_4d_fp16_expanded": (0.0002, 0.001),
    "test_attention_4d_gqa_with_past_and_present_fp16": (0.0002, 0.001),
    "test_attention_4d_gqa_with_past_and_present_fp16_expanded": (0.0002, 0.001),
    "test_causal_conv_with_state_fp16": (0.0002, 0.002),
    "test_causal_conv_with_state_silu_fp16": (0.0002, 0.002),
    "test_gelu_tanh_1": (0.00011, 0.00016),
    "test_gelu_tanh_2": (9e-05, 0.0005),
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

# Only parser + opencv_all apply to a CPU-only test; classic/CUDA/Vulkan/OCL denylists don't.
_CPP_DENYLIST_FILES = (
    'test_onnx_conformance_layer_parser_denylist.inl.hpp',
    'test_onnx_conformance_layer_filter_opencv_all_denylist.inl.hpp',
)
_DENYLIST_ENTRY_RE = re.compile(r'^\s*"(test_[A-Za-z0-9_]+)"\s*,', re.M)


def _cpp_denylisted(repoPath):
    """Names excluded by OpenCV's C++ ONNX denylists; empty if the source tree isn't available."""
    names = set()
    if not repoPath:
        return names
    test_dir = os.path.join(repoPath, 'modules', 'dnn', 'test')
    for fname in _CPP_DENYLIST_FILES:
        fpath = os.path.join(test_dir, fname)
        if os.path.isfile(fpath):
            with open(fpath) as fh:
                names.update(_DENYLIST_ENTRY_RE.findall(fh.read()))
    return names


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

    # inf - inf = nan in IEEE arithmetic, so non-finite values need an exact-equality check, not a subtraction.
    nonfinite = ~np.isfinite(ref64) | ~np.isfinite(act64)
    if np.any(nonfinite):
        test.assertTrue(
            np.array_equal(ref64[nonfinite], act64[nonfinite], equal_nan=True),
            '%s: non-finite value mismatch (NaN/Inf) at %d position(s)' % (msg, int(nonfinite.sum())))

    diff = np.where(nonfinite, 0.0, np.abs(ref64 - act64))
    normL1 = float(diff.sum()) / diff.size
    normInf = float(diff.max())
    test.assertLessEqual(normL1, l1, '%s: normL1=%r (l1=%r)' % (msg, normL1, l1))
    test.assertLessEqual(normInf, lInf, '%s: normInf=%r (lInf=%r)' % (msg, normInf, lInf))


class onnx_conformance_test(NewOpenCVTests):

    _cppDenylisted = None  # lazily built once per process, class-level cache

    def setUp(self):
        super(onnx_conformance_test, self).setUp()
        if onnx_conformance_test._cppDenylisted is None:
            onnx_conformance_test._cppDenylisted = _cpp_denylisted(self.repoPath)

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
