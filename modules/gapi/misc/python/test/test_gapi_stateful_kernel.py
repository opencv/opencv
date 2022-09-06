#!/usr/bin/env python

import numpy as np
import cv2 as cv
import os
import sys
import unittest

from tests_common import NewOpenCVTests


try:

    if sys.version_info[:2] < (3, 0):
        raise unittest.SkipTest('Python 2.x is not supported')


    class CounterState:
        def __init__(self):
            self.counter = 0


    @cv.gapi.op('stateful_counter',
                in_types=[cv.GOpaque.Int],
                out_types=[cv.GOpaque.Int])
    class GStatefulCounter:
        """Accumulate state counter on every call"""

        @staticmethod
        def outMeta(desc):
            return cv.empty_gopaque_desc()


    @cv.gapi.kernel(GStatefulCounter)
    class GStatefulCounterImpl:
        """Implementation for GStatefulCounter operation."""

        @staticmethod
        def setup(desc):
            return CounterState()

        @staticmethod
        def run(value, state):
            state.counter += value
            return state.counter


    class gapi_sample_pipelines(NewOpenCVTests):
        def test_stateful_kernel_single_instance(self):
            g_in  = cv.GOpaque.Int()
            g_out = GStatefulCounter.on(g_in)
            comp  = cv.GComputation(cv.GIn(g_in), cv.GOut(g_out))
            pkg   = cv.gapi.kernels(GStatefulCounterImpl)

            nums = [i for i in range(10)]
            acc = 0
            for v in nums:
                acc = comp.apply(cv.gin(v), args=cv.gapi.compile_args(pkg))

            self.assertEqual(sum(nums), acc)


        def test_stateful_kernel_multiple_instances(self):
            # NB: Every counter has his own independent state.
            g_in   = cv.GOpaque.Int()
            g_out0 = GStatefulCounter.on(g_in)
            g_out1 = GStatefulCounter.on(g_in)
            comp   = cv.GComputation(cv.GIn(g_in), cv.GOut(g_out0, g_out1))
            pkg    = cv.gapi.kernels(GStatefulCounterImpl)

            nums = [i for i in range(10)]
            acc0 = acc1 = 0
            for v in nums:
                acc0, acc1 = comp.apply(cv.gin(v), args=cv.gapi.compile_args(pkg))

            ref = sum(nums)
            self.assertEqual(ref, acc0)
            self.assertEqual(ref, acc1)


        def test_stateful_throw_setup(self):
            @cv.gapi.kernel(GStatefulCounter)
            class GThrowStatefulCounterImpl:
                """Implementation for GStatefulCounter operation
                   that throw exception in setup method"""

                @staticmethod
                def setup(desc):
                    raise Exception('Throw from setup method')

                @staticmethod
                def run(value, state):
                    raise Exception('Unreachable')

            g_in  = cv.GOpaque.Int()
            g_out = GStatefulCounter.on(g_in)
            comp  = cv.GComputation(cv.GIn(g_in), cv.GOut(g_out))
            pkg   = cv.gapi.kernels(GThrowStatefulCounterImpl)

            with self.assertRaises(Exception): comp.apply(cv.gin(42),
                                                          args=cv.gapi.compile_args(pkg))


        def test_stateful_reset(self):
            g_in  = cv.GOpaque.Int()
            g_out = GStatefulCounter.on(g_in)
            comp  = cv.GComputation(cv.GIn(g_in), cv.GOut(g_out))
            pkg   = cv.gapi.kernels(GStatefulCounterImpl)

            cc = comp.compileStreaming(args=cv.gapi.compile_args(pkg))

            cc.setSource(cv.gin(1))
            cc.start()
            for i in range(1, 10):
                _, actual = cc.pull()
                self.assertEqual(i, actual)
            cc.stop()

            cc.setSource(cv.gin(2))
            cc.start()
            for i in range(2, 10, 2):
                _, actual = cc.pull()
                self.assertEqual(i, actual)
            cc.stop()


except unittest.SkipTest as e:

    message = str(e)

    class TestSkip(unittest.TestCase):
        def setUp(self):
            self.skipTest('Skip tests: ' + message)

        def test_skip():
            pass

    pass


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
