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
        """Accumulates state counter on every call"""

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


    class SumState:
        def __init__(self):
            self.sum = 0


    @cv.gapi.op('stateful_sum',
                in_types=[cv.GOpaque.Int, cv.GOpaque.Int],
                out_types=[cv.GOpaque.Int])
    class GStatefulSum:
        """Accumulates sum on every call"""

        @staticmethod
        def outMeta(lhs_desc, rhs_desc):
            return cv.empty_gopaque_desc()


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


        def test_stateful_multiple_inputs(self):
            @cv.gapi.kernel(GStatefulSum)
            class GStatefulSumImpl:
                """Implementation for GStatefulCounter operation."""

                @staticmethod
                def setup(lhs_desc, rhs_desc):
                    return SumState()

                @staticmethod
                def run(lhs, rhs, state):
                    state.sum+= lhs + rhs
                    return state.sum


            g_in1 = cv.GOpaque.Int()
            g_in2 = cv.GOpaque.Int()
            g_out = GStatefulSum.on(g_in1, g_in2)
            comp = cv.GComputation(cv.GIn(g_in1, g_in2), cv.GOut(g_out))
            pkg  = cv.gapi.kernels(GStatefulSumImpl)

            lhs_list = [1, 10, 15]
            rhs_list = [2, 14, 32]

            ref_out = 0
            for lhs, rhs in zip(lhs_list, rhs_list):
                ref_out += lhs + rhs
                gapi_out = comp.apply(cv.gin(lhs, rhs), cv.gapi.compile_args(pkg))
                self.assertEqual(ref_out, gapi_out)


        def test_stateful_multiple_inputs_throw(self):
            @cv.gapi.kernel(GStatefulSum)
            class GStatefulSumImplIncorrect:
                """Incorrect implementation for GStatefulCounter operation."""

                # NB: setup methods is intentionally
                # incorrect - accepts one meta arg instead of two
                @staticmethod
                def setup(desc):
                    return SumState()

                @staticmethod
                def run(lhs, rhs, state):
                    state.sum+= lhs + rhs
                    return state.sum


            g_in1 = cv.GOpaque.Int()
            g_in2 = cv.GOpaque.Int()
            g_out = GStatefulSum.on(g_in1, g_in2)
            comp = cv.GComputation(cv.GIn(g_in1, g_in2), cv.GOut(g_out))
            pkg  = cv.gapi.kernels(GStatefulSumImplIncorrect)

            with self.assertRaises(Exception): comp.apply(cv.gin(42, 42),
                                                          args=cv.gapi.compile_args(pkg))


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
