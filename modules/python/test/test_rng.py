#!/usr/bin/env python

'''
Tests for the cv.RNG bindings: per-instance random number generators.

See https://github.com/opencv/opencv/issues/29591
'''

import threading

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests


class rng_test(NewOpenCVTests):

    def test_constructors(self):
        # the default constructor uses the documented 2**32-1 state
        self.assertEqual(cv.RNG().state, 0xffffffff)
        self.assertEqual(cv.RNG(137).state, 137)
        # zero state is replaced by the default one to avoid the singular sequence
        self.assertEqual(cv.RNG(0).state, 0xffffffff)
        # 64-bit states are supported
        self.assertEqual(cv.RNG(0x123456789abcdef).state, 0x123456789abcdef)

    def test_distribution_type_constants(self):
        # cv::RNG::UNIFORM and cv::RNG::NORMAL
        self.assertEqual(cv.RNG_UNIFORM, 0)
        self.assertEqual(cv.RNG_NORMAL, 1)

    def test_same_seed_gives_same_sequence(self):
        rng_a = cv.RNG(137)
        rng_b = cv.RNG(137)

        for _ in range(10):
            self.assertEqual(rng_a.next(), rng_b.next())
            self.assertEqual(rng_a.uniform(0, 100), rng_b.uniform(0, 100))
            self.assertEqual(rng_a.uniform(0.0, 1.0), rng_b.uniform(0.0, 1.0))
            self.assertEqual(rng_a.gaussian(2.0), rng_b.gaussian(2.0))

        self.assertEqual(rng_a.state, rng_b.state)

    def test_uniform_overloads(self):
        rng = cv.RNG(137)

        for _ in range(100):
            i = rng.uniform(3, 7)
            self.assertIsInstance(i, int)
            self.assertTrue(3 <= i < 7)

            # the floating-point overload must be selected by the type of the
            # boundaries, otherwise the result would always be truncated to 3
            f = rng.uniform(3.0, 7.0)
            self.assertIsInstance(f, float)
            self.assertTrue(3.0 <= f < 7.0)

    def test_state_is_read_write(self):
        rng = cv.RNG(137)
        saved = rng.state
        first = [rng.next() for _ in range(5)]

        # rewinding the generator by restoring the state repeats the sequence
        rng.state = saved
        self.assertEqual(first, [rng.next() for _ in range(5)])

        # ... and so does a fresh generator initialized with the same state
        rewound = cv.RNG(saved)
        self.assertEqual(first, [rewound.next() for _ in range(5)])

    def test_fill_uniform(self):
        shape = (7, 11)
        dst_a = np.zeros(shape, np.uint8)
        dst_b = np.zeros(shape, np.uint8)

        dst_a = cv.RNG(137).fill(dst_a, cv.RNG_UNIFORM, 0, 256)
        dst_b = cv.RNG(137).fill(dst_b, cv.RNG_UNIFORM, 0, 256)

        self.assertEqual(dst_a.shape, shape)
        np.testing.assert_array_equal(dst_a, dst_b)
        # a 77-element uniformly distributed array is not expected to be constant
        self.assertGreater(dst_a.max(), dst_a.min())

    def test_fill_normal(self):
        dst = np.zeros((1000, 1000), np.float32)
        dst = cv.RNG(137).fill(dst, cv.RNG_NORMAL, 10.0, 2.0)

        self.assertAlmostEqual(dst.mean(), 10.0, delta=0.05)
        self.assertAlmostEqual(dst.std(), 2.0, delta=0.05)

    def test_fill_matches_the_seeded_global_generator(self):
        shape = (16, 16)

        cv.setRNGSeed(137)
        expected_u = cv.randu(np.zeros(shape, np.float32), 0.0, 1.0)
        expected_n = cv.randn(np.zeros(shape, np.float32), 0.0, 1.0)

        rng = cv.RNG(137)
        actual_u = rng.fill(np.zeros(shape, np.float32), cv.RNG_UNIFORM, 0.0, 1.0)
        actual_n = rng.fill(np.zeros(shape, np.float32), cv.RNG_NORMAL, 0.0, 1.0)

        np.testing.assert_array_equal(actual_u, expected_u)
        np.testing.assert_array_equal(actual_n, expected_n)

    def test_generator_passed_to_a_function_is_updated(self):
        rng = cv.RNG(137)
        dst = cv.randShuffle(np.arange(64, dtype=np.float32), 1.0, rng)

        # the generator is passed by reference, so the swaps performed by
        # randShuffle() must be reflected in the caller's generator
        self.assertNotEqual(rng.state, 137)
        self.assertEqual(sorted(dst.tolist()), list(range(64)))

        # ... and the very same sequence of swaps is reproduced by an equally
        # seeded generator, which proves that randShuffle() got the state, too
        expected = cv.randShuffle(np.arange(64, dtype=np.float32), 1.0, cv.RNG(137))
        np.testing.assert_array_equal(dst, expected)

        # two consecutive calls must not repeat themselves: the state moved on
        again = cv.randShuffle(np.arange(64, dtype=np.float32), 1.0, rng)
        self.assertFalse(np.array_equal(dst, again))

    def test_function_with_explicit_generator_does_not_touch_the_global_one(self):
        cv.setRNGSeed(137)
        expected = cv.randShuffle(np.arange(64, dtype=np.float32), 1.0)

        cv.setRNGSeed(137)
        cv.randShuffle(np.arange(64, dtype=np.float32), 1.0, cv.RNG(42))
        actual = cv.randShuffle(np.arange(64, dtype=np.float32), 1.0)

        np.testing.assert_array_equal(actual, expected)

        # while without the argument randShuffle() uses (and advances) the default one
        cv.setRNGSeed(137)
        before = cv.theRNG().state
        cv.randShuffle(np.arange(64, dtype=np.float32), 1.0)
        self.assertNotEqual(cv.theRNG().state, before)

    def test_function_can_take_the_default_generator(self):
        cv.setRNGSeed(137)
        expected = cv.randShuffle(np.arange(64, dtype=np.float32), 1.0)

        # passing theRNG() explicitly must be equivalent to omitting the argument
        cv.setRNGSeed(137)
        actual = cv.randShuffle(np.arange(64, dtype=np.float32), 1.0, cv.theRNG())

        np.testing.assert_array_equal(actual, expected)

    def test_instances_are_independent(self):
        rng_a = cv.RNG(137)
        rng_b = cv.RNG(137)

        expected = [rng_a.next() for _ in range(4)]

        # interleaving the calls to other generators does not change the sequence
        actual = []
        for i in range(4):
            cv.RNG(i + 1).gaussian(1.0)
            actual.append(rng_b.next())

        self.assertEqual(expected, actual)

    def test_instances_do_not_affect_the_global_generator(self):
        shape = (8, 8)

        cv.setRNGSeed(137)
        expected = cv.randu(np.zeros(shape, np.float32), 0.0, 1.0)

        cv.setRNGSeed(137)
        rng = cv.RNG(42)
        rng.next()
        rng.gaussian(1.0)
        rng.fill(np.zeros(shape, np.float32), cv.RNG_UNIFORM, 0.0, 1.0)
        actual = cv.randu(np.zeros(shape, np.float32), 0.0, 1.0)

        np.testing.assert_array_equal(actual, expected)

    def test_global_generator_does_not_affect_instances(self):
        rng = cv.RNG(137)
        expected = [rng.next() for _ in range(4)]

        rng = cv.RNG(137)
        actual = []
        for _ in range(4):
            cv.setRNGSeed(1234)
            cv.randu(np.zeros((4, 4), np.float32), 0.0, 1.0)
            actual.append(rng.next())

        self.assertEqual(expected, actual)

    def test_the_rng_refers_to_the_default_generator(self):
        cv.setRNGSeed(137)
        self.assertEqual(cv.theRNG().state, 137)

        # theRNG() is a handle to the default generator, not a copy of it:
        # advancing it must advance the generator used by cv.randu()
        cv.setRNGSeed(137)
        rng = cv.theRNG()
        rng.next()
        expected = cv.randu(np.zeros((8, 8), np.float32), 0.0, 1.0)

        cv.setRNGSeed(137)
        cv.theRNG().next()
        actual = cv.randu(np.zeros((8, 8), np.float32), 0.0, 1.0)

        np.testing.assert_array_equal(actual, expected)

        # ... and cv.randu() in turn advances the object returned by theRNG()
        cv.setRNGSeed(137)
        rng = cv.theRNG()
        cv.randu(np.zeros((8, 8), np.float32), 0.0, 1.0)
        self.assertNotEqual(rng.state, 137)
        self.assertEqual(rng.state, cv.theRNG().state)

    def test_the_rng_outlives_its_thread(self):
        result = {}

        def worker():
            cv.setRNGSeed(137)
            result['rng'] = cv.theRNG()

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        # the returned object shares the ownership of the generator, so it stays
        # valid even though the thread that created it has already finished
        rng = result['rng']
        self.assertEqual(rng.state, 137)

        reference = cv.RNG(137)
        self.assertEqual([rng.next() for _ in range(4)],
                         [reference.next() for _ in range(4)])

        # each thread has its own default generator, so the one of the finished
        # thread is not the default generator of this thread anymore
        cv.setRNGSeed(42)
        self.assertNotEqual(cv.theRNG().state, rng.state)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
