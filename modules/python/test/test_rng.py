#!/usr/bin/env python

'''
Tests for the per-instance cv.RNG bindings.

They check that a cv.RNG object owns and advances its own C++ cv::RNG state and
that this state is fully isolated from the default (global) OpenCV generator.
'''

from __future__ import print_function

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests


# Deliberately larger than 32 bits so that a truncated uint64 seed is detectable.
BIG_SEED = (1 << 40) + 137


class rng_test(NewOpenCVTests):

    def global_uniform_bytes(self, shape=(8, 8)):
        """Draws from the default (global) generator via the module level API."""
        return cv.randu(np.zeros(shape, np.uint8), 0, 256).copy()

    def test_rng_is_exposed_and_constructible(self):
        self.assertTrue(hasattr(cv, 'RNG'))

        default_rng = cv.RNG()
        self.assertIsInstance(default_rng, cv.RNG)
        # cv::RNG() uses 2**32-1 as the pre-defined initial state.
        self.assertEqual(default_rng.state, 0xffffffff)

        seeded_rng = cv.RNG(137)
        self.assertIsInstance(seeded_rng, cv.RNG)
        self.assertEqual(seeded_rng.state, 137)

        # The distribution identifiers needed by fill() are available.
        self.assertEqual(cv.RNG_UNIFORM, 0)
        self.assertEqual(cv.RNG_NORMAL, 1)

    def test_state_is_readable_and_writable(self):
        rng = cv.RNG()
        rng.state = 137
        self.assertEqual(rng.state, 137)

        seeded = cv.RNG(137)
        self.assertEqual([rng.next() for _ in range(4)],
                         [seeded.next() for _ in range(4)])

    def test_seed_is_not_truncated_to_32_bits(self):
        self.assertEqual(cv.RNG(BIG_SEED).state, BIG_SEED)

        full, truncated = cv.RNG(BIG_SEED), cv.RNG(BIG_SEED & 0xffffffff)
        self.assertNotEqual([full.next() for _ in range(4)],
                            [truncated.next() for _ in range(4)])

    def test_same_seed_gives_identical_sequences(self):
        rng_a = cv.RNG(137)
        rng_b = cv.RNG(137)

        self.assertEqual([rng_a.next() for _ in range(16)],
                         [rng_b.next() for _ in range(16)])
        self.assertEqual([rng_a.uniform(0, 1000) for _ in range(16)],
                         [rng_b.uniform(0, 1000) for _ in range(16)])
        self.assertEqual([rng_a.uniform(-1.5, 2.5) for _ in range(16)],
                         [rng_b.uniform(-1.5, 2.5) for _ in range(16)])
        self.assertEqual([rng_a.gaussian(2.0) for _ in range(16)],
                         [rng_b.gaussian(2.0) for _ in range(16)])
        self.assertEqual(rng_a.state, rng_b.state)

    def test_instances_advance_independently(self):
        rng_a = cv.RNG(BIG_SEED)
        rng_b = cv.RNG(BIG_SEED)

        # Advancing rng_a must leave rng_b exactly where it was.
        head_a = [rng_a.next() for _ in range(5)]
        self.assertNotEqual(rng_a.state, rng_b.state)
        self.assertEqual(rng_b.state, BIG_SEED)

        head_b = [rng_b.next() for _ in range(5)]
        self.assertEqual(head_a, head_b)
        self.assertEqual(rng_a.state, rng_b.state)

        # Draws 5..9 of the sequence differ from draws 0..4, so a generator that
        # is ahead really does yield different values than a freshly seeded one.
        tail = [rng_a.next() for _ in range(5)]
        fresh = cv.RNG(BIG_SEED)
        self.assertNotEqual(tail, [fresh.next() for _ in range(5)])

    def test_fill_uniform(self):
        low, high = 10, 20
        shape = (4, 5)

        dst_a = np.zeros(shape, np.uint8)
        dst_b = np.zeros(shape, np.uint8)
        out_a = cv.RNG(137).fill(dst_a, cv.RNG_UNIFORM, low, high)
        out_b = cv.RNG(137).fill(dst_b, cv.RNG_UNIFORM, low, high)

        self.assertEqual(out_a.dtype, np.uint8)
        self.assertEqual(out_a.shape, shape)
        self.assertTrue(np.array_equal(out_a, out_b))
        self.assertTrue(np.all(out_a >= low))
        self.assertTrue(np.all(out_a < high))

        # Filling advances the instance, so a second fill differs from the first.
        rng = cv.RNG(137)
        first = rng.fill(np.zeros(shape, np.uint8), cv.RNG_UNIFORM, low, high).copy()
        state_after_first = rng.state
        second = rng.fill(np.zeros(shape, np.uint8), cv.RNG_UNIFORM, low, high).copy()
        self.assertTrue(np.array_equal(first, out_a))
        self.assertNotEqual(rng.state, state_after_first)
        self.assertFalse(np.array_equal(first, second))

    def test_fill_normal(self):
        shape = (3, 7)

        out_a = cv.RNG(137).fill(np.zeros(shape, np.float32), cv.RNG_NORMAL, 0.0, 1.0)
        out_b = cv.RNG(137).fill(np.zeros(shape, np.float32), cv.RNG_NORMAL, 0.0, 1.0)

        self.assertEqual(out_a.dtype, np.float32)
        self.assertEqual(out_a.shape, shape)
        self.assertTrue(np.array_equal(out_a, out_b))
        self.assertTrue(np.all(np.isfinite(out_a)))

        # A different seed drives a different sequence.
        out_c = cv.RNG(138).fill(np.zeros(shape, np.float32), cv.RNG_NORMAL, 0.0, 1.0)
        self.assertFalse(np.array_equal(out_a, out_c))

    def test_scalar_operations_mutate_only_the_instance(self):
        rng = cv.RNG(137)
        other = cv.RNG(137)

        state = rng.state
        value = rng.next()
        self.assertIsInstance(value, int)
        self.assertTrue(0 <= value <= 0xffffffff)
        self.assertNotEqual(rng.state, state)
        self.assertEqual(other.state, 137)

        state = rng.state
        value = rng.uniform(3, 9)
        self.assertIsInstance(value, int)
        self.assertTrue(3 <= value < 9)
        self.assertNotEqual(rng.state, state)

        state = rng.state
        value = rng.uniform(-2.0, 2.0)
        self.assertIsInstance(value, float)
        self.assertTrue(-2.0 <= value < 2.0)
        self.assertNotEqual(rng.state, state)

        state = rng.state
        value = rng.gaussian(1.5)
        self.assertIsInstance(value, float)
        self.assertNotEqual(rng.state, state)

        # None of the above touched the second instance.
        self.assertEqual(other.state, 137)

    def test_instance_does_not_disturb_global_rng(self):
        cv.setRNGSeed(12345)
        expected = self.global_uniform_bytes()

        cv.setRNGSeed(12345)
        rng = cv.RNG(999)
        rng.next()
        rng.uniform(0, 100)
        rng.uniform(0.0, 1.0)
        rng.gaussian(1.0)
        rng.fill(np.zeros((6, 6), np.uint8), cv.RNG_UNIFORM, 0, 256)
        rng.fill(np.zeros((6, 6), np.float32), cv.RNG_NORMAL, 0.0, 1.0)
        cv.randShuffle(np.arange(32, dtype=np.int32).reshape(1, -1), 1.0, rng)
        actual = self.global_uniform_bytes()

        self.assertTrue(np.array_equal(expected, actual))

    def test_global_rng_does_not_disturb_instances(self):
        rng_a = cv.RNG(137)
        rng_b = cv.RNG(137)

        seq_a = [rng_a.next() for _ in range(8)]

        cv.setRNGSeed(4242)
        cv.randu(np.zeros((16, 16), np.uint8), 0, 256)
        cv.randn(np.zeros((16, 16), np.float32), 0.0, 1.0)
        cv.setRNGSeed(7)
        cv.randShuffle(np.arange(64, dtype=np.int32).reshape(1, -1))

        seq_b = [rng_b.next() for _ in range(8)]

        self.assertEqual(seq_a, seq_b)
        self.assertEqual(rng_a.state, rng_b.state)

    def test_rand_shuffle_with_instance_rng(self):
        values = np.arange(64, dtype=np.int32).reshape(1, -1)

        shuffled_a = cv.randShuffle(values.copy(), 1.0, cv.RNG(137))
        shuffled_b = cv.randShuffle(values.copy(), 1.0, cv.RNG(137))

        self.assertTrue(np.array_equal(shuffled_a, shuffled_b))
        self.assertTrue(np.array_equal(np.sort(shuffled_a, axis=1), values))

        # The rng argument is really used: a different seed shuffles differently.
        shuffled_c = cv.randShuffle(values.copy(), 1.0, cv.RNG(555))
        self.assertFalse(np.array_equal(shuffled_a, shuffled_c))

        # ... and the global seed is irrelevant when an instance is supplied.
        cv.setRNGSeed(1)
        shuffled_d = cv.randShuffle(values.copy(), 1.0, cv.RNG(137))
        cv.setRNGSeed(2)
        shuffled_e = cv.randShuffle(values.copy(), 1.0, cv.RNG(137))
        self.assertTrue(np.array_equal(shuffled_a, shuffled_d))
        self.assertTrue(np.array_equal(shuffled_a, shuffled_e))

    def test_rand_shuffle_advances_the_supplied_instance(self):
        values = np.arange(64, dtype=np.int32).reshape(1, -1)

        rng = cv.RNG(137)
        state = rng.state
        first = cv.randShuffle(values.copy(), 1.0, rng).copy()
        # A copy of the RNG would have left the caller's state untouched.
        self.assertNotEqual(rng.state, state)

        second = cv.randShuffle(values.copy(), 1.0, rng).copy()
        self.assertFalse(np.array_equal(first, second))

        # Two independent instances stay in lockstep with each other.
        rng_a, rng_b = cv.RNG(137), cv.RNG(137)
        cv.randShuffle(values.copy(), 1.0, rng_a)
        self.assertNotEqual(rng_a.state, rng_b.state)
        cv.randShuffle(values.copy(), 1.0, rng_b)
        self.assertEqual(rng_a.state, rng_b.state)

    def test_rand_shuffle_without_instance_uses_global_rng(self):
        values = np.arange(64, dtype=np.int32).reshape(1, -1)

        cv.setRNGSeed(7)
        expected = cv.randShuffle(values.copy()).copy()
        cv.setRNGSeed(7)
        self.assertTrue(np.array_equal(cv.randShuffle(values.copy(), 1.0), expected))
        cv.setRNGSeed(7)
        self.assertTrue(np.array_equal(cv.randShuffle(values.copy(), 1.0, None), expected))

        # An interleaved instance based shuffle leaves the global sequence intact.
        cv.setRNGSeed(7)
        cv.randShuffle(values.copy(), 1.0, cv.RNG(137))
        self.assertTrue(np.array_equal(cv.randShuffle(values.copy()), expected))

    def test_rand_shuffle_rejects_non_rng_argument(self):
        values = np.arange(8, dtype=np.int32).reshape(1, -1)
        with self.assertRaises(cv.error):
            cv.randShuffle(values, 1.0, 137)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
