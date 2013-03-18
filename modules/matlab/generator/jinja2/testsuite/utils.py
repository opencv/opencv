# -*- coding: utf-8 -*-
"""
    jinja2.testsuite.utils
    ~~~~~~~~~~~~~~~~~~~~~~

    Tests utilities jinja uses.

    :copyright: (c) 2010 by the Jinja Team.
    :license: BSD, see LICENSE for more details.
"""
import gc
import unittest

import pickle

from jinja2.testsuite import JinjaTestCase

from jinja2.utils import LRUCache, escape, object_type_repr


class LRUCacheTestCase(JinjaTestCase):

    def test_simple(self):
        d = LRUCache(3)
        d["a"] = 1
        d["b"] = 2
        d["c"] = 3
        d["a"]
        d["d"] = 4
        assert len(d) == 3
        assert 'a' in d and 'c' in d and 'd' in d and 'b' not in d

    def test_pickleable(self):
        cache = LRUCache(2)
        cache["foo"] = 42
        cache["bar"] = 23
        cache["foo"]

        for protocol in range(3):
            copy = pickle.loads(pickle.dumps(cache, protocol))
            assert copy.capacity == cache.capacity
            assert copy._mapping == cache._mapping
            assert copy._queue == cache._queue


class HelpersTestCase(JinjaTestCase):

    def test_object_type_repr(self):
        class X(object):
            pass
        self.assert_equal(object_type_repr(42), 'int object')
        self.assert_equal(object_type_repr([]), 'list object')
        self.assert_equal(object_type_repr(X()),
                         'jinja2.testsuite.utils.X object')
        self.assert_equal(object_type_repr(None), 'None')
        self.assert_equal(object_type_repr(Ellipsis), 'Ellipsis')


class MarkupLeakTestCase(JinjaTestCase):

    def test_markup_leaks(self):
        counts = set()
        for count in xrange(20):
            for item in xrange(1000):
                escape("foo")
                escape("<foo>")
                escape(u"foo")
                escape(u"<foo>")
            counts.add(len(gc.get_objects()))
        assert len(counts) == 1, 'ouch, c extension seems to leak objects'


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(LRUCacheTestCase))
    suite.addTest(unittest.makeSuite(HelpersTestCase))

    # this test only tests the c extension
    if not hasattr(escape, 'func_code'):
        suite.addTest(unittest.makeSuite(MarkupLeakTestCase))

    return suite
