# -*- coding: utf-8 -*-
"""
    jinja2.testsuite.doctests
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    The doctests.  Collects all tests we want to test from
    the Jinja modules.

    :copyright: (c) 2010 by the Jinja Team.
    :license: BSD, see LICENSE for more details.
"""
import unittest
import doctest


def suite():
    from jinja2 import utils, sandbox, runtime, meta, loaders, \
        ext, environment, bccache, nodes
    suite = unittest.TestSuite()
    suite.addTest(doctest.DocTestSuite(utils))
    suite.addTest(doctest.DocTestSuite(sandbox))
    suite.addTest(doctest.DocTestSuite(runtime))
    suite.addTest(doctest.DocTestSuite(meta))
    suite.addTest(doctest.DocTestSuite(loaders))
    suite.addTest(doctest.DocTestSuite(ext))
    suite.addTest(doctest.DocTestSuite(environment))
    suite.addTest(doctest.DocTestSuite(bccache))
    suite.addTest(doctest.DocTestSuite(nodes))
    return suite
