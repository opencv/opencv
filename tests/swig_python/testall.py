#!/usr/bin/env python

# This script uses the unittest module to find all the tests in the 
# same directory and run them.
#
# 2009-01-23, Roman Stanchak (rstanchak@gmail.com)
#
# 
# For a test to be detected and run by this script, it must
# 1. Use unittest
# 2. define a suite() method that returns a unittest.TestSuite containing
#    the tests to be run

import cvtestutils
import unittest
import types
import os
import imp

def suites( dirname ):
    suite_list=[]

    for fn in os.listdir( dirname ):
        # tests must be named test_*.py or *_tests.py
        if not ( fn.lower().endswith('.py') and 
                 (fn.lower().startswith('test_') or fn.lower().endswith('_tests.py')) ):
            continue

        module_name = fn[0:-3]
        fullpath = os.path.realpath( dirname + os.path.sep + fn )
        test_module = None
        try:
            test_module = imp.load_source( module_name, fullpath )
        except:
            print "Error importing python code in '%s'" % fn
        if test_module:
            try:
                suite_list.append( test_module.suite() )
                print "Added tests from %s" % fn
            except:
                print "%s does not contain a suite() method, skipping" % fn
    return unittest.TestSuite(suite_list)

    
def col2( c1, c2, w=72 ):
    return "%s%s" % (c1, c2.rjust(w-len(c1)))

if __name__ == "__main__":
    print '----------------------------------------------------------------------'
    print 'Searching for tests...'
    print '----------------------------------------------------------------------'
    suite = suites( os.path.dirname( os.path.realpath(__file__) ))
    print '----------------------------------------------------------------------'
    print 'Running tests...'
    print '----------------------------------------------------------------------'
    unittest.TextTestRunner(verbosity=2).run(suite)
