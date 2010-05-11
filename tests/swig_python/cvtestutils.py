# cvtestutils.py
#
# This module is meant to aid writing and running Python-based tests 
# within the OpenCV tree.  
#
# 2009-01-23, Roman Stanchak <rstanchak@gmail.com>
#
#
# Upon importing, this module adds the following python module 
# directories from the dev tree to sys.path (i.e. PYTHON_PATH):
#    opencv/interfaces/swig/python and opencv/lib
#
# Using it in a test case is simple, just be sure to import 
# cvtestutils before cv, highgui, etc
#
# Usage:
# import cvtestutils
# import cv

import os
import imp
import sys

def top_srcdir():
    """
    Return a string containing the top-level source directory in the OpenCV tree
    """
    dir = os.path.dirname(os.path.realpath(__file__))

    # top level dir should be two levels up from this file
    return os.path.realpath( os.path.join( dir, '..', '..' ) )

def top_builddir():
    """
    Return a string containing the top-level build directory in the OpenCV tree.
    Either returns realpath(argv[1]) or top_srcdir();
    """
    if len(sys.argv)>1: return os.path.realpath( sys.argv[1] );
    else: return top_srcdir();

def initpath():
    """
    Prepend the python module directories from the dev tree to sys.path 
    (i.e. PYTHON_PATH)
    """

    # add path for OpenCV source directory (for adapters.py)
    moduledir = os.path.join(top_srcdir(), 'interfaces','swig','python')
    moduledir = os.path.realpath(moduledir)
    sys.path.insert(0, moduledir) 

    # add path for OpenCV build directory
    moduledir = os.path.join(top_builddir(), 'interfaces','swig','python')
    moduledir = os.path.realpath(moduledir)
    sys.path.insert(0, moduledir) 

    libdir = os.path.join(top_builddir(), 'lib' )
    libdir = os.path.realpath(libdir)
    sys.path.insert(0, libdir)

def which():
    """
    Print the directory containing cv.py
    """
    import cv 
    print "Using OpenCV Python in: " + os.path.dirname(cv.__file__)

def datadir():
    """
    Return a string containing the full path to the python testdata directory
    """
    return os.path.sep.join([top_srcdir(), '..', 'opencv_extra', 'testdata', 'python'])

### Module Initialization
try:
    if MODULE_LOADED:
        pass
except NameError:
    initpath()
    which()
    MODULE_LOADED=1
