############################################################################
#
#  Program: GDCM (Grassroots DICOM). A DICOM library
#
#  Copyright (c) 2006-2011 Mathieu Malaterre
#  All rights reserved.
#  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.
#
#     This software is distributed WITHOUT ANY WARRANTY; without even
#     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#     PURPOSE.  See the above copyright notice for more information.
#
############################################################################

""" This module loads all the classes from the GDCM library into
its namespace.  This is a required module."""

# warning: when loading 'import gdcm', if a gdcm.so is to be found in the same
# directory as a gdcm.py file, then there will be conflict. This will happen
# in the case of a user wrapping to PHP and Python at the same time.

import os
import sys

# This file is a thin wrapper to the swig generated python module. It allows us
# to do a couple of things:
# 1. do the RTLD_GLOBAL thingy on GNU system (with GNU compiler) before loading
#    the compiled python module
# 2. Load some secret path using directly the locate of this gdcm.py file.
#    a. If the gdcm.py is installed in a normal installation then we can deduce
#    where the Part3.xml can be found. This is the 'non frozen' case
#    b. Is the python executable is frozen then assume that everything is at
#    the same level and look for Part3.xml
#    at the same level as the frozen application is (see py2exe for more info)
# 3. Finally this is also a good time to look up the env var and if
# GDCM_RESOURCES_PATH is set, then fill the 'resource manager' via the
# Global.Prepend interface.

def main_is_frozen():
  return hasattr(sys, "frozen")

if os.name == 'posix':
  # extremely important !
  # http://gcc.gnu.org/faq.html#dso
  # http://mail.python.org/pipermail/python-dev/2002-May/023923.html
  # http://wiki.python.org/moin/boost.python/CrossExtensionModuleDependencies
  # http://mail.python.org/pipermail/cplusplus-sig/2005-August/009135.html
  orig_dlopen_flags = sys.getdlopenflags()
  try:
    import dl
  except ImportError:
    # are we on AMD64 ?
    try:
      import DLFCN as dl
    except ImportError:
      #print "Could not import dl"
      dl = None
  if dl:
    #print "dl was imported"
    #sys.setdlopenflags(dl.RTLD_LAZY|dl.RTLD_GLOBAL)
    sys.setdlopenflags(dl.RTLD_NOW|dl.RTLD_GLOBAL)
  from gdcmswig import *
  # revert:
  sys.setdlopenflags(orig_dlopen_flags)
  del dl
  del orig_dlopen_flags
else:
  from gdcmswig import *

# To finish up with module loading let's do some more stuff, like path to resource init:
if main_is_frozen():
  Global.GetInstance().Prepend( os.path.dirname(sys.executable) )
else:
  Global.GetInstance().Prepend( os.path.dirname(__file__) + "/../../../"  + GDCM_INSTALL_DATA_DIR + "/XML/" )

# Do it afterward so that it comes in first in the list
try:
  Global.GetInstance().Prepend( os.environ["GDCM_RESOURCES_PATH"] )
except:
  pass

# bye bye
# once the process dies, the changed environment dies with it.
del os,sys
