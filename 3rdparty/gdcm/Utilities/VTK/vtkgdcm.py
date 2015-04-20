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

""" This module loads all the classes from the GDCM-VTK library into
its namespace.  This is a required module."""

import os

# FIXME: GDCM_WHEREAMI need also to be set here, since the lib is dlopen'ed before gdcm.py is
# actually read...
os.environ["GDCM_WHEREAMI"]=os.path.dirname(__file__)
if os.name == 'posix':
  # extremely important !
  # http://gcc.gnu.org/faq.html#dso
  # http://mail.python.org/pipermail/python-dev/2002-May/023923.html
  # http://wiki.python.org/moin/boost.python/CrossExtensionModuleDependencies
  # This is now merged in VTK 5.2:
  # http://vtk.org/cgi-bin/viewcvs.cgi/Wrapping/Python/vtk/__init__.py?r1=1.13&r2=1.14
  import sys
  orig_dlopen_flags = sys.getdlopenflags()
  try:
    import dl
  except ImportError:
    # are we on AMD64 ?
    try:
      import DLFCN as dl
    except ImportError:
      print("Could not import dl")
      dl = None
  if dl:
    #print "dl was imported"
    #sys.setdlopenflags(dl.RTLD_LAZY|dl.RTLD_GLOBAL)
    sys.setdlopenflags(dl.RTLD_NOW|dl.RTLD_GLOBAL)
  from libvtkgdcmPython import *
  # revert:
  sys.setdlopenflags(orig_dlopen_flags)
  del sys, dl
  del orig_dlopen_flags
else:
  from vtkgdcmPython import *

# to provide a compatibilty layer with VTK 4.2 and VTK 4.4 where vtkStringArray was not present
# and VTK 5.x where there is one...
try:
  # if vtkStringArray can be found in vtk let's use it !
  from vtk import vtkStringArray
except:
  print("Using compatibility layer (VTK 4) for vtkStringArray")

# bye bye
del os
