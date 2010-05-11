#########################################################################################
#
#  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
#
#  By downloading, copying, installing or using the software you agree to this license.
#  If you do not agree to this license, do not download, install,
#  copy or use the software.
#
#
#                        Intel License Agreement
#                For Open Source Computer Vision Library
#
# Copyright (C) 2000, Intel Corporation, all rights reserved.
# Third party copyrights are property of their respective owners.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#   * Redistribution's of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#   * Redistribution's in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#   * The name of Intel Corporation may not be used to endorse or promote products
#     derived from this software without specific prior written permission.
#
# This software is provided by the copyright holders and contributors "as is" and
# any express or implied warranties, including, but not limited to, the implied
# warranties of merchantability and fitness for a particular purpose are disclaimed.
# In no event shall the Intel Corporation or contributors be liable for any direct,
# indirect, incidental, special, exemplary, or consequential damages
# (including, but not limited to, procurement of substitute goods or services;
# loss of use, data, or profits; or business interruption) however caused
# and on any theory of liability, whether in contract, strict liability,
# or tort (including negligence or otherwise) arising in any way out of
# the use of this software, even if advised of the possibility of such damage.
#
#########################################################################################

# 2004-03-16, Mark Asbach       <asbach@ient.rwth-aachen.de>
#             Institute of Communications Engineering, RWTH Aachen University

"""The Open Computer Vision Library

OpenCV is the Open Computer Vision library, an open source effort originally started
by intel to provide computer vision algorithms for standard PC hardware.

This wrapper was semi-automatically created from the C/C++ headers and therefore
contains no Python documentation. Because all identifiers are identical to their
C/C++ counterparts, you can consult the standard manuals that come with OpenCV.

In detail, this python package contains four sub-modules:

  cv             core components (cxcore and cv)
  ml             machine learning
  highgui        simple user interface, video and image I/O
  adaptors       pure python module offering conversion to numpy/scipy matrices 
                 and PIL (python imaging library) images
  matlab_syntax  pure python module offering syntax that is similar to Matlab
                 for those who switched

All methods and data types from cv, ml and adaptors are automatically imported into 
the opencv namespace. Contents from highgui and matlab_syntax must be explicitly 
imported - to avoid conflicts with other UI toolkits and the python matlab interface.
"""

# the module consists of these four sub-modules
__all__ = ['cv', 'ml', 'highgui', 'adaptors', 'matlab_syntax']

# always import functionality from cxcore, cv and ml to this namespace
# try to import PIL and numpy adaptors
from cv import *
from ml import *
from adaptors import *
