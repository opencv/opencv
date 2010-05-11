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


# 2004-03-16, Mark Asbach <asbach@ient.rwth-aachen.de>
#             Institute of Communications Engineering, RWTH Aachen University
# 2007-02-xx, direct interface to numpy by Vicent Mas <vmas@carabos.com>
#             Carabos Coop. V.
# 2007-10-08, try/catch 

"""Adaptors to interchange data with numpy and/or PIL

This module provides explicit conversion of OpenCV images/matrices to and from
the Python Imaging Library (PIL) and python's newest numeric library (numpy).

Currently supported image/matrix formats are:
    - 3 x  8 bit  RGB (GBR)
    - 1 x  8 bit  Grayscale
    - 1 x 32 bit  Float

In numpy, images are represented as multidimensional arrays with
a third dimension representing the image channels if more than one
channel is present.
"""

import cv

try:
  import PIL.Image

  ###########################################################################
  def Ipl2PIL(input):
      """Converts an OpenCV/IPL image to PIL the Python Imaging Library.
  
      Supported input image formats are
         IPL_DEPTH_8U  x 1 channel
         IPL_DEPTH_8U  x 3 channels
         IPL_DEPTH_32F x 1 channel
      """
  
      if not isinstance(input, cv.CvMat):
          raise TypeError, 'must be called with a cv.CvMat!'
    
      #orientation
      if input.origin == 0:
          orientation = 1 # top left
      elif input.origin == 1:
          orientation = -1 # bottom left
      else:
          raise ValueError, 'origin must be 0 or 1!'
  
      # mode dictionary:
      # (channels, depth) : (source mode, dest mode, depth in byte)
      mode_list = {
          (1, cv.IPL_DEPTH_8U)  : ("L", "L", 1),
          (3, cv.IPL_DEPTH_8U)  : ("BGR", "RGB", 3),
          (1, cv.IPL_DEPTH_32F) : ("F", "F", 4)
          }
  
      key = (input.nChannels, input.depth)
      if not mode_list.has_key(key):
          raise ValueError, 'unknown or unsupported input mode'
  
      modes = mode_list[key]
  
      return PIL.Image.fromstring(
          modes[1], # mode
          (input.width, input.height), # size tuple
          input.imageData, # data
          "raw",
          modes[0], # raw mode
          input.widthStep, # stride
          orientation # orientation
          )
  
  
  ###########################################################################
  def PIL2Ipl(input):
      """Converts a PIL image to the OpenCV/IPL CvMat data format.
  
      Supported input image formats are:
          RGB
          L
          F
      """
  
      if not (isinstance(input, PIL.Image.Image)):
          raise TypeError, 'Must be called with PIL.Image.Image!'
      
      # mode dictionary:
      # (pil_mode : (ipl_depth, ipl_channels)
      mode_list = {
          "RGB" : (cv.IPL_DEPTH_8U, 3),
          "L"   : (cv.IPL_DEPTH_8U, 1),
          "F"   : (cv.IPL_DEPTH_32F, 1)
          }
      
      if not mode_list.has_key(input.mode):
          raise ValueError, 'unknown or unsupported input mode'
      
      result = cv.cvCreateImage(
          cv.cvSize(input.size[0], input.size[1]),  # size
          mode_list[input.mode][0],  # depth
          mode_list[input.mode][1]  # channels
          )
  
      # set imageData
      result.imageData = input.tostring()
      return result

except ImportError:
  pass


#############################################################################
#############################################################################

try:
  
  import numpy
  
  
  ###########################################################################
  def NumPy2Ipl(input):
      """Converts a numpy array to the OpenCV/IPL CvMat data format.
  
      Supported input array layouts:
         2 dimensions of numpy.uint8
         3 dimensions of numpy.uint8
         2 dimensions of numpy.float32
         2 dimensions of numpy.float64
      """
      
      if not isinstance(input, numpy.ndarray):
          raise TypeError, 'Must be called with numpy.ndarray!'
  
      # Check the number of dimensions of the input array
      ndim = input.ndim
      if not ndim in (2, 3):
          raise ValueError, 'Only 2D-arrays and 3D-arrays are supported!'
      
      # Get the number of channels
      if ndim == 2:
          channels = 1
      else:
          channels = input.shape[2]
      
      # Get the image depth
      if input.dtype == numpy.uint8:
          depth = cv.IPL_DEPTH_8U
      elif input.dtype == numpy.float32:
          depth = cv.IPL_DEPTH_32F
      elif input.dtype == numpy.float64:
          depth = cv.IPL_DEPTH_64F
      
      # supported modes list: [(channels, dtype), ...]
      modes_list = [(1, numpy.uint8), (3, numpy.uint8), (1, numpy.float32), (1, numpy.float64)]
      
      # Check if the input array layout is supported
      if not (channels, input.dtype) in modes_list:
          raise ValueError, 'Unknown or unsupported input mode'
      
      result = cv.cvCreateImage(
          cv.cvSize(input.shape[1], input.shape[0]),  # size
          depth,  # depth
          channels  # channels
          )
      
      # set imageData
      result.imageData = input.tostring()
      
      return result
  
  
  ###########################################################################
  def Ipl2NumPy(input):
      """Converts an OpenCV/IPL image to a numpy array.
  
      Supported input image formats are
         IPL_DEPTH_8U  x 1 channel
         IPL_DEPTH_8U  x 3 channels
         IPL_DEPTH_32F x 1 channel
         IPL_DEPTH_32F x 2 channels
         IPL_DEPTH_32S x 1 channel
         IPL_DEPTH_64F x 1 channel
         IPL_DEPTH_64F x 2 channels
      """
      
      if not isinstance(input, cv.CvMat):
          raise TypeError, 'must be called with a cv.CvMat!'
            
      # data type dictionary:
      # (channels, depth) : numpy dtype
      ipl2dtype = {
          (1, cv.IPL_DEPTH_8U)  : numpy.uint8,
          (3, cv.IPL_DEPTH_8U)  : numpy.uint8,
          (1, cv.IPL_DEPTH_32F) : numpy.float32,
          (2, cv.IPL_DEPTH_32F) : numpy.float32,
          (1, cv.IPL_DEPTH_32S) : numpy.int32,
          (1, cv.IPL_DEPTH_64F) : numpy.float64,
          (2, cv.IPL_DEPTH_64F) : numpy.float64
          }
      
      key = (input.nChannels, input.depth)
      if not ipl2dtype.has_key(key):
          raise ValueError, 'unknown or unsupported input mode'
      
      # Get the numpy array and reshape it correctly
      # ATTENTION: flipped dimensions width/height on 2007-11-15
      if input.nChannels == 1:
          array_1d = numpy.fromstring(input.imageData, dtype=ipl2dtype[key])
          return numpy.reshape(array_1d, (input.height, input.width))
      elif input.nChannels == 2:
          array_1d = numpy.fromstring(input.imageData, dtype=ipl2dtype[key])
          return numpy.reshape(array_1d, (input.height, input.width, 2))
      elif input.nChannels == 3:
          # Change the order of channels from BGR to RGB
          rgb = cv.cvCreateImage(cv.cvSize(input.width, input.height), input.depth, 3)
          cv.cvCvtColor(input, rgb, cv.CV_BGR2RGB)
          array_1d = numpy.fromstring(rgb.imageData, dtype=ipl2dtype[key])
          return numpy.reshape(array_1d, (input.height, input.width, 3))

except ImportError:
  pass


###########################################################################
###########################################################################


try:

  import PIL.Image
  import numpy

  ###########################################################################
  def PIL2NumPy(input):
      """THIS METHOD IS DEPRECATED
      
      Converts a PIL image to a numpy array.
  
      Supported input image formats are:
          RGB
          L
          F
      """
  
      if not (isinstance(input, PIL.Image.Image)):
          raise TypeError, 'Must be called with PIL.Image.Image!'
  
      # modes dictionary:
      # pil_mode : numpy dtype
      modes_map = {
          "RGB" : numpy.uint8,
          "L"   : numpy.uint8,
          "F"   : numpy.float32
          }
  
      if not modes_map.has_key(input.mode):
          raise ValueError, 'Unknown or unsupported input mode!. Supported modes are RGB, L and F.'
  
      result_ro = numpy.asarray(input, dtype=modes_map[input.mode])  # Read-only array
      return result_ro.copy()  # Return a writeable array
  
  
  ###########################################################################
  def NumPy2PIL(input):
      """THIS METHOD IS DEPRECATED
      
      Converts a numpy array to a PIL image.
  
      Supported input array layouts:
         2 dimensions of numpy.uint8
         3 dimensions of numpy.uint8
         2 dimensions of numpy.float32
      """
  
      if not isinstance(input, numpy.ndarray):
          raise TypeError, 'Must be called with numpy.ndarray!'
  
      # Check the number of dimensions of the input array
      ndim = input.ndim
      if not ndim in (2, 3):
          raise ValueError, 'Only 2D-arrays and 3D-arrays are supported!'
  
      if ndim == 2:
          channels = 1
      else:
          channels = input.shape[2]
  
      # supported modes list: [(channels, dtype), ...]
      modes_list = [(1, numpy.uint8), (3, numpy.uint8), (1, numpy.float32)]
  
      mode = (channels, input.dtype)
      if not mode in modes_list:
          raise ValueError, 'Unknown or unsupported input mode'
  
      return PIL.Image.fromarray(input)

except ImportError:
  pass
