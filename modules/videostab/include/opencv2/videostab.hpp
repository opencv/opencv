/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_VIDEOSTAB_HPP
#define OPENCV_VIDEOSTAB_HPP

/**
  @defgroup videostab Video Stabilization

The video stabilization module contains a set of functions and classes that can be used to solve the
problem of video stabilization. There are a few methods implemented, most of them are described in
the papers @cite OF06 and @cite G11 . However, there are some extensions and deviations from the original
paper methods.

### References

 1. "Full-Frame Video Stabilization with Motion Inpainting"
     Yasuyuki Matsushita, Eyal Ofek, Weina Ge, Xiaoou Tang, Senior Member, and Heung-Yeung Shum
 2. "Auto-Directed Video Stabilization with Robust L1 Optimal Camera Paths"
     Matthias Grundmann, Vivek Kwatra, Irfan Essa

     @{
         @defgroup videostab_motion Global Motion Estimation

The video stabilization module contains a set of functions and classes for global motion estimation
between point clouds or between images. In the last case features are extracted and matched
internally. For the sake of convenience the motion estimation functions are wrapped into classes.
Both the functions and the classes are available.

         @defgroup videostab_marching Fast Marching Method

The Fast Marching Method @cite Telea04 is used in of the video stabilization routines to do motion and
color inpainting. The method is implemented is a flexible way and it's made public for other users.

     @}

*/

#include "opencv2/videostab/stabilizer.hpp"
#include "opencv2/videostab/ring_buffer.hpp"

#endif
