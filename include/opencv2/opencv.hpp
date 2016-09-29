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
// Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
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

#ifndef OPENCV_ALL_HPP
#define OPENCV_ALL_HPP

// File that defines what modules where included during the build of OpenCV
// These are purely the defines of the correct HAVE_OPENCV_modulename values
#include "opencv2/opencv_modules.hpp"

// Then the list of defines is checked to include the correct headers
// Core library is always included --> without no OpenCV functionality available
#include "opencv2/core.hpp"

// Then the optional modules are checked
#ifdef HAVE_OPENCV_CALIB3D
#include "opencv2/calib3d.hpp"
#endif
#ifdef HAVE_OPENCV_FEATURES2D
#include "opencv2/features2d.hpp"
#endif
#ifdef HAVE_OPENCV_FLANN
#include "opencv2/flann.hpp"
#endif
#ifdef HAVE_OPENCV_HIGHGUI
#include "opencv2/highgui.hpp"
#endif
#ifdef HAVE_OPENCV_IMGCODECS
#include "opencv2/imgcodecs.hpp"
#endif
#ifdef HAVE_OPENCV_IMGPROC
#include "opencv2/imgproc.hpp"
#endif
#ifdef HAVE_OPENCV_ML
#include "opencv2/ml.hpp"
#endif
#ifdef HAVE_OPENCV_OBJDETECT
#include "opencv2/objdetect.hpp"
#endif
#ifdef HAVE_OPENCV_PHOTO
#include "opencv2/photo.hpp"
#endif
#ifdef HAVE_OPENCV_SHAPE
#include "opencv2/shape.hpp"
#endif
#ifdef HAVE_OPENCV_STITCHING
#include "opencv2/stitching.hpp"
#endif
#ifdef HAVE_OPENCV_SUPERRES
#include "opencv2/superres.hpp"
#endif
#ifdef HAVE_OPENCV_VIDEO
#include "opencv2/video.hpp"
#endif
#ifdef HAVE_OPENCV_VIDEOIO
#include "opencv2/videoio.hpp"
#endif
#ifdef HAVE_OPENCV_VIDEOSTAB
#include "opencv2/videostab.hpp"
#endif
#ifdef HAVE_OPENCV_VIZ
#include "opencv2/viz.hpp"
#endif

// Finally CUDA specific entries are checked and added
#ifdef HAVE_OPENCV_CUDAARITHM
#include "opencv2/cudaarithm.hpp"
#endif
#ifdef HAVE_OPENCV_CUDABGSEGM
#include "opencv2/cudabgsegm.hpp"
#endif
#ifdef HAVE_OPENCV_CUDACODEC
#include "opencv2/cudacodec.hpp"
#endif
#ifdef HAVE_OPENCV_CUDAFEATURES2D
#include "opencv2/cudafeatures2d.hpp"
#endif
#ifdef HAVE_OPENCV_CUDAFILTERS
#include "opencv2/cudafilters.hpp"
#endif
#ifdef HAVE_OPENCV_CUDAIMGPROC
#include "opencv2/cudaimgproc.hpp"
#endif
#ifdef HAVE_OPENCV_CUDAOBJDETECT
#include "opencv2/cudaobjdetect.hpp"
#endif
#ifdef HAVE_OPENCV_CUDAOPTFLOW
#include "opencv2/cudaoptflow.hpp"
#endif
#ifdef HAVE_OPENCV_CUDASTEREO
#include "opencv2/cudastereo.hpp"
#endif
#ifdef HAVE_OPENCV_CUDAWARPING
#include "opencv2/cudawarping.hpp"
#endif

#endif
