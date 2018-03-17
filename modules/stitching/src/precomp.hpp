/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#ifndef __OPENCV_STITCHING_PRECOMP_H__
#define __OPENCV_STITCHING_PRECOMP_H__

#include "opencv2/opencv_modules.hpp"

#include <vector>
#include <algorithm>
#include <utility>
#include <set>
#include <functional>
#include <sstream>
#include <iostream>
#include <cmath>
#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"

#ifdef HAVE_OPENCV_CUDAARITHM
#  include "opencv2/cudaarithm.hpp"
#endif

#ifdef HAVE_OPENCV_CUDAWARPING
#  include "opencv2/cudawarping.hpp"
#endif

#ifdef HAVE_OPENCV_CUDAFEATURES2D
#  include "opencv2/cudafeatures2d.hpp"
#endif

#ifdef HAVE_OPENCV_CUDALEGACY
#  include "opencv2/cudalegacy.hpp"
#endif

#ifdef HAVE_OPENCV_XFEATURES2D
#  include "opencv2/xfeatures2d/cuda.hpp"
#endif

#include "../../imgproc/src/gcgraph.hpp"

#include "opencv2/core/private.hpp"

#ifdef HAVE_TEGRA_OPTIMIZATION
# include "opencv2/stitching/stitching_tegra.hpp"
#endif

#include "util_log.hpp"

#endif
