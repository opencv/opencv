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

#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__

#if _MSC_VER >= 1200
    #pragma warning( disable: 4251 4710 4711 4514 4996 )
#endif

#ifdef HAVE_CVCONFIG_H
    #include "cvconfig.h"
#endif

#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>
#include <sstream>
#include <exception>
#include <iterator>
#include <functional>
#include <utility>

#include "opencv2/gpu/gpu.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/internal.hpp"
#include "opencv2/video/video.hpp"

#define OPENCV_GPU_UNUSED(x) (void)x

#ifdef HAVE_CUDA

    #include "cuda.h"
    #include "cuda_runtime_api.h"
    #include "npp.h"

    #ifdef HAVE_CUFFT
        #include "cufft.h"
    #endif

    #ifdef HAVE_CUBLAS
        #include "cublas.h"
    #endif

    #include "internal_shared.hpp"
    #include "opencv2/gpu/stream_accessor.hpp"

    #include "nvidia/core/NCV.hpp"
    #include "nvidia/NPP_staging/NPP_staging.hpp"
    #include "nvidia/NCVHaarObjectDetection.hpp"
    #include "nvidia/NCVBroxOpticalFlow.hpp"

    #define CUDART_MINIMUM_REQUIRED_VERSION 4010
    #define NPP_MINIMUM_REQUIRED_VERSION 4100

    #if (CUDART_VERSION < CUDART_MINIMUM_REQUIRED_VERSION)
        #error "Insufficient Cuda Runtime library version, please update it."
    #endif

    #if (NPP_VERSION_MAJOR * 1000 + NPP_VERSION_MINOR * 100 + NPP_VERSION_BUILD < NPP_MINIMUM_REQUIRED_VERSION)
        #error "Insufficient NPP version, please update it."
    #endif

    #if defined(CUDA_ARCH_BIN_OR_PTX_10)
        #error "OpenCV GPU module doesn't support NVIDIA compute capability 1.0"
    #endif

    static inline void throw_nogpu() { CV_Error(CV_StsNotImplemented, "The called functionality is disabled for current build or platform"); }

#else /* defined(HAVE_CUDA) */

    static inline void throw_nogpu() { CV_Error(CV_GpuNotSupported, "The library is compiled without GPU support"); }

#endif /* defined(HAVE_CUDA) */

#endif /* __OPENCV_PRECOMP_H__ */
