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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
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

#ifndef OPENCV_CALIB3D_HAL_REPLACEMENT_HPP
#define OPENCV_CALIB3D_HAL_REPLACEMENT_HPP

#include "opencv2/core/hal/interface.h"

#if defined(__clang__)  // clang or MSVC clang
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

//! @addtogroup calib3d_hal_interface
//! @note Define your functions to override default implementations:
//! @code
//! #undef hal_add8u
//! #define hal_add8u my_add8u
//! @endcode
//! @{

/**
   @brief Project points from 3D world space to 2D screen space using rotation and translation matrix and camera intrinsic parameters
   @param src_data Pointer to source 3D points array
   @param src_step1 Step between X, Y and Z coordinates of the same point in 3D points array
   @param src_step2 Step between consecutive points in 3D points array
   @param src_size Amount of points in 3D points array
   @param dst_data Pointer to resulting 2D points array
   @param dst_step1 Step between x and y coordinates of the same point in resulting 2D array
   @param dst_step2 Step between consecutive points in resulting 2D array
   @param rt_data Pointer to 3x4 array containing rotation-then-translation matrix
   @param intrinsics_data Pointer to camera intrinsic parameters vector containing [fx, fy, cx, cy]
   @param distortion_data Distortion coefficients in the same order as in OpenCV, set to zero if not used: [k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6, s_1, s_2, s_3, s_4, tau_x, tau_y]
*/
inline int hal_ni_project_points_pinhole32f(const float* src_data, size_t src_step1, size_t src_step2, size_t src_size,
                                            float* dst_data, size_t dst_step1, size_t dst_step2,
                                            const float* rt_data, const float* intrinsics_data, const float* distortion_data)
{ return CV_HAL_ERROR_NOT_IMPLEMENTED; }

//! @cond IGNORED
#define cv_hal_project_points_pinhole32f hal_ni_project_points_pinhole32f
//! @endcond

/**
   @brief Project points from 3D world space to 2D screen space using rotation and translation matrix and camera intrinsic parameters
   @param src_data Pointer to source 3D points array
   @param src_step1 Step between X, Y and Z coordinates of the same point in 3D points array
   @param src_step2 Step between consecutive points in 3D points array
   @param src_size Amount of points in 3D points array
   @param dst_data Pointer to resulting 2D points array
   @param dst_step1 Step between x and y coordinates of the same point in resulting 2D array
   @param dst_step2 Step between consecutive points in resulting 2D array
   @param rt_data Pointer to 3x4 array containing rotation-then-translation matrix
   @param intrinsics_data Pointer to camera intrinsic parameters vector containing [fx, fy, cx, cy]
   @param distortion_data Distortion coefficients in the same order as in OpenCV, set to zero if not used: [k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6, s_1, s_2, s_3, s_4, tau_x, tau_y]
*/
inline int hal_ni_project_points_pinhole64f(const double* src_data, size_t src_step1, size_t src_step2, size_t src_size,
                                            double* dst_data, size_t dst_step1, size_t dst_step2,
                                            const double* rt_data, const double* intrinsics_data, const double* distortion_data)
{ return CV_HAL_ERROR_NOT_IMPLEMENTED; }

//! @cond IGNORED
#define cv_hal_project_points_pinhole64f hal_ni_project_points_pinhole64f
//! @endcond

//! @}

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include "custom_hal.hpp"

//! @cond IGNORED
#define CALL_HAL_RET(name, fun, retval, ...) \
    int res = __CV_EXPAND(fun(__VA_ARGS__, &retval)); \
    if (res == CV_HAL_ERROR_OK) \
        return retval; \
    else if (res != CV_HAL_ERROR_NOT_IMPLEMENTED) \
        CV_Error_(cv::Error::StsInternal, \
            ("HAL implementation " CVAUX_STR(name) " ==> " CVAUX_STR(fun) " returned %d (0x%08x)", res, res));


#define CALL_HAL(name, fun, ...) \
    int res = __CV_EXPAND(fun(__VA_ARGS__)); \
    if (res == CV_HAL_ERROR_OK) \
        return; \
    else if (res != CV_HAL_ERROR_NOT_IMPLEMENTED) \
        CV_Error_(cv::Error::StsInternal, \
            ("HAL implementation " CVAUX_STR(name) " ==> " CVAUX_STR(fun) " returned %d (0x%08x)", res, res));
//! @endcond

#endif
