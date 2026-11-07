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
 * @brief Camera intrinsics structure, see projectPoints() documentation for details
 */
struct cv_camera_intrinsics_pinhole_32f
{
    // focal length, principal point
    float fx, fy, cx, cy;
    // radial distortion coefficients
    float k[6];
    // amount of radial distortion coefficients passed
    int amt_k;
    // tangential distortion coefficients
    float p[2];
    // amount of tangential distortion coefficients passed
    int amt_p;
    // prism distortion coefficients
    float s[4];
    // amount of prism distortion coefficients passed
    int amt_s;
    // tilt distortion coefficients
    float tau_x, tau_y;
    // to use tilt distortion coefficients or not
    bool use_tau;
};

/**
   @brief Project points from 3D world space to 2D screen space using rotation and translation matrix and camera intrinsic parameters
   @param src_data Pointer to 3D points array with coordinates interleaved as X, Y, Z, X, Y, Z,..
   @param src_step Step between consecutive 3D points
   @param src_size Amount of points
   @param dst_data Pointer to resulting projected 2D points with coordinates interleaved as u, v, u, v,..
   @param dst_step Step between consecutive projected 2D points
   @param rt_data Pointer to 3x4 array containing rotation-then-translation matrix
   @param intr_data Pointer to camera intrinsics structure
*/
inline int hal_ni_project_points_pinhole32f(const float* src_data, size_t src_step, size_t src_size,
                                            float* dst_data, size_t dst_step, const float* rt_data,
                                            const cv_camera_intrinsics_pinhole_32f* intr_data)
{ return CV_HAL_ERROR_NOT_IMPLEMENTED; }

//! @cond IGNORED
#define cv_hal_project_points_pinhole32f hal_ni_project_points_pinhole32f
//! @endcond

/**
 * @brief Camera intrinsics structure, see projectPoints() documentation for details
 */
struct cv_camera_intrinsics_pinhole_64f
{
    // focal length, principal point
    double fx, fy, cx, cy;
    // radial distortion coefficients
    double k[6];
    // amount of radial distortion coefficients passed
    int amt_k;
    // tangential distortion coefficients
    double p[2];
    // amount of tangential distortion coefficients passed
    int amt_p;
    // prism distortion coefficients
    double s[4];
    // amount of prism distortion coefficients passed
    int amt_s;
    // tilt distortion coefficients
    double tau_x, tau_y;
    // to use tilt distortion coefficients or not
    bool use_tau;
};

/**
   @brief Project points from 3D world space to 2D screen space using rotation and translation matrix and camera intrinsic parameters
   @param src_data Pointer to 3D points array with coordinates interleaved as X, Y, Z, X, Y, Z,..
   @param src_step Step between consecutive 3D points
   @param src_size Amount of points
   @param dst_data Pointer to resulting projected 2D points with coordinates interleaved as u, v, u, v,..
   @param dst_step Step between consecutive projected 2D points
   @param rt_data Pointer to 3x4 array containing rotation-then-translation matrix
   @param intr_data Pointer to camera intrinsics structure
*/
inline int hal_ni_project_points_pinhole64f(const double* src_data, size_t src_step, size_t src_size,
                                            double* dst_data, size_t dst_step, const double* rt_data,
                                            const cv_camera_intrinsics_pinhole_64f* intr_data)
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
