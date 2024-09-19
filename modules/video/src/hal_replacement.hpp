// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_VIDEO_HAL_REPLACEMENT_HPP
#define OPENCV_VIDEO_HAL_REPLACEMENT_HPP

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

//! @addtogroup video_hal_interface
//! @note Define your functions to override default implementations:
//! @code
//! #undef cv_hal_LK_optical_flow_level
//! #define cv_hal_LK_optical_flow_level my_hal_LK_optical_flow_level
//! @endcode
//! @{

/**
@brief Lucas-Kanade optical flow for single pyramid layer. See calcOpticalFlowPyrLK
@param prev_data previous frame image data
@param prev_data_step previous frame image data step
@param prev_deriv_data previous frame Schaar derivatives
@param prev_deriv_step previous frame Schaar derivatives step
@param next_data next frame image data
@param next_step next frame image step
@param width input images width
@param height input images height
@param cn source image channels
@param prev_points 2d points coordinates (x,y) on the previous frame
@param next_points points coordinates (x,y) on the next frame
@param point_count - amount of input points
@param status optical flow status for each point. Optional output, expected if not nullptr is provided
@param err optical flow estimation error for each point. Optional output, expected if not nullptr is provided
@param win_width optical flow window width
@param win_height optical flow window heigh
@param termination_count maximum algorithm iterations. 0 means unlimited
@param termination_epsilon maximal allowed algorithm error
@param get_min_eigen_vals return minimal egen values as point errors in err buffer
@param min_eigen_vals_threshold eigen values threshold
**/
inline int hal_ni_LKOpticalFlowLevel(const uchar *prev_data, size_t prev_data_step,
                       const short* prev_deriv_data, size_t prev_deriv_step,
                       const uchar* next_data, size_t next_step,
                       int width, int height, int cn,
                       const float *prev_points, float *next_points, size_t point_count,
                       uchar *status, float *err,
                       const int win_width, const int win_height,
                       int termination_count, double termination_epsilon,
                       bool get_min_eigen_vals,
                       float min_eigen_vals_threshold)
{
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

//! @cond IGNORED
#define cv_hal_LKOpticalFlowLevel hal_ni_LKOpticalFlowLevel
//! @endcond

/**
@brief Computes Schaar derivatives with inteleaved layout xyxy...
@param src_data source image data
@param src_step source image step
@param dst_data destination buffer data
@param dst_step destination buffer step
@param width image width
@param height image height
@param cn source image channels
**/
inline int hal_ni_ScharrDeriv(const uchar* src_data, size_t src_step,
                              short* dst_data, size_t dst_step,
                              int width, int height, int cn)
{
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

//! @cond IGNORED
#define cv_hal_ScharrDeriv hal_ni_ScharrDeriv
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
