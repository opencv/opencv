/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0  
*/

#ifndef OPENCV_FASTCV_HAL_IMGPROC_HPP_INCLUDED
#define OPENCV_FASTCV_HAL_IMGPROC_HPP_INCLUDED

#include <opencv2/core/hal/interface.h>

#undef  cv_hal_medianBlur
#define cv_hal_medianBlur           fastcv_hal_medianBlur
#undef  cv_hal_sobel
#define cv_hal_sobel                fastcv_hal_sobel
#undef cv_hal_boxFilter
#define cv_hal_boxFilter            fastcv_hal_boxFilter
#undef cv_hal_adaptiveThreshold
#define cv_hal_adaptiveThreshold    fastcv_hal_adaptiveThreshold

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief Calculate medianBlur filter
/// @param src_data Source image data
/// @param src_step Source image step
/// @param dst_data Destination image data
/// @param dst_step Destination image step
/// @param width    Source image width
/// @param height   Source image height
/// @param depth    Depths of source and destination image
/// @param cn       Number of channels
/// @param ksize    Size of kernel
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int fastcv_hal_medianBlur(
    const uchar*    src_data,
    size_t          src_step,
    uchar*          dst_data,
    size_t          dst_step,
    int             width,
    int             height,
    int             depth,
    int             cn,
    int             ksize);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief Computes Sobel derivatives
///
/// @param src_data         Source image data
/// @param src_step         Source image step
/// @param dst_data         Destination image data
/// @param dst_step         Destination image step
/// @param width            Source image width
/// @param height           Source image height
/// @param src_depth        Depth of source image
/// @param dst_depth        Depths of destination image
/// @param cn               Number of channels
/// @param margin_left      Left margins for source image
/// @param margin_top       Top margins for source image
/// @param margin_right     Right margins for source image
/// @param margin_bottom    Bottom margins for source image
/// @param dx               orders of the derivative x
/// @param dy               orders of the derivative y
/// @param ksize            Size of kernel
/// @param scale            Scale factor for the computed derivative values
/// @param delta            Delta value that is added to the results prior to storing them in dst
/// @param border_type      Border type
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int fastcv_hal_sobel(
    const uchar*    src_data,
    size_t          src_step,
    uchar*          dst_data,
    size_t          dst_step,
    int             width,
    int             height,
    int             src_depth,
    int             dst_depth,
    int             cn,
    int             margin_left,
    int             margin_top,
    int             margin_right,
    int             margin_bottom,
    int             dx,
    int             dy,
    int             ksize,
    double          scale,
    double          delta,
    int             border_type);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief Canny edge detector
/// @param src_data Source image data
/// @param src_step Source image step
/// @param dst_data Destination image data
/// @param dst_step Destination image step
/// @param width Source image width
/// @param height Source image height
/// @param cn Number of channels
/// @param lowThreshold low thresholds value
/// @param highThreshold high thresholds value
/// @param ksize Kernel size for Sobel operator.
/// @param L2gradient Flag, indicating use L2 or L1 norma.
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int fastcv_hal_canny(
    const uchar*    src_data,
    size_t          src_step,
    uchar*          dst_data,
    size_t          dst_step,
    int             width,
    int             height,
    int             cn,
    double          lowThreshold,
    double          highThreshold,
    int             ksize,
    bool            L2gradient);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int fastcv_hal_boxFilter(
    const uchar*     src_data,
    size_t           src_step,
    uchar*           dst_data,
    size_t           dst_step,
    int              width,
    int              height,
    int              src_depth,
    int              dst_depth,
    int              cn,
    int              margin_left,
    int              margin_top,
    int              margin_right,
    int              margin_bottom,
    size_t           ksize_width,
    size_t           ksize_height,
    int              anchor_x,
    int              anchor_y,
    bool             normalize,
    int              border_type);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int fastcv_hal_adaptiveThreshold(
    const uchar*    src_data,
    size_t          src_step,
    uchar*          dst_data,
    size_t          dst_step,
    int             width,
    int             height,
    double          maxValue,
    int             adaptiveMethod,
    int             thresholdType,
    int             blockSize,
    double          C);

#endif
