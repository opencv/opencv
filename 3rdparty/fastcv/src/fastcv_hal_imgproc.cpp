/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "fastcv_hal_imgproc.hpp"
#include "fastcv_hal_utils.hpp"
#include <opencv2/core/base.hpp>
#include <opencv2/imgproc.hpp>


int fastcv_hal_medianBlur(
    const uchar*    src_data,
    size_t          src_step,
    uchar*          dst_data,
    size_t          dst_step,
    int             width,
    int             height,
    int             depth,
    int             cn,
    int             ksize)
{
    // Do not support inplace case
    if (src_data == dst_data)
        CV_HAL_RETURN_NOT_IMPLEMENTED("Inplace is not supported");

    // The input image width and height should greater than kernel size
    if ((height <= ksize) || (width <= ksize))
        CV_HAL_RETURN_NOT_IMPLEMENTED("Input image size should be larger than kernel size");

    // The input channel should be 1
    if (cn != 1)
        CV_HAL_RETURN_NOT_IMPLEMENTED("Multi-channels is not supported");

    INITIALIZATION_CHECK;

    fcvStatus status;
    int fcvFuncType = FCV_MAKETYPE(ksize,depth);

    switch (fcvFuncType)
    {
        case FCV_MAKETYPE(3,CV_8U):
        {
            status = fcvFilterMedian3x3u8_v3(src_data, width, height, src_step, dst_data, dst_step,
                fcvBorderType::FASTCV_BORDER_REPLICATE, 0);
            break;
        }
        default:
            CV_HAL_RETURN_NOT_IMPLEMENTED(cv::format("Ksize:%d, depth:%s is not supported", ksize, cv::depthToString(depth)));
    }

    CV_HAL_RETURN(status, hal_medianBlur);
}

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
    int             border_type)
{

    if(scale != 1.0f || delta != 0.0f)
        CV_HAL_RETURN_NOT_IMPLEMENTED(cv::format("Scale:%f, delta:%f is not supported", scale, delta));

    // Do not support inplace case
    if (src_data == dst_data)
        CV_HAL_RETURN_NOT_IMPLEMENTED("Inplace is not supported");

    // The input image width and height should greater than kernel size
    if ((height <= ksize) || (width <= ksize))
        CV_HAL_RETURN_NOT_IMPLEMENTED("Input image size should be larger than kernel size");

    // The input channel should be 1
    if (cn != 1)
        CV_HAL_RETURN_NOT_IMPLEMENTED("Multi-channels is not supported");

    // Do not support for ROI case
    if((margin_left!=0) || (margin_top != 0) || (margin_right != 0) || (margin_bottom !=0))
        CV_HAL_RETURN_NOT_IMPLEMENTED("ROI is not supported");

    // 1. When ksize <= 0, OpenCV will use Scharr Derivatives
    // 2. When ksize == 1, OpenCV will use 3×1 or 1×3 kernel(no Gaussian smoothing is done)
    // FastCV doesn't support above two situation
    if (ksize <= 1)
        CV_HAL_RETURN_NOT_IMPLEMENTED("Scharr derivatives or non square kernel are not supported");

    // Only support the result type is CV_16S
    if (dst_depth != CV_16S)
        CV_HAL_RETURN_NOT_IMPLEMENTED(cv::format("Dst depth:%s is not supported", cv::depthToString(dst_depth)));

    INITIALIZATION_CHECK;

    // Only support one direction derivatives and the order is 1.(dx=1 && dy=0)||(dx=0 && dy=1)
    int16_t *dxBuffer, *dyBuffer;

    if ((dx == 1) && (dy == 0))
    {
        dxBuffer = (int16_t*)dst_data;
        dyBuffer = NULL;
    }
    else if ((dx == 0) && (dy == 1))
    {
        dxBuffer = NULL;
        dyBuffer = (int16_t*)dst_data;
    }
    else
        CV_HAL_RETURN_NOT_IMPLEMENTED(cv::format("Dx:%d Dy:%d is not supported",dx, dy));

    fcvStatus       status;
    fcvBorderType   fcvBorder;

    switch (border_type)
    {
        // For constant border, there are no border value, OpenCV default value is 0
        case cv::BorderTypes::BORDER_CONSTANT:
        {
            fcvBorder = fcvBorderType::FASTCV_BORDER_CONSTANT;
            break;
        }
        case cv::BorderTypes::BORDER_REPLICATE:
        {
            fcvBorder = fcvBorderType::FASTCV_BORDER_REPLICATE;
            break;
        }
        default:
            CV_HAL_RETURN_NOT_IMPLEMENTED(cv::format("Border type:%s is not supported", borderToString(border_type)));
    }

    int fcvFuncType = FCV_MAKETYPE(ksize,src_depth);

    switch (fcvFuncType)
    {
        case FCV_MAKETYPE(3,CV_8U):
        {
            status = fcvFilterSobel3x3u8s16(src_data, width, height, src_step, dxBuffer, dyBuffer, dst_step, fcvBorder, 0);
            break;
        }
        case FCV_MAKETYPE(5,CV_8U):
        {
            status = fcvFilterSobel5x5u8s16(src_data, width, height, src_step, dxBuffer, dyBuffer, dst_step, fcvBorder, 0);
            break;
        }
        case FCV_MAKETYPE(7,CV_8U):
        {
            status = fcvFilterSobel7x7u8s16(src_data, width, height, src_step, dxBuffer, dyBuffer, dst_step, fcvBorder, 0);
            break;
        }
        default:
            CV_HAL_RETURN_NOT_IMPLEMENTED(cv::format("Ksize:%d, src_depth:%s, border type:%s is not supported",
                ksize, cv::depthToString(src_depth), borderToString(border_type)));
    }

    CV_HAL_RETURN(status, hal_sobel);
}

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
    int              border_type)
{
    if((width*height) < (320*240))
    {
        CV_HAL_RETURN_NOT_IMPLEMENTED("input size not supported");
    }
    else if(src_data == dst_data)
    {
        CV_HAL_RETURN_NOT_IMPLEMENTED("in-place processing not supported");
    }
    else if(src_depth != CV_8U || cn != 1)
    {
        CV_HAL_RETURN_NOT_IMPLEMENTED("src type not supported");
    }
    else if(dst_depth != src_depth)
    {
        CV_HAL_RETURN_NOT_IMPLEMENTED("same src and dst type supported");
    }
    else if(ksize_width != ksize_height ||
           (ksize_width != 3 && ksize_width != 5))
    {
        CV_HAL_RETURN_NOT_IMPLEMENTED("kernel size not supported");
    }
    else if(anchor_x != -1 || anchor_y != -1 ||
            margin_left != 0 || margin_top != 0 ||
            margin_right != 0 || margin_bottom != 0)
    {
        CV_HAL_RETURN_NOT_IMPLEMENTED("ROI not supported");
    }

    INITIALIZATION_CHECK;

    fcvBorderType bdr;
    uint8_t bdrVal = 0;
    switch(border_type)
    {
        case cv::BORDER_REPLICATE:
            bdr = FASTCV_BORDER_REPLICATE;
            break;
        case cv::BORDER_REFLECT:
            bdr = FASTCV_BORDER_REFLECT;
            break;
        case cv::BORDER_REFLECT101:    // cv::BORDER_REFLECT_101, BORDER_DEFAULT
            bdr = FASTCV_BORDER_REFLECT_V2;
            break;
        default:
            CV_HAL_RETURN_NOT_IMPLEMENTED("border type not supported");
    }

    fcvStatus status = FASTCV_SUCCESS;
    if(ksize_width == 3)
    {
        status = fcvBoxFilter3x3u8_v3(src_data, width, height, src_step,
                                      dst_data, dst_step, normalize, bdr, bdrVal);
    }
    else if(ksize_width == 5)
    {
        status = fcvBoxFilter5x5u8_v2(src_data, width, height, src_step,
                                      dst_data, dst_step, normalize, bdr, bdrVal);
    }

    CV_HAL_RETURN(status,hal_boxFilter);
}

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
    double          C)
{
    if((width*height) < (320*240))
    {
        CV_HAL_RETURN_NOT_IMPLEMENTED("input size not supported");
    }
    else if (src_data == dst_data)
    {
        CV_HAL_RETURN_NOT_IMPLEMENTED("In place processing not supported");
    }

    int value  = (thresholdType == cv::THRESH_BINARY) ? cvCeil(C) : cvFloor(C);

    if ((maxValue < 1) || (maxValue > 255))
    {
        CV_HAL_RETURN_NOT_IMPLEMENTED("max value 1-255 supported");
    }

    INITIALIZATION_CHECK;

    uchar maxVal  = cv::saturate_cast<uchar>(maxValue);

    fcvThreshType threshType = (thresholdType == cv::THRESH_BINARY) ? FCV_THRESH_BINARY : FCV_THRESH_BINARY_INV;

    fcvStatus status = FASTCV_SUCCESS;
    if(adaptiveMethod == cv::ADAPTIVE_THRESH_GAUSSIAN_C)
    {
        if(blockSize == 3)
            status = fcvAdaptiveThresholdGaussian3x3u8_v2(src_data, width, height, src_step, maxVal, threshType, value, dst_data, dst_step);
        else if(blockSize == 5)
            status = fcvAdaptiveThresholdGaussian5x5u8_v2(src_data, width, height, src_step, maxVal, threshType, value, dst_data, dst_step);
        else
        {
            CV_HAL_RETURN_NOT_IMPLEMENTED("block size not supported");
        }
    }
    else if(adaptiveMethod == cv::ADAPTIVE_THRESH_MEAN_C)
    {
        if(blockSize == 3)
            status = fcvAdaptiveThresholdMean3x3u8_v2(src_data, width, height, src_step, maxVal, threshType, value, dst_data, dst_step);
        else if(blockSize == 5)
            status = fcvAdaptiveThresholdMean5x5u8_v2(src_data, width, height, src_step, maxVal, threshType, value, dst_data, dst_step);
        else
        {
            CV_HAL_RETURN_NOT_IMPLEMENTED("block size not supported");
        }
    }
    else
    {
        CV_HAL_RETURN_NOT_IMPLEMENTED("adaptive method not supported");
    }

    CV_HAL_RETURN(status,hal_adaptiveThreshold);
}
