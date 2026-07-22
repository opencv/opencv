// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "ipp_hal_imgproc.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry.hpp>  // DistanceTypes (DIST_L1/L2/C) live in the geometry module on 5.x

#include <climits>
#include <cmath>

#if IPP_VERSION_X100 >= 700

// Mirrors CV_IPP_MALLOC from modules/core/include/opencv2/core/private.hpp: ippicv (IPP >= 2017)
// exposes only the 64-bit ippMalloc_L.
#if IPP_VERSION_X100 >= 201700
#define IPP_HAL_MALLOC(SIZE) ippMalloc_L(SIZE)
#else
#define IPP_HAL_MALLOC(SIZE) ippMalloc((int)(SIZE))
#endif

#ifndef IPP_DISABLE_PERF_TRUE_DIST_MT
#define IPP_DISABLE_PERF_TRUE_DIST_MT   1
#endif

// Fixed distance-transform mask weights, mirroring getDistanceTransformMask() in
// modules/imgproc/src/distransform.cpp. maskType == distTypeIndex + mask_size*10.
static bool ippGetDistanceTransformMask(int dist_type, int mask_size, float *metrics)
{
    int maskType = (dist_type == cv::DIST_C ? 0 : dist_type == cv::DIST_L1 ? 1 : 2) + mask_size * 10;
    switch (maskType)
    {
    case 30: metrics[0] = 1.0f;   metrics[1] = 1.0f;                        break;
    case 31: metrics[0] = 1.0f;   metrics[1] = 2.0f;                        break;
    case 32: metrics[0] = 0.955f; metrics[1] = 1.3693f;                     break;
    case 50: metrics[0] = 1.0f;   metrics[1] = 1.0f; metrics[2] = 2.0f;     break;
    case 51: metrics[0] = 1.0f;   metrics[1] = 2.0f; metrics[2] = 3.0f;     break;
    case 52: metrics[0] = 1.0f;   metrics[1] = 1.4f; metrics[2] = 2.1969f;  break;
    default:
        return false;
    }
    return true;
}

int ipp_hal_distanceTransform(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                              int width, int height, int dst_type, int dist_type, int mask_size)
{
    CV_HAL_CHECK_USE_IPP();

    IppiSize roi = { width, height };

    // DIST_L1 with 8-bit output: fixed L1 / 3x3 metrics.
    if (dst_type == CV_8U)
    {
        Ipp32s pMetrics[2] = { 1, 2 }; // L1, 3x3 mask
        if (CV_INSTRUMENT_FUN_IPP(ippiDistanceTransform_3x3_8u_C1R, src_data, (int)src_step,
                                  dst_data, (int)dst_step, roi, pMetrics) >= 0)
        {
            return CV_HAL_ERROR_OK;
        }
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    // All remaining paths produce a 32-bit float distance map.
    if (dst_type != CV_32F)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (mask_size == cv::DIST_MASK_PRECISE)
    {
        // 4097 can't square into a float.
#if IPP_DISABLE_PERF_TRUE_DIST_MT
        if (!((cv::getNumThreads() <= 1 || ((size_t)width * height < (size_t)(1 << 14))) &&
              height < 4097 && width < 4097))
        {
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }
#endif

        int bufSize = 0;
        if (ippiTrueDistanceTransformGetBufferSize_8u32f_C1R(roi, &bufSize) < 0)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        Ipp8u *pBuffer = (Ipp8u *)IPP_HAL_MALLOC(bufSize);
        if (!pBuffer)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        IppStatus status = CV_INSTRUMENT_FUN_IPP(ippiTrueDistanceTransform_8u32f_C1R, src_data, (int)src_step,
                                                 (Ipp32f *)dst_data, (int)dst_step, roi, pBuffer);
        ippFree(pBuffer);
        if (status < 0)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        // https://github.com/opencv/opencv/issues/24082
        // There is probably a rounding issue that leads to non-deterministic behavior
        // between runs on positions closer to zeros by x-axis in straight direction.
        // As a workaround, we detect the distances that expected to be exact
        // number of pixels and round manually.
        static const float correctionDiff = 1.0f / (1 << 11);
        for (int i = 0; i < height; ++i)
        {
            float* row = (float*)(dst_data + (size_t)i * dst_step);
            for (int j = 0; j < width; ++j)
            {
                float rounded = static_cast<float>(cvRound(row[j]));
                if (std::fabs(row[j] - rounded) <= correctionDiff)
                    row[j] = rounded;
            }
        }
        return CV_HAL_ERROR_OK;
    }

    // DIST_MASK_3 / DIST_MASK_5 chamfer distances.
    if (mask_size != cv::DIST_MASK_3 && mask_size != cv::DIST_MASK_5)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    // IPP uses 32-bit signed intermediates; bail out when the image is too large.
    bool has_int_overflow = (int64)width * height >= INT_MAX;
    if (has_int_overflow)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    float metrics[5] = {0};
    if (!ippGetDistanceTransformMask(dist_type, mask_size, metrics))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (mask_size == cv::DIST_MASK_3)
    {
        if (CV_INSTRUMENT_FUN_IPP(ippiDistanceTransform_3x3_8u32f_C1R, src_data, (int)src_step,
                                  (Ipp32f *)dst_data, (int)dst_step, roi, metrics) >= 0)
        {
            return CV_HAL_ERROR_OK;
        }
    }
    else // DIST_MASK_5
    {
        if (CV_INSTRUMENT_FUN_IPP(ippiDistanceTransform_5x5_8u32f_C1R, src_data, (int)src_step,
                                  (Ipp32f *)dst_data, (int)dst_step, roi, metrics) >= 0)
        {
            return CV_HAL_ERROR_OK;
        }
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

#endif // IPP_VERSION_X100 >= 700
