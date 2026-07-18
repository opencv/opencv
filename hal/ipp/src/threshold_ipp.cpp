// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "ipp_hal_imgproc.hpp"

#include <opencv2/core.hpp>
#include "precomp_ipp.hpp"

#include <cmath>
#include <limits>

#if IPP_VERSION_X100 >= 700

int ipp_hal_threshold(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                      int width, int height, int depth, int cn, double thresh, double maxValue,
                      int thresholdType)
{
    CV_HAL_CHECK_USE_IPP();
    CV_UNUSED(maxValue);

    // IPP only implements these three types and depths; everything else falls back to OpenCV.
    if (thresholdType != cv::THRESH_TRUNC && thresholdType != cv::THRESH_TOZERO &&
        thresholdType != cv::THRESH_TOZERO_INV)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (depth != CV_8U && depth != CV_16S && depth != CV_32F)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    int roi_width = width * cn;   // threshold is elementwise, so channels fold into the width
    int roi_height = height;
    int elemSize1 = (depth == CV_8U) ? 1 : (depth == CV_16S) ? 2 : 4;

    if (src_step == (size_t)roi_width * elemSize1 && dst_step == (size_t)roi_width * elemSize1)
    {
        roi_width *= roi_height;
        roi_height = 1;
        src_step = dst_step = (size_t)roi_width * elemSize1;
    }

    IppiSize sz = { roi_width, roi_height };
    const bool inplace = (src_data == dst_data);
    IppStatus status = ippStsErr;

    CV_SUPPRESS_DEPRECATED_START
    switch (depth)
    {
    case CV_8U:
    {
        Ipp8u t = (Ipp8u)thresh;
        switch (thresholdType)
        {
        case cv::THRESH_TRUNC:
            if (inplace)
                status = CV_INSTRUMENT_FUN_IPP(ippiThreshold_GT_8u_C1IR, dst_data, (int)dst_step, sz, t);
            if (!inplace || status < 0)
                status = CV_INSTRUMENT_FUN_IPP(ippiThreshold_GT_8u_C1R, src_data, (int)src_step, dst_data, (int)dst_step, sz, t);
            break;
        case cv::THRESH_TOZERO:
            if (inplace)
                status = CV_INSTRUMENT_FUN_IPP(ippiThreshold_LTVal_8u_C1IR, dst_data, (int)dst_step, sz, t + 1, 0);
            if (!inplace || status < 0)
                status = CV_INSTRUMENT_FUN_IPP(ippiThreshold_LTVal_8u_C1R, src_data, (int)src_step, dst_data, (int)dst_step, sz, t + 1, 0);
            break;
        case cv::THRESH_TOZERO_INV:
            if (inplace)
                status = CV_INSTRUMENT_FUN_IPP(ippiThreshold_GTVal_8u_C1IR, dst_data, (int)dst_step, sz, t, 0);
            if (!inplace || status < 0)
                status = CV_INSTRUMENT_FUN_IPP(ippiThreshold_GTVal_8u_C1R, src_data, (int)src_step, dst_data, (int)dst_step, sz, t, 0);
            break;
        }
        break;
    }
    case CV_16S:
    {
        const Ipp16s* src = (const Ipp16s*)src_data;
        Ipp16s* dst = (Ipp16s*)dst_data;
        Ipp16s t = (Ipp16s)thresh;
        switch (thresholdType)
        {
        case cv::THRESH_TRUNC:
            if (inplace)
                status = CV_INSTRUMENT_FUN_IPP(ippiThreshold_GT_16s_C1IR, dst, (int)dst_step, sz, t);
            if (!inplace || status < 0)
                status = CV_INSTRUMENT_FUN_IPP(ippiThreshold_GT_16s_C1R, src, (int)src_step, dst, (int)dst_step, sz, t);
            break;
        case cv::THRESH_TOZERO:
            if (inplace)
                status = CV_INSTRUMENT_FUN_IPP(ippiThreshold_LTVal_16s_C1IR, dst, (int)dst_step, sz, t + 1, 0);
            if (!inplace || status < 0)
                status = CV_INSTRUMENT_FUN_IPP(ippiThreshold_LTVal_16s_C1R, src, (int)src_step, dst, (int)dst_step, sz, t + 1, 0);
            break;
        case cv::THRESH_TOZERO_INV:
            if (inplace)
                status = CV_INSTRUMENT_FUN_IPP(ippiThreshold_GTVal_16s_C1IR, dst, (int)dst_step, sz, t, 0);
            if (!inplace || status < 0)
                status = CV_INSTRUMENT_FUN_IPP(ippiThreshold_GTVal_16s_C1R, src, (int)src_step, dst, (int)dst_step, sz, t, 0);
            break;
        }
        break;
    }
    case CV_32F:
    {
        const Ipp32f* src = (const Ipp32f*)src_data;
        Ipp32f* dst = (Ipp32f*)dst_data;
        Ipp32f t = (Ipp32f)thresh;
        switch (thresholdType)
        {
        case cv::THRESH_TRUNC:
            status = CV_INSTRUMENT_FUN_IPP(ippiThreshold_GT_32f_C1R, src, (int)src_step, dst, (int)dst_step, sz, t);
            break;
        case cv::THRESH_TOZERO:
            status = CV_INSTRUMENT_FUN_IPP(ippiThreshold_LTVal_32f_C1R, src, (int)src_step, dst, (int)dst_step, sz, nextafterf(t, std::numeric_limits<float>::infinity()), 0);
            break;
        case cv::THRESH_TOZERO_INV:
            status = CV_INSTRUMENT_FUN_IPP(ippiThreshold_GTVal_32f_C1R, src, (int)src_step, dst, (int)dst_step, sz, t, 0);
            break;
        }
        break;
    }
    }
    CV_SUPPRESS_DEPRECATED_END

    return status >= 0 ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
}

#endif // IPP_VERSION_X100 >= 700
