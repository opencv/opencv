// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "ipp_hal_imgproc.hpp"

#include <opencv2/core.hpp>

#if IPP_VERSION_X100 >= 700

int ipp_hal_integral(int depth, int sdepth, int sqdepth,
                     const uchar * src_data, size_t src_step,
                     uchar * sum_data, size_t sum_step,
                     uchar * sqsum_data, size_t sqsum_step,
                     uchar * tilted_data, size_t tilted_step,
                     int width, int height, int cn)
{
    CV_HAL_CHECK_USE_IPP();

    if (cn > 1 || tilted_data)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    IppiSize size = {width, height};
    IppStatus status = ippStsErr;

    if (!sqsum_data)
    {
        if (depth == CV_8U && sdepth == CV_32S)
            status = CV_INSTRUMENT_FUN_IPP(ippiIntegral_8u32s_C1R, (const Ipp8u*)src_data, (int)src_step, (Ipp32s*)sum_data, (int)sum_step, size, 0);
        else if (depth == CV_8U && sdepth == CV_32F)
            status = CV_INSTRUMENT_FUN_IPP(ippiIntegral_8u32f_C1R, (const Ipp8u*)src_data, (int)src_step, (Ipp32f*)sum_data, (int)sum_step, size, 0);
        else if (depth == CV_32F && sdepth == CV_32F)
            status = CV_INSTRUMENT_FUN_IPP(ippiIntegral_32f_C1R, (const Ipp32f*)src_data, (int)src_step, (Ipp32f*)sum_data, (int)sum_step, size);
        else
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }
    else
    {
        if (depth == CV_8U && sdepth == CV_32S && sqdepth == CV_32S)
            status = CV_INSTRUMENT_FUN_IPP(ippiSqrIntegral_8u32s_C1R, (const Ipp8u*)src_data, (int)src_step, (Ipp32s*)sum_data, (int)sum_step, (Ipp32s*)sqsum_data, (int)sqsum_step, size, 0, 0);
        else if (depth == CV_8U && sdepth == CV_32S && sqdepth == CV_64F)
            status = CV_INSTRUMENT_FUN_IPP(ippiSqrIntegral_8u32s64f_C1R, (const Ipp8u*)src_data, (int)src_step, (Ipp32s*)sum_data, (int)sum_step, (Ipp64f*)sqsum_data, (int)sqsum_step, size, 0, 0);
        else if (depth == CV_8U && sdepth == CV_32F && sqdepth == CV_64F)
            status = CV_INSTRUMENT_FUN_IPP(ippiSqrIntegral_8u32f64f_C1R, (const Ipp8u*)src_data, (int)src_step, (Ipp32f*)sum_data, (int)sum_step, (Ipp64f*)sqsum_data, (int)sqsum_step, size, 0, 0);
        else
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    return status >= 0 ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
}

#endif
