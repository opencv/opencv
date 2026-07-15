// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "ipp_hal_imgproc.hpp"

#include <opencv2/core.hpp>
#include "precomp_ipp.hpp"

#if IPP_VERSION_X100 >= 810 && !DISABLE_IPP_MEDIAN_BLUR

// Mirrors CV_IPP_MALLOC from modules/core/include/opencv2/core/private.hpp: ippicv (IPP >= 2017)
// exposes only the 64-bit ippMalloc_L.
#if IPP_VERSION_X100 >= 201700
#define IPP_HAL_MALLOC(SIZE) ippMalloc_L(SIZE)
#else
#define IPP_HAL_MALLOC(SIZE) ippMalloc((int)(SIZE))
#endif

int ipp_hal_medianBlur(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                       int width, int height, int depth, int cn, int ksize)
{
    CV_HAL_CHECK_USE_IPP();

#if IPP_VERSION_X100 < 201801
    // Degradations for big kernel
    if(ksize > 7)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
#endif

    IppDataType ippType = ippiGetDataType(depth);
    if (ippType == (IppDataType)-1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    IppiSize dstRoiSize = ippiSize(width, height);
    IppiSize maskSize   = ippiSize(ksize, ksize);

    // IPP filter cannot run in place; clone the source when input and output alias.
    cv::Mat srcClone;
    const uchar* src = src_data;
    int sstep = (int)src_step;
    if (src_data == dst_data)
    {
        cv::Mat(height, width, CV_MAKETYPE(depth, cn), (void*)src_data, src_step).copyTo(srcClone);
        src = srcClone.ptr();
        sstep = (int)srcClone.step;
    }

    int bufSize = 0;
    if (ippiFilterMedianBorderGetBufferSize(dstRoiSize, maskSize, ippType, cn, &bufSize) < 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    Ipp8u* buffer = (Ipp8u*)IPP_HAL_MALLOC(bufSize);
    if (!buffer)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    IppStatus status = ippStsErr;
    switch (ippType)
    {
    case ipp8u:
        if (cn == 1)
            status = CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_8u_C1R, (const Ipp8u*)src, sstep, (Ipp8u*)dst_data, (int)dst_step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer);
        else if (cn == 3)
            status = CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_8u_C3R, (const Ipp8u*)src, sstep, (Ipp8u*)dst_data, (int)dst_step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer);
        else if (cn == 4)
            status = CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_8u_C4R, (const Ipp8u*)src, sstep, (Ipp8u*)dst_data, (int)dst_step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer);
        break;
    case ipp16u:
        if (cn == 1)
            status = CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_16u_C1R, (const Ipp16u*)src, sstep, (Ipp16u*)dst_data, (int)dst_step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer);
        else if (cn == 3)
            status = CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_16u_C3R, (const Ipp16u*)src, sstep, (Ipp16u*)dst_data, (int)dst_step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer);
        else if (cn == 4)
            status = CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_16u_C4R, (const Ipp16u*)src, sstep, (Ipp16u*)dst_data, (int)dst_step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer);
        break;
    case ipp16s:
        if (cn == 1)
            status = CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_16s_C1R, (const Ipp16s*)src, sstep, (Ipp16s*)dst_data, (int)dst_step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer);
        else if (cn == 3)
            status = CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_16s_C3R, (const Ipp16s*)src, sstep, (Ipp16s*)dst_data, (int)dst_step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer);
        else if (cn == 4)
            status = CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_16s_C4R, (const Ipp16s*)src, sstep, (Ipp16s*)dst_data, (int)dst_step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer);
        break;
    case ipp32f:
        if (cn == 1)
            status = CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_32f_C1R, (const Ipp32f*)src, sstep, (Ipp32f*)dst_data, (int)dst_step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer);
        break;
    default:
        break;
    }

    ippFree(buffer);
    return status >= 0 ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
}

#endif // IPP_VERSION_X100 >= 810 && !DISABLE_IPP_MEDIAN_BLUR
