// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "ipp_hal_imgproc.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>

#if IPP_VERSION_X100 >= 810

#if IPP_VERSION_X100 >= 201700
#define IPP_HAL_MALLOC(SIZE) ippMalloc_L(SIZE)
#else
#define IPP_HAL_MALLOC(SIZE) ippMalloc((int)(SIZE))
#endif

int ipp_hal_cornerHarris(const uchar* src_data, size_t src_step, int src_type,
                         uchar* dst_data, size_t dst_step, int width, int height,
                         int block_size, int ksize, double k, int border_type, bool is_submatrix)
{
    CV_HAL_CHECK_USE_IPP();

    int depth = CV_MAT_DEPTH(src_type), cn = CV_MAT_CN(src_type);
    int borderTypeNI = border_type & ~cv::BORDER_ISOLATED;
    bool isolated = (border_type & cv::BORDER_ISOLATED) != 0;

    if (!((ksize == 3 || ksize == 5) && (src_type == CV_8UC1 || src_type == CV_32FC1) &&
          (borderTypeNI == cv::BORDER_CONSTANT || borderTypeNI == cv::BORDER_REPLICATE) &&
          cn == 1 && (!is_submatrix || isolated)))
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    IppiSize roisize = { width, height };
    IppiMaskSize masksize = ksize == 5 ? ippMskSize5x5 : ippMskSize3x3;
    IppDataType datatype = src_type == CV_8UC1 ? ipp8u : ipp32f;
    Ipp32s bufsize = 0;

    double scale = (double)(1 << ((ksize > 0 ? ksize : 3) - 1)) * block_size;
    if (ksize < 0)
        scale *= 2.0;
    if (depth == CV_8U)
        scale *= 255.0;
    scale = std::pow(scale, -4);

    if (ippiHarrisCornerGetBufferSize(roisize, masksize, block_size, datatype, cn, &bufsize) < 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    Ipp8u* buffer = (Ipp8u*)IPP_HAL_MALLOC(bufsize);
    if (!buffer)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    IppiDifferentialKernel filterType = ksize > 0 ? ippFilterSobel : ippFilterScharr;
    IppiBorderType borderTypeIpp = borderTypeNI == cv::BORDER_CONSTANT ? ippBorderConst : ippBorderRepl;
    IppStatus status = (IppStatus)-1;

    if (depth == CV_8U)
        status = CV_INSTRUMENT_FUN_IPP(ippiHarrisCorner_8u32f_C1R, (const Ipp8u*)src_data, (int)src_step,
                    (Ipp32f*)dst_data, (int)dst_step, roisize, filterType, masksize, block_size,
                    (Ipp32f)k, (Ipp32f)scale, borderTypeIpp, 0, buffer);
    else if (depth == CV_32F)
        status = CV_INSTRUMENT_FUN_IPP(ippiHarrisCorner_32f_C1R, (const Ipp32f*)src_data, (int)src_step,
                    (Ipp32f*)dst_data, (int)dst_step, roisize, filterType, masksize, block_size,
                    (Ipp32f)k, (Ipp32f)scale, borderTypeIpp, 0, buffer);
    ippFree(buffer);

    return status >= 0 ? CV_HAL_ERROR_OK : CV_HAL_ERROR_NOT_IMPLEMENTED;
}

#endif // IPP_VERSION_X100 >= 810
