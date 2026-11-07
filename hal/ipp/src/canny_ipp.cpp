// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "ipp_hal_imgproc.hpp"

#ifdef HAVE_IPP_IW

#include <opencv2/core.hpp>
#include "precomp_ipp.hpp"

int ipp_hal_canny(const uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
                  int width, int height, int cn,
                  double lowThreshold, double highThreshold, int ksize, bool L2gradient)
{
    CV_HAL_CHECK_USE_IPP();

#if IPP_DISABLE_PERF_CANNY_MT
    if(cv::getNumThreads() > 1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
#endif

    if(width <= 3 || height <= 3)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if(cn != 1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    IppiMaskSize kernel;
    if(ksize == 3)
        kernel = ippMskSize3x3;
    else if(ksize == 5)
        kernel = ippMskSize5x5;
    else
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    IppNormType norm = L2gradient ? ippNormL2 : ippNormL1;

    try
    {
        ::ipp::IwiImage iwSrc, iwDst;
        iwSrc.Init(IwiSize{width, height}, ipp8u, cn, ::ipp::IwiBorderSize(), (void*)src_data, IwSize(src_step));
        iwDst.Init(IwiSize{width, height}, ipp8u, cn, ::ipp::IwiBorderSize(), dst_data, IwSize(dst_step));

        CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterCanny, iwSrc, iwDst, (float)lowThreshold, (float)highThreshold,
                              ::ipp::IwiFilterCannyParams(ippFilterSobel, kernel, norm), ippBorderRepl);
    }
    catch (const ::ipp::IwException &)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    return CV_HAL_ERROR_OK;
}

int ipp_hal_canny_deriv(const short* dx_data, size_t dx_step, const short* dy_data, size_t dy_step,
                        uchar* dst_data, size_t dst_step, int width, int height, int cn,
                        double lowThreshold, double highThreshold, bool L2gradient)
{
    CV_HAL_CHECK_USE_IPP();

#if IPP_DISABLE_PERF_CANNY_MT
    if(cv::getNumThreads() > 1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
#endif

    if(width <= 3 || height <= 3)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if(cn != 1)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    IppNormType norm = L2gradient ? ippNormL2 : ippNormL1;

    try
    {
        ::ipp::IwiImage iwSrcDx, iwSrcDy, iwDst;
        iwSrcDx.Init(IwiSize{width, height}, ipp16s, cn, ::ipp::IwiBorderSize(), (void*)dx_data, IwSize(dx_step));
        iwSrcDy.Init(IwiSize{width, height}, ipp16s, cn, ::ipp::IwiBorderSize(), (void*)dy_data, IwSize(dy_step));
        iwDst.Init(IwiSize{width, height}, ipp8u, cn, ::ipp::IwiBorderSize(), dst_data, IwSize(dst_step));

        CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterCannyDeriv, iwSrcDx, iwSrcDy, iwDst, (float)lowThreshold, (float)highThreshold,
                              ::ipp::IwiFilterCannyDerivParams(norm));
    }
    catch (const ::ipp::IwException &)
    {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    return CV_HAL_ERROR_OK;
}

#endif
