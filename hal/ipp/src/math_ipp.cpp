// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "ipp_hal_core.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/core/base.hpp>

int ipp_hal_invSqrt32f(const float* src, float* dst, int len)
{
    CV_HAL_CHECK_USE_IPP();
    IppStatus status = CV_INSTRUMENT_FUN_IPP(ippsInvSqrt_32f_A21, src, dst, len);
    if (status >= 0)
    {
        return CV_HAL_ERROR_OK;
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int ipp_hal_invSqrt64f(const double* src, double* dst, int len)
{
    CV_HAL_CHECK_USE_IPP();
    IppStatus status = CV_INSTRUMENT_FUN_IPP(ippsInvSqrt_64f_A50, src, dst, len);
    if (status >= 0)
    {
        return CV_HAL_ERROR_OK;
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int ipp_hal_exp32f(const float* src, float* dst, int len)
{
    CV_HAL_CHECK_USE_IPP();
    IppStatus status = CV_INSTRUMENT_FUN_IPP(ippsExp_32f_A21, src, dst, len);
    if (status >= 0)
    {
        return CV_HAL_ERROR_OK;
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int ipp_hal_exp64f(const double* src, double* dst, int len)
{
    CV_HAL_CHECK_USE_IPP();
    IppStatus status = CV_INSTRUMENT_FUN_IPP(ippsExp_64f_A50, src, dst, len);
    if (status >= 0)
    {
        return CV_HAL_ERROR_OK;
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int ipp_hal_log32f(const float* src, float* dst, int len)
{
    CV_HAL_CHECK_USE_IPP();
    IppStatus status = CV_INSTRUMENT_FUN_IPP(ippsLn_32f_A21, src, dst, len);
    if (status >= 0)
    {
        return CV_HAL_ERROR_OK;
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int ipp_hal_log64f(const double* src, double* dst, int len)
{
    CV_HAL_CHECK_USE_IPP();
    IppStatus status = CV_INSTRUMENT_FUN_IPP(ippsLn_64f_A50, src, dst, len);
    if (status >= 0)
    {
        return CV_HAL_ERROR_OK;
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
