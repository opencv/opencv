// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "ipp_hal_core.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/core/base.hpp>

int ipp_hal_polarToCart32f(const float* mag, const float* angle, float* x, float* y, int len, bool angleInDegrees)
{
    printf("ipp_hal_polarToCart32f call\n");
    const bool isInPlace = (x == mag) || (x == angle) || (y == mag) || (y == angle);
    if (isInPlace || angleInDegrees)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (CV_INSTRUMENT_FUN_IPP(ippsPolarToCart_32f, mag, angle, x, y, len) < 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    return CV_HAL_ERROR_OK;
}

int ipp_hal_polarToCart64f(const double* mag, const double* angle, double* x, double* y, int len, bool angleInDegrees)
{
    printf("ipp_hal_polarToCart64f call\n");
    const bool isInPlace = (x == mag) || (x == angle) || (y == mag) || (y == angle);
    if (isInPlace || angleInDegrees)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (CV_INSTRUMENT_FUN_IPP(ippsPolarToCart_64f, mag, angle, x, y, len) < 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    return CV_HAL_ERROR_OK;
}
