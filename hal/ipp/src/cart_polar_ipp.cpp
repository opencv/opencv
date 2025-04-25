#include "ipp_hal_core.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/core/base.hpp>

int ipp_hal_polarToCart32f(const float* mag, const float* angle, float* x, float* y, int len, bool angleInDegrees)
{
    const bool isInPlace = (x == mag) || (x == angle) || (y == mag) || (y == angle);
    if (isInPlace || angleInDegrees)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (CV_INSTRUMENT_FUN_IPP(ippsPolarToCart_32f, mag, angle, x, y, len) < 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    return CV_HAL_ERROR_OK;
}

int ipp_hal_polarToCart64f(const double* mag, const double* angle, double* x, double* y, int len, bool angleInDegrees)
{
    const bool isInPlace = (x == mag) || (x == angle) || (y == mag) || (y == angle);
    if (isInPlace || angleInDegrees)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    if (CV_INSTRUMENT_FUN_IPP(ippsPolarToCart_64f, mag, angle, x, y, len) < 0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    return CV_HAL_ERROR_OK;
}
