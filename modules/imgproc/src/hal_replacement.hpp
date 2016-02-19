#ifndef OPENCV_IMGPROC_HAL_REPLACEMENT_HPP
#define OPENCV_IMGPROC_HAL_REPLACEMENT_HPP

#include "opencv2/core/hal/interface.h"

struct cvhalFilter2D {};

inline int hal_ni_filterInit(cvhalFilter2D **, uchar *, size_t, int, int, int, int, int, int, int, int, double, int, int, bool, bool) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_filter(cvhalFilter2D *, uchar *, size_t, uchar *, size_t, int, int, int, int, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_filterFree(cvhalFilter2D *) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

#define cv_hal_filterInit hal_ni_filterInit
#define cv_hal_filter hal_ni_filter
#define cv_hal_filterFree hal_ni_filterFree

inline int hal_ni_sepFilterInit(cvhalFilter2D **, int, int, int, uchar *, size_t, int, int, uchar *, size_t, int, int, int, int, double, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_sepFilter(cvhalFilter2D *, uchar *, size_t, uchar*, size_t, int, int, int, int, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_sepFilterFree(cvhalFilter2D *) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

#define cv_hal_sepFilterInit hal_ni_sepFilterInit
#define cv_hal_sepFilter hal_ni_sepFilter
#define cv_hal_sepFilterFree hal_ni_sepFilterFree

inline int hal_ni_morphInit(cvhalFilter2D **, int, int, int, int, int, int, uchar *, size_t, int, int, int, int, int, const double[4], int, bool, bool) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_morph(cvhalFilter2D *, uchar *, size_t, uchar *, size_t, int, int, int, int, int, int, int, int, int, int) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }
inline int hal_ni_morphFree(cvhalFilter2D *) { return CV_HAL_ERROR_NOT_IMPLEMENTED; }

#define cv_hal_morphInit hal_ni_morphInit
#define cv_hal_morph hal_ni_morph
#define cv_hal_morphFree hal_ni_morphFree

#include "custom_hal.hpp"

#endif // OPENCV_IMGPROC_HAL_REPLACEMENT_HPP
