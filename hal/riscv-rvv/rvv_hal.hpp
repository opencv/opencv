// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_HAL_RVV_HPP_INCLUDED
#define OPENCV_HAL_RVV_HPP_INCLUDED

#include "opencv2/core/base.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/hal/interface.h"

#if defined(__riscv_v) && __riscv_v == 1000000
#define CV_HAL_RVV_1P0_ENABLED 1
#else
#define CV_HAL_RVV_1P0_ENABLED 0
#endif

#if defined(__riscv_v) && __riscv_v == 7000 && defined(__GNUC__) && __GNUC__ == 10 && __GNUC_MINOR__ == 4 && defined(__THEAD_VERSION__)
#define CV_HAL_RVV_071_ENABLED 1
#else
#define CV_HAL_RVV_071_ENABLED 0
#endif

#if CV_HAL_RVV_1P0_ENABLED || CV_HAL_RVV_071_ENABLED
#include <riscv_vector.h>
#endif
#include "include/types.hpp"
#include "include/core.hpp"
#include "include/imgproc.hpp"
#include "include/features2d.hpp"

#endif // OPENCV_HAL_RVV_HPP_INCLUDED
