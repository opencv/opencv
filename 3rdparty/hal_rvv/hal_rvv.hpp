// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_HAL_RVV_HPP_INCLUDED
#define OPENCV_HAL_RVV_HPP_INCLUDED

#include "opencv2/core/base.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/imgproc/hal/interface.h"

#ifndef CV_HAL_RVV_071_ENABLED
#  if defined(__GNUC__) && __GNUC__ == 10 && __GNUC_MINOR__ == 4 && defined(__THEAD_VERSION__) && defined(__riscv_v) && __riscv_v == 7000
#    define CV_HAL_RVV_071_ENABLED 1
#  else
#    define CV_HAL_RVV_071_ENABLED 0
#  endif
#endif

#if CV_HAL_RVV_071_ENABLED
#include "version/hal_rvv_071.hpp"
#endif

#if defined(__riscv_v) && __riscv_v == 1000000
#include "hal_rvv_1p0/types.hpp"
#include "hal_rvv_1p0/merge.hpp" // core
#include "hal_rvv_1p0/mean.hpp" // core
#include "hal_rvv_1p0/dxt.hpp" // core
#include "hal_rvv_1p0/norm.hpp" // core
#include "hal_rvv_1p0/norm_diff.hpp" // core
#include "hal_rvv_1p0/norm_hamming.hpp" // core
#include "hal_rvv_1p0/convert_scale.hpp" // core
#include "hal_rvv_1p0/minmax.hpp" // core
#include "hal_rvv_1p0/atan.hpp" // core
#include "hal_rvv_1p0/split.hpp" // core
#include "hal_rvv_1p0/magnitude.hpp" // core
#include "hal_rvv_1p0/cart_to_polar.hpp" // core
#include "hal_rvv_1p0/polar_to_cart.hpp" // core
#include "hal_rvv_1p0/flip.hpp" // core
#include "hal_rvv_1p0/lut.hpp" // core
#include "hal_rvv_1p0/exp.hpp" // core
#include "hal_rvv_1p0/log.hpp" // core
#include "hal_rvv_1p0/lu.hpp" // core
#include "hal_rvv_1p0/cholesky.hpp" // core
#include "hal_rvv_1p0/qr.hpp" // core
#include "hal_rvv_1p0/svd.hpp" // core
#include "hal_rvv_1p0/sqrt.hpp" // core

#include "hal_rvv_1p0/moments.hpp" // imgproc
#include "hal_rvv_1p0/filter.hpp" // imgproc
#include "hal_rvv_1p0/pyramids.hpp" // imgproc
#include "hal_rvv_1p0/color.hpp" // imgproc
#include "hal_rvv_1p0/warp.hpp" // imgproc
#include "hal_rvv_1p0/thresh.hpp" // imgproc
#include "hal_rvv_1p0/histogram.hpp" // imgproc
#include "hal_rvv_1p0/resize.hpp" // imgproc
#endif

#endif
