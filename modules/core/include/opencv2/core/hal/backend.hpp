// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_HAL_BACKEND_HPP
#define OPENCV_CORE_HAL_BACKEND_HPP

#include "opencv2/core/cvdef.h"
#include "opencv2/core/mat.hpp"

namespace cv { namespace hal {

// Abstract GPU backend: one typed method per op; default false => CPU fallback.
class CV_EXPORTS Backend
{
public:
    virtual ~Backend() {}

    // resize: dsize, inv_scale_x, inv_scale_y, interpolation
    virtual bool resize(InputArray, OutputArray, Size, double, double, int) { return false; }

    // Gaussian blur: ksize, sigma1, sigma2
    virtual bool gaussianBlur(InputArray, OutputArray, Size, double, double) { return false; }

    // color conversion: code, dst channel count (dcn)
    virtual bool cvtColor(InputArray, OutputArray, int, int) { return false; }

    // threshold: thresh, maxval, type
    virtual bool threshold(InputArray, OutputArray, double, double, int) { return false; }

    // subtract: src - src2
    virtual bool subtract(InputArray, InputArray, OutputArray) { return false; }

    // multiply: src * src2 (element-wise)
    virtual bool multiply(InputArray, InputArray, OutputArray) { return false; }

    // divide: src / src2
    virtual bool divide(InputArray, InputArray, OutputArray) { return false; }

    // convertTo: element type conversion with scale/shift — rtype, alpha, beta
    virtual bool convertTo(InputArray, OutputArray, int, double, double) { return false; }

    // split: de-interleave a multi-channel src into single-channel planes
    virtual bool split(InputArray, OutputArrayOfArrays) { return false; }

    // device-aware MatAllocator (keeps UMat data resident), or nullptr for default
    virtual MatAllocator* allocator() const { return NULL; }
};

} // namespace hal
} // namespace cv

#endif // OPENCV_CORE_HAL_BACKEND_HPP
