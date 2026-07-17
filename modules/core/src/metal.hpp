// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_SRC_METAL_HPP
#define OPENCV_CORE_SRC_METAL_HPP

#include "opencv2/core/mat.hpp"

namespace cv {
namespace metal {

bool haveMetal();
MatAllocator* getMetalAllocator();
bool copyToMask(const UMat& src, const UMat& mask, UMat& dst, bool haveDstUninit);
bool add(const UMat& src1, const UMat& src2, UMat& dst);
bool subtract(const UMat& src1, const UMat& src2, UMat& dst);
bool multiply(const UMat& src1, const UMat& src2, UMat& dst, double scale);
bool bitwise(const UMat& src1, const UMat& src2, UMat& dst, int op);
bool compare(const UMat& src1, const UMat& src2, UMat& dst, int op);
bool convertTo(const UMat& src, UMat& dst, int ddepth, double alpha, double beta);
bool threshold(const UMat& src, UMat& dst, double thresh, double maxval, int thresholdType);
bool setTo(UMat& dst, const Mat& value, const UMat* mask);

} // namespace metal
} // namespace cv

#endif // OPENCV_CORE_SRC_METAL_HPP
