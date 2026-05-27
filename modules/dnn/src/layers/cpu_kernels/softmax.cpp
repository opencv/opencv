// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// This file is modified from the ficus (https://github.com/vpisarev/ficus/blob/master/lib/NN/OpNN.fx).
// Here is the original license:
/*
    This file is a part of ficus language project.
    See ficus/LICENSE for the licensing terms
*/

#include "../../precomp.hpp"
#include "softmax.hpp"

#define CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
#include "activation_kernels.simd.hpp"
#include "layers/cpu_kernels/activation_kernels.simd_declarations.hpp"
#undef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace cv { namespace dnn {

void softmax(Mat &dst, const Mat &src, int axis, int axisBias, int axisStep, float scale) {
    CV_CPU_DISPATCH(softmax_, (dst, src, axis, axisBias, axisStep, scale),
                    CV_CPU_DISPATCH_MODES_ALL);
}

void softmax(Mat &dst, const Mat &src, int axis) {
    softmax(dst, src, axis, 0, src.size[axis], 1.f);
}

void softmax(Mat &dst, const Mat &src, int axis, float scale) {
    softmax(dst, src, axis, 0, src.size[axis], scale);
}

void logSoftmax(Mat &dst, const Mat &src, int axis) {
    softmax(dst, src, axis);
    log(dst, dst);
}

}} // cv::dnn
