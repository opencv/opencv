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

template<typename T>
static void softmaxHalfT(Mat &dst, const Mat &src, int axis, float scale, bool logSoftMax) {
    CV_Assert(src.isContinuous() && dst.isContinuous());
    CV_CheckTypeEQ(src.type(), dst.type(), "softmaxHalf: dst must match src type");

    const MatShape s = src.shape();
    CV_CheckGE(axis, 0, "softmaxHalf: axis out of range");
    CV_CheckLT(axis, s.dims, "softmaxHalf: axis out of range");

    const size_t outer = (size_t)total(s, 0, axis);
    const size_t n     = (size_t)s[axis];
    const size_t inner = (size_t)total(s, axis + 1);

    const T* sp = src.ptr<T>();
    T* dp = dst.ptr<T>();

    parallel_for_(Range(0, (int)(outer * inner)), [&](const Range& r) {
        for (int i = r.start; i < r.end; i++) {
            const size_t o = (size_t)i / inner, in = (size_t)i % inner;
            const size_t base = o * n * inner + in;

            float mx = -FLT_MAX;
            for (size_t j = 0; j < n; j++)
                mx = std::max(mx, (float)sp[base + j * inner] * scale);

            float sum = 0.f;
            for (size_t j = 0; j < n; j++) {
                const float e = std::exp((float)sp[base + j * inner] * scale - mx);
                dp[base + j * inner] = T(e);
                sum += e;
            }

            if (logSoftMax) {
                const float lsum = std::log(sum);
                for (size_t j = 0; j < n; j++)
                    dp[base + j * inner] = T((float)sp[base + j * inner] * scale - mx - lsum);
            } else {
                const float inv = 1.f / sum;
                for (size_t j = 0; j < n; j++)
                    dp[base + j * inner] = T((float)dp[base + j * inner] * inv);
            }
        }
    }, (double)(outer * inner * n) * (1 / 1024.0));
}

void softmaxHalf(Mat &dst, const Mat &src, int axis, float scale, bool logSoftMax) {
    int type = src.type();
    CV_CheckType(type, type == CV_16F || type == CV_16BF, "softmaxHalf: unsupported type");

    if (type == CV_16F)
        softmaxHalfT<hfloat>(dst, src, axis, scale, logSoftMax);
    else
        softmaxHalfT<bfloat>(dst, src, axis, scale, logSoftMax);
}

}} // cv::dnn
