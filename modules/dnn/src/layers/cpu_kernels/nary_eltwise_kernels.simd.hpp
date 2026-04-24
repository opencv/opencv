// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include <opencv2/core.hpp>
#include "opencv2/core/hal/intrin.hpp"

namespace cv { namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

// Op codes that simd_binop_f32_ understands.
enum SimdBinOp { SIMD_BIN_ADD = 0, SIMD_BIN_SUB = 1, SIMD_BIN_MUL = 2, SIMD_BIN_DIV = 3 };

// Apply binary op on n contiguous floats.
int simd_binop_f32_(const float* a, const float* b, float* out, int n, int op);

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace cv { namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

int simd_binop_f32_(const float* a, const float* b, float* out, int n, int op) {
    int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int lanes = VTraits<v_float32>::nlanes;
    if (op == SIMD_BIN_ADD) {
        for (; i <= n - lanes * 4; i += lanes * 4) {
            v_store(out + i,             v_add(vx_load(a + i),             vx_load(b + i)));
            v_store(out + i + lanes,     v_add(vx_load(a + i + lanes),     vx_load(b + i + lanes)));
            v_store(out + i + lanes * 2, v_add(vx_load(a + i + lanes * 2), vx_load(b + i + lanes * 2)));
            v_store(out + i + lanes * 3, v_add(vx_load(a + i + lanes * 3), vx_load(b + i + lanes * 3)));
        }
        for (; i <= n - lanes; i += lanes)
            v_store(out + i, v_add(vx_load(a + i), vx_load(b + i)));
    } else if (op == SIMD_BIN_MUL) {
        for (; i <= n - lanes * 4; i += lanes * 4) {
            v_store(out + i,             v_mul(vx_load(a + i),             vx_load(b + i)));
            v_store(out + i + lanes,     v_mul(vx_load(a + i + lanes),     vx_load(b + i + lanes)));
            v_store(out + i + lanes * 2, v_mul(vx_load(a + i + lanes * 2), vx_load(b + i + lanes * 2)));
            v_store(out + i + lanes * 3, v_mul(vx_load(a + i + lanes * 3), vx_load(b + i + lanes * 3)));
        }
        for (; i <= n - lanes; i += lanes)
            v_store(out + i, v_mul(vx_load(a + i), vx_load(b + i)));
    } else if (op == SIMD_BIN_SUB) {
        for (; i <= n - lanes * 4; i += lanes * 4) {
            v_store(out + i,             v_sub(vx_load(a + i),             vx_load(b + i)));
            v_store(out + i + lanes,     v_sub(vx_load(a + i + lanes),     vx_load(b + i + lanes)));
            v_store(out + i + lanes * 2, v_sub(vx_load(a + i + lanes * 2), vx_load(b + i + lanes * 2)));
            v_store(out + i + lanes * 3, v_sub(vx_load(a + i + lanes * 3), vx_load(b + i + lanes * 3)));
        }
        for (; i <= n - lanes; i += lanes)
            v_store(out + i, v_sub(vx_load(a + i), vx_load(b + i)));
    } else if (op == SIMD_BIN_DIV) {
        for (; i <= n - lanes * 4; i += lanes * 4) {
            v_store(out + i,             v_div(vx_load(a + i),             vx_load(b + i)));
            v_store(out + i + lanes,     v_div(vx_load(a + i + lanes),     vx_load(b + i + lanes)));
            v_store(out + i + lanes * 2, v_div(vx_load(a + i + lanes * 2), vx_load(b + i + lanes * 2)));
            v_store(out + i + lanes * 3, v_div(vx_load(a + i + lanes * 3), vx_load(b + i + lanes * 3)));
        }
        for (; i <= n - lanes; i += lanes)
            v_store(out + i, v_div(vx_load(a + i), vx_load(b + i)));
    }
#else
    (void)a; (void)b; (void)out; (void)n; (void)op;
#endif
    return i;
}

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}  // cv::dnn

#endif  // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
