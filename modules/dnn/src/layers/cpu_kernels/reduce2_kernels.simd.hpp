// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include "opencv2/core/hal/intrin.hpp"
#include <cfloat>
#include <cmath>
#include <vector>
#include <algorithm>

namespace cv { namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

// Reduce-all over a contiguous CV_32F src
void reduceAllFloatParallel_(const Mat& src, Mat& dst, int reduce_type);

// Reduce a contiguous trailing block of axes for CV_32F src.
void reduceLastAxesFloatParallel_(const Mat& src, Mat& dst, size_t innerLen, int reduce_type);

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace cv { namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void reduceAllFloatParallel_(const Mat& src, Mat& dst, int reduce_type_int) {
    const Reduce2Layer::ReduceType rt = (Reduce2Layer::ReduceType)reduce_type_int;
    const float* p = src.ptr<const float>();
    const size_t total = src.total();
    const int nThreads = std::max(1, cv::getNumThreads());
    const int stripes = std::max(1, std::min<int>(
        (int)((total + 16383) / 16384), nThreads));

    float init_f = 0.0f;
    if (rt == Reduce2Layer::ReduceType::MAX)       init_f = -FLT_MAX;
    else if (rt == Reduce2Layer::ReduceType::MIN)  init_f = FLT_MAX;
    else if (rt == Reduce2Layer::ReduceType::PROD) init_f = 1.0f;

    std::vector<float>  partial_f(stripes, init_f);
    std::vector<double> partial_d(stripes, (rt == Reduce2Layer::ReduceType::PROD) ? 1.0 : 0.0);

    parallel_for_(Range(0, stripes), [&](const Range& r) {
        for (int s = r.start; s < r.end; s++) {
            size_t start = (size_t)s * total / stripes;
            size_t end   = (size_t)(s + 1) * total / stripes;
            const float* q = p + start;
            size_t n = end - start, i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            const int L = VTraits<v_float32>::vlanes();
#endif
            switch (rt) {
            case Reduce2Layer::ReduceType::MAX: {
                float acc = -FLT_MAX;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                v_float32 va = vx_setall_f32(-FLT_MAX);
                for (; i + L <= n; i += L) va = v_max(va, vx_load(q + i));
                acc = v_reduce_max(va);
#endif
                for (; i < n; i++) acc = acc > q[i] ? acc : q[i];
                partial_f[s] = acc;
                break;
            }
            case Reduce2Layer::ReduceType::MIN: {
                float acc = FLT_MAX;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                v_float32 va = vx_setall_f32(FLT_MAX);
                for (; i + L <= n; i += L) va = v_min(va, vx_load(q + i));
                acc = v_reduce_min(va);
#endif
                for (; i < n; i++) acc = acc < q[i] ? acc : q[i];
                partial_f[s] = acc;
                break;
            }
            case Reduce2Layer::ReduceType::SUM:
            case Reduce2Layer::ReduceType::MEAN:
            case Reduce2Layer::ReduceType::LOG_SUM: {
                double acc = 0.0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                v_float32 va = vx_setzero_f32();
                for (; i + L <= n; i += L) va = v_add(va, vx_load(q + i));
                acc += (double)v_reduce_sum(va);
#endif
                for (; i < n; i++) acc += q[i];
                partial_d[s] = acc;
                break;
            }
            case Reduce2Layer::ReduceType::L1: {
                double acc = 0.0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                v_float32 va = vx_setzero_f32();
                for (; i + L <= n; i += L) va = v_add(va, v_abs(vx_load(q + i)));
                acc += (double)v_reduce_sum(va);
#endif
                for (; i < n; i++) acc += std::fabs(q[i]);
                partial_d[s] = acc;
                break;
            }
            case Reduce2Layer::ReduceType::L2:
            case Reduce2Layer::ReduceType::SUM_SQUARE: {
                double acc = 0.0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                v_float32 va = vx_setzero_f32();
                for (; i + L <= n; i += L) {
                    v_float32 x = vx_load(q + i);
                    va = v_add(va, v_mul(x, x));
                }
                acc += (double)v_reduce_sum(va);
#endif
                for (; i < n; i++) { double x = q[i]; acc += x * x; }
                partial_d[s] = acc;
                break;
            }
            case Reduce2Layer::ReduceType::PROD: {
                double acc = 1.0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                v_float32 va = vx_setall_f32(1.0f);
                for (; i + L <= n; i += L) va = v_mul(va, vx_load(q + i));
                float buf[VTraits<v_float32>::max_nlanes];
                v_store(buf, va);
                for (int k = 0; k < L; k++) acc *= (double)buf[k];
#endif
                for (; i < n; i++) acc *= (double)q[i];
                partial_d[s] = acc;
                break;
            }
            default: CV_Error(Error::StsInternal, "reduceAllFloatParallel_: unhandled type");
            }
        }
    });

    float* out = dst.ptr<float>();
    switch (rt) {
    case Reduce2Layer::ReduceType::MAX: {
        float m = -FLT_MAX;
        for (int s = 0; s < stripes; s++) m = m > partial_f[s] ? m : partial_f[s];
        *out = m;
        break;
    }
    case Reduce2Layer::ReduceType::MIN: {
        float m = FLT_MAX;
        for (int s = 0; s < stripes; s++) m = m < partial_f[s] ? m : partial_f[s];
        *out = m;
        break;
    }
    case Reduce2Layer::ReduceType::SUM: {
        double acc = 0.0; for (int s = 0; s < stripes; s++) acc += partial_d[s];
        *out = (float)acc;
        break;
    }
    case Reduce2Layer::ReduceType::MEAN: {
        double acc = 0.0; for (int s = 0; s < stripes; s++) acc += partial_d[s];
        *out = total > 0 ? (float)(acc / (double)total) : 0.0f;
        break;
    }
    case Reduce2Layer::ReduceType::L1:
    case Reduce2Layer::ReduceType::SUM_SQUARE: {
        double acc = 0.0; for (int s = 0; s < stripes; s++) acc += partial_d[s];
        *out = (float)acc;
        break;
    }
    case Reduce2Layer::ReduceType::L2: {
        double acc = 0.0; for (int s = 0; s < stripes; s++) acc += partial_d[s];
        *out = (float)std::sqrt(acc);
        break;
    }
    case Reduce2Layer::ReduceType::LOG_SUM: {
        double acc = 0.0; for (int s = 0; s < stripes; s++) acc += partial_d[s];
        *out = total > 0 ? (float)std::log(acc) : -std::numeric_limits<float>::infinity();
        break;
    }
    case Reduce2Layer::ReduceType::PROD: {
        double acc = 1.0; for (int s = 0; s < stripes; s++) acc *= partial_d[s];
        *out = (float)acc;
        break;
    }
    default: CV_Error(Error::StsInternal, "reduceAllFloatParallel_: unhandled type");
    }
}

void reduceLastAxesFloatParallel_(const Mat& src, Mat& dst, size_t innerLen, int reduce_type_int) {
    const Reduce2Layer::ReduceType rt = (Reduce2Layer::ReduceType)reduce_type_int;
    const float* p = src.ptr<const float>();
    float* q = dst.ptr<float>();
    const size_t nOut = src.total() / innerLen;

    const double inv_inner = innerLen > 0 ? 1.0 / (double)innerLen : 0.0;

    parallel_for_(Range(0, (int)nOut), [&](const Range& r) {
        for (int row = r.start; row < r.end; row++) {
            const float* s0 = p + (size_t)row * innerLen;
            size_t i = 0, n = innerLen;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            const int L = VTraits<v_float32>::vlanes();
#endif
            switch (rt) {
            case Reduce2Layer::ReduceType::MAX: {
                float acc = -FLT_MAX;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                if (n >= (size_t)L) {
                    v_float32 va = vx_setall_f32(-FLT_MAX);
                    for (; i + L <= n; i += L) va = v_max(va, vx_load(s0 + i));
                    acc = v_reduce_max(va);
                }
#endif
                for (; i < n; i++) acc = acc > s0[i] ? acc : s0[i];
                q[row] = acc;
                break;
            }
            case Reduce2Layer::ReduceType::MIN: {
                float acc = FLT_MAX;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                if (n >= (size_t)L) {
                    v_float32 va = vx_setall_f32(FLT_MAX);
                    for (; i + L <= n; i += L) va = v_min(va, vx_load(s0 + i));
                    acc = v_reduce_min(va);
                }
#endif
                for (; i < n; i++) acc = acc < s0[i] ? acc : s0[i];
                q[row] = acc;
                break;
            }
            case Reduce2Layer::ReduceType::SUM:
            case Reduce2Layer::ReduceType::MEAN:
            case Reduce2Layer::ReduceType::LOG_SUM: {
                double acc = 0.0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                if (n >= (size_t)L) {
                    v_float32 va = vx_setzero_f32();
                    for (; i + L <= n; i += L) va = v_add(va, vx_load(s0 + i));
                    acc += (double)v_reduce_sum(va);
                }
#endif
                for (; i < n; i++) acc += s0[i];
                if (rt == Reduce2Layer::ReduceType::MEAN)        q[row] = (float)(acc * inv_inner);
                else if (rt == Reduce2Layer::ReduceType::LOG_SUM) q[row] = (float)std::log(acc);
                else                                              q[row] = (float)acc;
                break;
            }
            case Reduce2Layer::ReduceType::L1: {
                double acc = 0.0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                if (n >= (size_t)L) {
                    v_float32 va = vx_setzero_f32();
                    for (; i + L <= n; i += L) va = v_add(va, v_abs(vx_load(s0 + i)));
                    acc += (double)v_reduce_sum(va);
                }
#endif
                for (; i < n; i++) acc += std::fabs(s0[i]);
                q[row] = (float)acc;
                break;
            }
            case Reduce2Layer::ReduceType::L2:
            case Reduce2Layer::ReduceType::SUM_SQUARE: {
                double acc = 0.0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                if (n >= (size_t)L) {
                    v_float32 va = vx_setzero_f32();
                    for (; i + L <= n; i += L) {
                        v_float32 x = vx_load(s0 + i);
                        va = v_add(va, v_mul(x, x));
                    }
                    acc += (double)v_reduce_sum(va);
                }
#endif
                for (; i < n; i++) { double x = s0[i]; acc += x * x; }
                q[row] = (rt == Reduce2Layer::ReduceType::L2) ? (float)std::sqrt(acc) : (float)acc;
                break;
            }
            case Reduce2Layer::ReduceType::PROD: {
                double acc = 1.0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                if (n >= (size_t)L) {
                    v_float32 va = vx_setall_f32(1.0f);
                    for (; i + L <= n; i += L) va = v_mul(va, vx_load(s0 + i));
                    float buf[VTraits<v_float32>::max_nlanes];
                    v_store(buf, va);
                    for (int k = 0; k < L; k++) acc *= (double)buf[k];
                }
#endif
                for (; i < n; i++) acc *= (double)s0[i];
                q[row] = (float)acc;
                break;
            }
            default: CV_Error(Error::StsInternal, "reduceLastAxesFloatParallel_: unhandled type");
            }
        }
    }, (double)nOut * (double)innerLen / 16384.0);
}

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}  // cv::dnn

#endif  // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
