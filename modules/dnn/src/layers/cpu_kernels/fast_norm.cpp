// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "fast_norm.hpp"
#include <opencv2/core/hal/intrin.hpp>
#include <type_traits>

namespace cv { namespace dnn {

#if (CV_SIMD || CV_SIMD_SCALABLE)
// Sum and squared sum of n floats with single-precision accumulators, tail processed in scalar.
static inline void normAccumSumSqSum(const float* x, size_t n, float& sum, float& sqsum) {
    const size_t VEC_SZ = (size_t)VTraits<v_float32>::vlanes();
    v_float32 vsum0 = vx_setzero_f32(), vsum1 = vx_setzero_f32(),
              vsqsum0 = vx_setzero_f32(), vsqsum1 = vx_setzero_f32();
    size_t j = 0;
    for (; j + 2 * VEC_SZ <= n; j += 2 * VEC_SZ) {
        v_float32 v0 = vx_load(x + j), v1 = vx_load(x + j + VEC_SZ);
        vsum0 = v_add(vsum0, v0);
        vsum1 = v_add(vsum1, v1);
        vsqsum0 = v_fma(v0, v0, vsqsum0);
        vsqsum1 = v_fma(v1, v1, vsqsum1);
    }
    if (j + VEC_SZ <= n) {
        v_float32 v0 = vx_load(x + j);
        vsum0 = v_add(vsum0, v0);
        vsqsum0 = v_fma(v0, v0, vsqsum0);
        j += VEC_SZ;
    }
    float s = v_reduce_sum(v_add(vsum0, vsum1));
    float sq = v_reduce_sum(v_add(vsqsum0, vsqsum1));
    for (; j < n; j++) {
        float v = x[j];
        s += v;
        sq += v * v;
    }
    sum = s;
    sqsum = sq;
}
#endif

#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
// Same as normAccumSumSqSum but with double-precision accumulators (used where the scalar
// reference accumulates in double).
static inline void normAccumSumSqSum64f(const float* x, size_t n, double& sum, double& sqsum) {
    const size_t VEC_SZ = (size_t)VTraits<v_float32>::vlanes();
    v_float64 vsum_lo = vx_setzero_f64(), vsum_hi = vx_setzero_f64(),
              vsqsum_lo = vx_setzero_f64(), vsqsum_hi = vx_setzero_f64();
    size_t j = 0;
    for (; j + VEC_SZ <= n; j += VEC_SZ) {
        v_float32 v = vx_load(x + j);
        v_float64 vlo = v_cvt_f64(v), vhi = v_cvt_f64_high(v);
        vsum_lo = v_add(vsum_lo, vlo);
        vsum_hi = v_add(vsum_hi, vhi);
        vsqsum_lo = v_fma(vlo, vlo, vsqsum_lo);
        vsqsum_hi = v_fma(vhi, vhi, vsqsum_hi);
    }
    double s = v_reduce_sum(v_add(vsum_lo, vsum_hi));
    double sq = v_reduce_sum(v_add(vsqsum_lo, vsqsum_hi));
    for (; j < n; j++) {
        double v = (double)x[j];
        s += v;
        sq += v * v;
    }
    sum = s;
    sqsum = sq;
}
#endif

// Unchanged: only mvn_layer.cpp uses this overload, out of scope here.
void fastNorm(const Mat &input, Mat &output, float epsilon, size_t normalized_axis, bool normalize_variance) {
    const auto input_shape = shape(input);
    CV_CheckLT(normalized_axis, input_shape.size(), "fastNorm: axis out of range");

    size_t loops = static_cast<size_t>(total(input_shape, 0, static_cast<int>(normalized_axis))),
           norm_size = static_cast<size_t>(total(input_shape, static_cast<int>(normalized_axis)));
    float inv_norm_size = 1.0 / norm_size;

    auto fn = [&](const Range &r) {
        const auto *input_data = input.ptr<const float>();
        auto *output_data = output.ptr<float>();
        for (int i = r.start; i < r.end; i++) {
            const auto *x = input_data + norm_size * i;
            auto *y = output_data + norm_size * i;

            float mean = 0.f, mean_square = 0.f;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            normAccumSumSqSum(x, norm_size, mean, mean_square);
#else
            for (int j = 0; j < norm_size; j++) {
                float v = x[j];
                mean += v;
                mean_square += v * v;
            }
#endif

            mean *= inv_norm_size;
            mean_square = std::sqrt(std::max(0.f, mean_square * inv_norm_size - mean * mean) + epsilon);
            float inv_stdev = normalize_variance ? 1.f / mean_square : 1.f;

            size_t j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            const size_t VEC_SZ = (size_t)VTraits<v_float32>::vlanes();
            v_float32 vmean = vx_setall_f32(mean), vinv_stdev = vx_setall_f32(inv_stdev);
            for (; j + VEC_SZ <= norm_size; j += VEC_SZ)
                vx_store(y + j, v_mul(v_sub(vx_load(x + j), vmean), vinv_stdev));
#endif
            for (; j < norm_size; j++) {
                y[j] = (x[j] - mean) * inv_stdev;
            }
        }
    };
    double nstripes = loops * norm_size * (1 / 1024.0);
    parallel_for_(Range(0, loops), fn, nstripes);
}

// Templated on T so CV_64F gets a genuine double accumulator, not a narrowed float one.
template<typename T>
static void fastNormMeanInvStdDevImpl(const Mat& input, Mat& mean, Mat& invStdDev, T epsilon, size_t normalized_axis)
{
    CV_Assert(input.isContinuous() && mean.isContinuous() && invStdDev.isContinuous());

    const auto input_shape = shape(input);
    CV_CheckLT(normalized_axis, input_shape.size(), "fastNormMeanInvStdDev: axis out of range");

    const size_t loops = static_cast<size_t>(total(input_shape, 0, static_cast<int>(normalized_axis)));
    const size_t norm_size = static_cast<size_t>(total(input_shape, static_cast<int>(normalized_axis)));
    const T inv_norm_size = (T)1 / (T)norm_size;

    CV_CheckEQ((size_t)mean.total(), loops, "fastNormMeanInvStdDev: mean output size mismatch");
    CV_CheckEQ((size_t)invStdDev.total(), loops, "fastNormMeanInvStdDev: invStdDev output size mismatch");

    auto fn = [&](const Range& r) {
        const T* input_data = input.ptr<T>();
        T* mean_data = mean.ptr<T>();
        T* invstd_data = invStdDev.ptr<T>();
        for (int i = r.start; i < r.end; ++i)
        {
            const T* x = input_data + norm_size * (size_t)i;
            T m = 0, mean_square = 0;
            bool simd_done = false;
            if constexpr (std::is_same<T, float>::value) {
#if (CV_SIMD || CV_SIMD_SCALABLE)
                normAccumSumSqSum(x, norm_size, m, mean_square);
                simd_done = true;
#endif
            }
            if (!simd_done) {
                for (size_t j = 0; j < norm_size; ++j)
                {
                    T v = x[j];
                    m += v;
                    mean_square += v * v;
                }
            }
            m *= inv_norm_size;
            const T var = std::max((T)0, mean_square * inv_norm_size - m * m);
            const T stdev = std::sqrt(var + epsilon);
            mean_data[i] = m;
            invstd_data[i] = (T)1 / stdev;
        }
    };

    const double nstripes = loops * norm_size * (1 / 1024.0);
    parallel_for_(Range(0, (int)loops), fn, nstripes);
}

void fastNormMeanInvStdDev(const Mat& input, Mat& mean, Mat& invStdDev, float epsilon, size_t normalized_axis)
{
    int type = input.type();
    CV_CheckType(type, type == CV_32F || type == CV_64F, "fastNormMeanInvStdDev: unsupported type");
    CV_CheckTypeEQ(type, mean.type(), "fastNormMeanInvStdDev: mean must match input type");
    CV_CheckTypeEQ(type, invStdDev.type(), "fastNormMeanInvStdDev: invStdDev must match input type");

    if (type == CV_64F)
        fastNormMeanInvStdDevImpl<double>(input, mean, invStdDev, (double)epsilon, normalized_axis);
    else
        fastNormMeanInvStdDevImpl<float>(input, mean, invStdDev, epsilon, normalized_axis);
}

// RMSNorm (recenter=false) and LayerNorm/LayerNorm2's no-bias path (recenter=true).
template<typename T>
static void fastNormImpl(const Mat &input, const Mat &scale, Mat &output, T epsilon, size_t normalized_axis, bool recenter) {
    const auto input_shape = shape(input);
    CV_CheckLT(normalized_axis, input_shape.size(), "fastNorm: axis out of range");

    size_t loops = static_cast<size_t>(total(input_shape, 0, static_cast<int>(normalized_axis))),
           norm_size = static_cast<size_t>(total(input_shape, static_cast<int>(normalized_axis)));
    T inv_norm_size = (T)1 / (T)norm_size;

    auto fn = [&](const Range &r) {
        const T *input_data = input.ptr<const T>();
        const T *scale_data = scale.ptr<const T>();
        T *output_data = output.ptr<T>();
        for (int i = r.start; i < r.end; i++) {
            const T *x = input_data + norm_size * i;
            T *y = output_data + norm_size * i;

            T mean = 0, mean_square = 0;
            bool simd_done = false;
            if constexpr (std::is_same<T, float>::value) {
#if (CV_SIMD || CV_SIMD_SCALABLE)
                normAccumSumSqSum(x, norm_size, mean, mean_square);
                if (!recenter)
                    mean = 0.f;
                simd_done = true;
#endif
            }
            if (!simd_done) {
                for (int j = 0; j < norm_size; j++) {
                    T v = x[j];
                    if (recenter)
                        mean += v;
                    mean_square += v * v;
                }
            }

            mean *= inv_norm_size;
            mean_square = std::sqrt(std::max((T)0, mean_square * inv_norm_size - mean * mean) + epsilon);
            T inv_stdev = (T)1 / mean_square;

            size_t j = 0;
            if constexpr (std::is_same<T, float>::value) {
#if (CV_SIMD || CV_SIMD_SCALABLE)
                const size_t VEC_SZ = (size_t)VTraits<v_float32>::vlanes();
                v_float32 vmean = vx_setall_f32(mean), vinv_stdev = vx_setall_f32(inv_stdev);
                for (; j + VEC_SZ <= norm_size; j += VEC_SZ) {
                    v_float32 vs = vx_load(scale_data + j);
                    vx_store(y + j, v_mul(v_mul(vs, v_sub(vx_load(x + j), vmean)), vinv_stdev));
                }
#endif
            }
            for (; j < norm_size; j++) {
                y[j] = scale_data[j] * (x[j] - mean) * inv_stdev;
            }
        }
    };
    double nstripes = loops * norm_size * (1 / 1024.0);
    parallel_for_(Range(0, loops), fn, nstripes);
}

void fastNorm(const Mat &input, const Mat &scale, Mat &output, float epsilon, size_t normalized_axis, bool recenter) {
    int type = input.type();
    CV_CheckType(type, type == CV_32F || type == CV_64F, "fastNorm: unsupported type");
    CV_CheckTypeEQ(type, scale.type(), "fastNorm: scale must match input type");
    CV_CheckTypeEQ(type, output.type(), "fastNorm: output must match input type");

    if (type == CV_64F)
        fastNormImpl<double>(input, scale, output, (double)epsilon, normalized_axis, recenter);
    else
        fastNormImpl<float>(input, scale, output, epsilon, normalized_axis, recenter);
}

// Full LayerNorm (scale + bias) -- LayerNorm and LayerNorm2's with-bias path.
template<typename T>
static void fastNormImpl(const Mat &input, const Mat &scale, const Mat &bias, Mat &output, T epsilon, size_t normalized_axis) {
    const auto input_shape = shape(input);
    CV_CheckLT(normalized_axis, input_shape.size(), "fastNorm: axis out of range");
    CV_CheckEQ(scale.total(), bias.total(), "fastNorm: scale and bias should have the same shape");

    size_t loops = static_cast<size_t>(total(input_shape, 0, static_cast<int>(normalized_axis))),
           norm_size = static_cast<size_t>(total(input_shape, static_cast<int>(normalized_axis)));
    T inv_norm_size = (T)1 / (T)norm_size;

    auto fn = [&](const Range &r) {
        const T *input_data = input.ptr<const T>();
        const T *scale_data = scale.ptr<const T>();
        const T *bias_data = bias.ptr<const T>();
        T *output_data = output.ptr<T>();
        for (int i = r.start; i < r.end; i++) {
            const T *x = input_data + norm_size * i;
            T *y = output_data + norm_size * i;

            T mean = 0, mean_square = 0;
            bool simd_done = false;
            if constexpr (std::is_same<T, float>::value) {
#if (CV_SIMD || CV_SIMD_SCALABLE)
                normAccumSumSqSum(x, norm_size, mean, mean_square);
                simd_done = true;
#endif
            }
            if (!simd_done) {
                for (int j = 0; j < norm_size; j++) {
                    T v = x[j];
                    mean += v;
                    mean_square += v * v;
                }
            }

            mean *= inv_norm_size;
            mean_square = std::sqrt(std::max((T)0, mean_square * inv_norm_size - mean * mean) + epsilon);
            T inv_stdev = (T)1 / mean_square;

            size_t j = 0;
            if constexpr (std::is_same<T, float>::value) {
#if (CV_SIMD || CV_SIMD_SCALABLE)
                const size_t VEC_SZ = (size_t)VTraits<v_float32>::vlanes();
                v_float32 vmean = vx_setall_f32(mean), vinv_stdev = vx_setall_f32(inv_stdev);
                for (; j + VEC_SZ <= norm_size; j += VEC_SZ) {
                    v_float32 vs = vx_load(scale_data + j);
                    v_float32 vb = vx_load(bias_data + j);
                    vx_store(y + j, v_fma(v_mul(vs, v_sub(vx_load(x + j), vmean)), vinv_stdev, vb));
                }
#endif
            }
            for (; j < norm_size; j++) {
                y[j] = scale_data[j] * (x[j] - mean) * inv_stdev + bias_data[j];
            }
        }
    };
    double nstripes = loops * norm_size * (1 / 1024.0);
    parallel_for_(Range(0, loops), fn, nstripes);
}

void fastNorm(const Mat &input, const Mat &scale, const Mat &bias, Mat &output, float epsilon, size_t normalized_axis) {
    int type = input.type();
    CV_CheckType(type, type == CV_32F || type == CV_64F, "fastNorm: unsupported type");
    CV_CheckTypeEQ(type, scale.type(), "fastNorm: scale must match input type");
    CV_CheckTypeEQ(type, bias.type(), "fastNorm: bias must match input type");
    CV_CheckTypeEQ(type, output.type(), "fastNorm: output must match input type");

    if (type == CV_64F)
        fastNormImpl<double>(input, scale, bias, output, (double)epsilon, normalized_axis);
    else
        fastNormImpl<float>(input, scale, bias, output, epsilon, normalized_axis);
}

// InstanceNorm, BLOCK layout, CV_32F -- extracted verbatim, unchanged.
static void fastNormChannelBlockF32(const Mat &input, const Mat &scale, const Mat &bias, Mat &output, float epsilon) {
    const auto input_shape = shape(input);
    size_t C = (size_t)input_shape.C;

    CV_Assert(input.dims == 5 && output.dims == 5);
    CV_Assert(input.isContinuous() && output.isContinuous());

    const int N  = input.size[0];
    const int C1 = input.size[1];
    const int H  = input.size[2];
    const int W  = input.size[3];
    const int C0 = input.size[4];
    const int Ci = (int)C;

    const float* scale_data = scale.ptr<float>();
    const float* bias_data  = bias.ptr<float>();

    const size_t inStep0 = input.step.p[0] / sizeof(float);
    const size_t inStep1 = input.step.p[1] / sizeof(float);
    const size_t inStep2 = input.step.p[2] / sizeof(float);
    const size_t inStep3 = input.step.p[3] / sizeof(float);

    const size_t outStep0 = output.step.p[0] / sizeof(float);
    const size_t outStep1 = output.step.p[1] / sizeof(float);
    const size_t outStep2 = output.step.p[2] / sizeof(float);
    const size_t outStep3 = output.step.p[3] / sizeof(float);

    const size_t norm_size = (size_t)H * (size_t)W;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VEC_SZ = VTraits<v_float32>::vlanes();
#endif

    // Accumulators are double-precision
    const double inv_norm_size_d = 1.0 / (double)norm_size;

    parallel_for_(Range(0, N * C1), [&](const Range& r) {
        const float* inptr0 = (const float*)input.data;
        float* outptr0 = (float*)output.data;

        AutoBuffer<double> sumBuf(C0 * 2);
        double* sum   = sumBuf.data();
        double* sqsum = sum + C0;
        AutoBuffer<float> abBuf(C0 * 2);
        float* alpha = abBuf.data();
        float* beta  = alpha + C0;

        for (int i = r.start; i < r.end; ++i) {
            int n  = i / C1;
            int c1 = i - n * C1;
            int cbase = c1 * C0;
            int validC0 = std::min(C0, std::max(0, Ci - cbase));

            const float* inbase  = inptr0  + n * inStep0 + c1 * inStep1;
            float*       outbase = outptr0 + n * outStep0 + c1 * outStep1;

            int c0 = 0;
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
            const int VEC_SZ_D = VTraits<v_float64>::vlanes();
            CV_DbgAssert(VEC_SZ == 2 * VEC_SZ_D);
            for (; c0 <= validC0 - VEC_SZ; c0 += VEC_SZ) {
                v_float64 vsum_lo = vx_setzero_f64(), vsum_hi = vx_setzero_f64();
                v_float64 vsqsum_lo = vx_setzero_f64(), vsqsum_hi = vx_setzero_f64();
                for (int h = 0; h < H; ++h) {
                    const float* inrow = inbase + h * inStep2;
                    for (int w = 0; w < W; ++w) {
                        v_float32 v = vx_load(inrow + w * inStep3 + c0);
                        v_float64 vlo = v_cvt_f64(v);
                        v_float64 vhi = v_cvt_f64_high(v);
                        vsum_lo = v_add(vsum_lo, vlo);
                        vsum_hi = v_add(vsum_hi, vhi);
                        vsqsum_lo = v_fma(vlo, vlo, vsqsum_lo);
                        vsqsum_hi = v_fma(vhi, vhi, vsqsum_hi);
                    }
                }
                vx_store(sum   + c0,             vsum_lo);
                vx_store(sum   + c0 + VEC_SZ_D, vsum_hi);
                vx_store(sqsum + c0,             vsqsum_lo);
                vx_store(sqsum + c0 + VEC_SZ_D, vsqsum_hi);
            }
#endif
            for (; c0 < validC0; ++c0) {
                double s = 0., sq = 0.;
                for (int h = 0; h < H; ++h) {
                    const float* inrow = inbase + h * inStep2;
                    for (int w = 0; w < W; ++w) {
                        double v = (double)inrow[w * inStep3 + c0];
                        s += v;
                        sq += v * v;
                    }
                }
                sum[c0] = s;
                sqsum[c0] = sq;
            }

            for (int c = 0; c < validC0; ++c) {
                double mean = sum[c] * inv_norm_size_d;
                double var = std::max(0., sqsum[c] * inv_norm_size_d - mean * mean);
                float inv_stdev = 1.f / std::sqrt((float)var + epsilon);
                alpha[c] = scale_data[cbase + c] * inv_stdev;
                beta[c]  = bias_data[cbase + c] - alpha[c] * (float)mean;
            }

            c0 = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            for (; c0 <= validC0 - VEC_SZ; c0 += VEC_SZ) {
                v_float32 va = vx_load(alpha + c0);
                v_float32 vb = vx_load(beta + c0);
                for (int h = 0; h < H; ++h) {
                    const float* inrow  = inbase + h * inStep2;
                    float*       outrow = outbase + h * outStep2;
                    for (int w = 0; w < W; ++w) {
                        v_float32 vin = vx_load(inrow + w * inStep3 + c0);
                        vx_store(outrow + w * outStep3 + c0, v_fma(vin, va, vb));
                    }
                }
            }
#endif
            for (; c0 < validC0; ++c0) {
                float a = alpha[c0], b = beta[c0];
                for (int h = 0; h < H; ++h) {
                    const float* inrow  = inbase + h * inStep2;
                    float*       outrow = outbase + h * outStep2;
                    for (int w = 0; w < W; ++w)
                        outrow[w * outStep3 + c0] = inrow[w * inStep3 + c0] * a + b;
                }
            }

            int c0_pad = validC0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            for (; c0_pad <= C0 - VEC_SZ; c0_pad += VEC_SZ) {
                v_float32 vzero = vx_setzero_f32();
                for (int h = 0; h < H; ++h) {
                    float* outrow = outbase + h * outStep2;
                    for (int w = 0; w < W; ++w)
                        vx_store(outrow + w * outStep3 + c0_pad, vzero);
                }
            }
#endif
            for (; c0_pad < C0; ++c0_pad)
                for (int h = 0; h < H; ++h) {
                    float* outrow = outbase + h * outStep2;
                    for (int w = 0; w < W; ++w)
                        outrow[w * outStep3 + c0_pad] = 0.f;
                }
        }
    });
}

// InstanceNorm, BLOCK layout, CV_64F -- scalar counterpart to fastNormChannelBlockF32.
template<typename T>
static void fastNormChannelBlockT(const Mat &input, const Mat &scale, const Mat &bias, Mat &output, T epsilon) {
    const auto input_shape = shape(input);
    size_t C = (size_t)input_shape.C;

    CV_Assert(input.dims == 5 && output.dims == 5);
    CV_Assert(input.isContinuous() && output.isContinuous());

    const int N  = input.size[0];
    const int C1 = input.size[1];
    const int H  = input.size[2];
    const int W  = input.size[3];
    const int C0 = input.size[4];
    const int Ci = (int)C;

    const T* scale_data = scale.ptr<T>();
    const T* bias_data  = bias.ptr<T>();

    const size_t inStep0 = input.step.p[0] / sizeof(T);
    const size_t inStep1 = input.step.p[1] / sizeof(T);
    const size_t inStep2 = input.step.p[2] / sizeof(T);
    const size_t inStep3 = input.step.p[3] / sizeof(T);

    const size_t outStep0 = output.step.p[0] / sizeof(T);
    const size_t outStep1 = output.step.p[1] / sizeof(T);
    const size_t outStep2 = output.step.p[2] / sizeof(T);
    const size_t outStep3 = output.step.p[3] / sizeof(T);

    const size_t norm_size = (size_t)H * (size_t)W;
    const T inv_norm_size = (T)1 / (T)norm_size;

    parallel_for_(Range(0, N * C1), [&](const Range& r) {
        const T* inptr0 = (const T*)input.data;
        T* outptr0 = (T*)output.data;

        AutoBuffer<T> sumBuf(C0 * 2);
        T* sum   = sumBuf.data();
        T* sqsum = sum + C0;
        AutoBuffer<T> abBuf(C0 * 2);
        T* alpha = abBuf.data();
        T* beta  = alpha + C0;

        for (int i = r.start; i < r.end; ++i) {
            int n  = i / C1;
            int c1 = i - n * C1;
            int cbase = c1 * C0;
            int validC0 = std::min(C0, std::max(0, Ci - cbase));

            const T* inbase  = inptr0  + n * inStep0 + c1 * inStep1;
            T*       outbase = outptr0 + n * outStep0 + c1 * outStep1;

            for (int c0 = 0; c0 < validC0; ++c0) {
                T s = 0, sq = 0;
                for (int h = 0; h < H; ++h) {
                    const T* inrow = inbase + h * inStep2;
                    for (int w = 0; w < W; ++w) {
                        T v = inrow[w * inStep3 + c0];
                        s += v;
                        sq += v * v;
                    }
                }
                sum[c0] = s;
                sqsum[c0] = sq;
            }

            for (int c = 0; c < validC0; ++c) {
                T mean = sum[c] * inv_norm_size;
                T var = std::max((T)0, sqsum[c] * inv_norm_size - mean * mean);
                T inv_stdev = (T)1 / std::sqrt(var + epsilon);
                alpha[c] = scale_data[cbase + c] * inv_stdev;
                beta[c]  = bias_data[cbase + c] - alpha[c] * mean;
            }

            for (int c0 = 0; c0 < validC0; ++c0) {
                T a = alpha[c0], b = beta[c0];
                for (int h = 0; h < H; ++h) {
                    const T* inrow  = inbase + h * inStep2;
                    T*       outrow = outbase + h * outStep2;
                    for (int w = 0; w < W; ++w)
                        outrow[w * outStep3 + c0] = inrow[w * inStep3 + c0] * a + b;
                }
            }

            for (int c0_pad = validC0; c0_pad < C0; ++c0_pad)
                for (int h = 0; h < H; ++h) {
                    T* outrow = outbase + h * outStep2;
                    for (int w = 0; w < W; ++w)
                        outrow[w * outStep3 + c0_pad] = 0;
                }
        }
    });
}

// InstanceNorm, plain NCHW layout.
template<typename T>
static void fastNormChannelImpl(const Mat &input, const Mat &scale, const Mat &bias, Mat &output, T epsilon) {
    const auto input_shape = shape(input);
    size_t C = input_shape[1];

    size_t N = input_shape[0];
    CV_CheckGE(input.dims, 3, "fastNormChannel: input dimension >= 3");

    size_t loops = N * C,
           norm_size = static_cast<size_t>(total(input_shape, 2));

    auto fn = [&](const Range &r) {
        const T *input_data = input.ptr<const T>();
        const T *scale_data = scale.ptr<const T>();
        const T *bias_data = bias.ptr<const T>();
        T *output_data = output.ptr<T>();
        for (int i = r.start; i < r.end; i++) {
            const T *x = input_data + norm_size * i;
            T *y = output_data + norm_size * i;

            double dmean = 0., dmean_sq = 0.;
            bool simd_done = false;
            if constexpr (std::is_same<T, float>::value) {
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
                normAccumSumSqSum64f(x, norm_size, dmean, dmean_sq);
                simd_done = true;
#endif
            }
            if (!simd_done) {
                for (size_t j = 0; j < norm_size; j++) {
                    double v = (double)x[j];
                    dmean += v;
                    dmean_sq += v * v;
                }
            }

            T mean = (T)(dmean / norm_size);
            T var = (T)std::max(0., dmean_sq / norm_size - (double)mean * (double)mean);
            T inv_stdev = (T)1 / std::sqrt(var + epsilon);

            size_t c = i % C;
            T s = scale_data[c] * inv_stdev, b = bias_data[c];
            size_t j = 0;
            if constexpr (std::is_same<T, float>::value) {
#if (CV_SIMD || CV_SIMD_SCALABLE)
                const size_t VEC_SZ = (size_t)VTraits<v_float32>::vlanes();
                v_float32 vmean = vx_setall_f32(mean), vs = vx_setall_f32(s), vb = vx_setall_f32(b);
                for (; j + VEC_SZ <= norm_size; j += VEC_SZ)
                    vx_store(y + j, v_fma(v_sub(vx_load(x + j), vmean), vs, vb));
#endif
            }
            for (; j < norm_size; j++) {
                y[j] = s * (x[j] - mean) + b;
            }
        }
    };
    double nstripes = loops * norm_size * (1 / 1024.0);
    parallel_for_(Range(0, loops), fn, nstripes);
}

void fastNormChannel(const Mat &input, const Mat &scale, const Mat &bias, Mat &output, float epsilon) {
    const auto input_shape = shape(input);
    size_t C = input_shape.layout == DATA_LAYOUT_BLOCK ? (size_t)input_shape.C : input_shape[1];
    CV_CheckEQ(scale.total(), C, "fastNormChannel: scale should be a 1d tensor and match the channel of input");
    CV_CheckEQ(bias.total(), C, "fastNormChannel: bias should be a 1d tensor and match the channel of input");

    int type = input.type();
    CV_CheckType(type, type == CV_32F || type == CV_64F, "fastNormChannel: unsupported type");
    CV_CheckTypeEQ(type, output.type(), "fastNormChannel: output must match input type");
    CV_CheckTypeEQ(type, scale.type(), "fastNormChannel: scale must match input type");
    CV_CheckTypeEQ(type, bias.type(), "fastNormChannel: bias must match input type");

    if (input_shape.layout == DATA_LAYOUT_BLOCK) {
        if (type == CV_64F)
            fastNormChannelBlockT<double>(input, scale, bias, output, (double)epsilon);
        else
            fastNormChannelBlockF32(input, scale, bias, output, epsilon);
        return;
    }

    if (type == CV_64F)
        fastNormChannelImpl<double>(input, scale, bias, output, (double)epsilon);
    else
        fastNormChannelImpl<float>(input, scale, bias, output, epsilon);
}

// GroupNorm, BLOCK layout, CV_32F -- extracted verbatim, unchanged.
static void fastNormGroupBlockF32(const Mat &input, const Mat &scale, const Mat &bias, Mat &output, float epsilon, size_t num_groups) {
    const auto input_shape = shape(input);
    size_t C = (size_t)input_shape.C;

    CV_Assert(input.dims == 5 && output.dims == 5);
    CV_Assert(input.isContinuous() && output.isContinuous());

    const int N  = input.size[0];
    const int H  = input.size[2];
    const int W  = input.size[3];
    const int C0 = input.size[4];
    const int Ci = (int)C;

    const float* scale_data = scale.ptr<float>();
    const float* bias_data  = bias.ptr<float>();

    const size_t inStep0 = input.step.p[0] / sizeof(float);
    const size_t inStep1 = input.step.p[1] / sizeof(float);
    const size_t inStep2 = input.step.p[2] / sizeof(float);
    const size_t inStep3 = input.step.p[3] / sizeof(float);

    const size_t outStep0 = output.step.p[0] / sizeof(float);
    const size_t outStep1 = output.step.p[1] / sizeof(float);
    const size_t outStep2 = output.step.p[2] / sizeof(float);
    const size_t outStep3 = output.step.p[3] / sizeof(float);

    const int channels_per_group = Ci / (int)num_groups;
    const size_t norm_size = (size_t)channels_per_group * (size_t)H * (size_t)W;
    const double inv_norm_size = 1.0 / (double)norm_size;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VEC_SZ = VTraits<v_float32>::vlanes();
#endif

    parallel_for_(Range(0, N * (int)num_groups), [&](const Range& r) {
        const float* inptr = (const float*)input.data;
        float* outptr = (float*)output.data;

        AutoBuffer<float> buf(C0 * 2);
        float* alpha = buf.data();
        float* beta  = alpha + C0;

        for (int i = r.start; i < r.end; ++i) {
            int n = i / (int)num_groups;
            int g = i - n * (int)num_groups;
            int c_start = g * channels_per_group;
            int c_end   = c_start + channels_per_group;

            double group_sum = 0., group_sqsum = 0.;
            for (int c = c_start; c < c_end; c++) {
                int c1 = c / C0;
                int c0 = c % C0;
                const float* inbase = inptr + n * inStep0 + c1 * inStep1;
                for (int h = 0; h < H; ++h) {
                    const float* inrow = inbase + h * inStep2;
                    for (int w = 0; w < W; ++w) {
                        double v = (double)inrow[w * inStep3 + c0];
                        group_sum += v;
                        group_sqsum += v * v;
                    }
                }
            }

            float mean = (float)(group_sum * inv_norm_size);
            float var  = std::max(0.f, (float)(group_sqsum * inv_norm_size - (double)mean * (double)mean));
            float inv_stdev = 1.f / std::sqrt(var + epsilon);

            for (int c1_start = c_start / C0, c1_end_idx = (c_end - 1) / C0 + 1,
                     c1 = c1_start; c1 < c1_end_idx; ++c1) {
                int cbase = c1 * C0;
                int c0_lo = std::max(0, c_start - cbase);
                int c0_hi = std::min(C0, c_end - cbase);
                int validC0 = std::min(C0, std::max(0, Ci - cbase));

                for (int c0 = c0_lo; c0 < c0_hi; ++c0) {
                    alpha[c0] = scale_data[cbase + c0] * inv_stdev;
                    beta[c0]  = bias_data[cbase + c0] - alpha[c0] * mean;
                }

                const float* inbase  = inptr  + n * inStep0 + c1 * inStep1;
                float*       outbase = outptr + n * outStep0 + c1 * outStep1;

                int c0 = c0_lo;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                for (; c0 <= c0_hi - VEC_SZ; c0 += VEC_SZ) {
                    v_float32 va = vx_load(alpha + c0);
                    v_float32 vb = vx_load(beta + c0);
                    for (int h = 0; h < H; ++h) {
                        const float* inrow  = inbase + h * inStep2;
                        float*       outrow = outbase + h * outStep2;
                        for (int w = 0; w < W; ++w) {
                            v_float32 vin = vx_load(inrow + w * inStep3 + c0);
                            vx_store(outrow + w * outStep3 + c0, v_fma(vin, va, vb));
                        }
                    }
                }
#endif
                for (; c0 < c0_hi; ++c0) {
                    float a = alpha[c0], b = beta[c0];
                    for (int h = 0; h < H; ++h) {
                        const float* inrow  = inbase + h * inStep2;
                        float*       outrow = outbase + h * outStep2;
                        for (int w = 0; w < W; ++w)
                            outrow[w * outStep3 + c0] = inrow[w * inStep3 + c0] * a + b;
                    }
                }

                for (int c0_pad = std::max(c0_hi, validC0); c0_pad < C0; ++c0_pad)
                    for (int h = 0; h < H; ++h) {
                        float* outrow = outbase + h * outStep2;
                        for (int w = 0; w < W; ++w)
                            outrow[w * outStep3 + c0_pad] = 0.f;
                    }
            }
        }
    });
}

// GroupNorm, BLOCK layout, CV_64F -- scalar counterpart to fastNormGroupBlockF32.
template<typename T>
static void fastNormGroupBlockT(const Mat &input, const Mat &scale, const Mat &bias, Mat &output, T epsilon, size_t num_groups) {
    const auto input_shape = shape(input);
    size_t C = (size_t)input_shape.C;

    CV_Assert(input.dims == 5 && output.dims == 5);
    CV_Assert(input.isContinuous() && output.isContinuous());

    const int N  = input.size[0];
    const int H  = input.size[2];
    const int W  = input.size[3];
    const int C0 = input.size[4];
    const int Ci = (int)C;

    const T* scale_data = scale.ptr<T>();
    const T* bias_data  = bias.ptr<T>();

    const size_t inStep0 = input.step.p[0] / sizeof(T);
    const size_t inStep1 = input.step.p[1] / sizeof(T);
    const size_t inStep2 = input.step.p[2] / sizeof(T);
    const size_t inStep3 = input.step.p[3] / sizeof(T);

    const size_t outStep0 = output.step.p[0] / sizeof(T);
    const size_t outStep1 = output.step.p[1] / sizeof(T);
    const size_t outStep2 = output.step.p[2] / sizeof(T);
    const size_t outStep3 = output.step.p[3] / sizeof(T);

    const int channels_per_group = Ci / (int)num_groups;
    const size_t norm_size = (size_t)channels_per_group * (size_t)H * (size_t)W;
    const T inv_norm_size = (T)1 / (T)norm_size;

    parallel_for_(Range(0, N * (int)num_groups), [&](const Range& r) {
        const T* inptr = (const T*)input.data;
        T* outptr = (T*)output.data;

        AutoBuffer<T> buf(C0 * 2);
        T* alpha = buf.data();
        T* beta  = alpha + C0;

        for (int i = r.start; i < r.end; ++i) {
            int n = i / (int)num_groups;
            int g = i - n * (int)num_groups;
            int c_start = g * channels_per_group;
            int c_end   = c_start + channels_per_group;

            T group_sum = 0, group_sqsum = 0;
            for (int c = c_start; c < c_end; c++) {
                int c1 = c / C0;
                int c0 = c % C0;
                const T* inbase = inptr + n * inStep0 + c1 * inStep1;
                for (int h = 0; h < H; ++h) {
                    const T* inrow = inbase + h * inStep2;
                    for (int w = 0; w < W; ++w) {
                        T v = inrow[w * inStep3 + c0];
                        group_sum += v;
                        group_sqsum += v * v;
                    }
                }
            }

            T mean = group_sum * inv_norm_size;
            T var  = std::max((T)0, group_sqsum * inv_norm_size - mean * mean);
            T inv_stdev = (T)1 / std::sqrt(var + epsilon);

            for (int c1_start = c_start / C0, c1_end_idx = (c_end - 1) / C0 + 1,
                     c1 = c1_start; c1 < c1_end_idx; ++c1) {
                int cbase = c1 * C0;
                int c0_lo = std::max(0, c_start - cbase);
                int c0_hi = std::min(C0, c_end - cbase);
                int validC0 = std::min(C0, std::max(0, Ci - cbase));

                for (int c0 = c0_lo; c0 < c0_hi; ++c0) {
                    alpha[c0] = scale_data[cbase + c0] * inv_stdev;
                    beta[c0]  = bias_data[cbase + c0] - alpha[c0] * mean;
                }

                const T* inbase  = inptr  + n * inStep0 + c1 * inStep1;
                T*       outbase = outptr + n * outStep0 + c1 * outStep1;

                for (int c0 = c0_lo; c0 < c0_hi; ++c0) {
                    T a = alpha[c0], b = beta[c0];
                    for (int h = 0; h < H; ++h) {
                        const T* inrow  = inbase + h * inStep2;
                        T*       outrow = outbase + h * outStep2;
                        for (int w = 0; w < W; ++w)
                            outrow[w * outStep3 + c0] = inrow[w * inStep3 + c0] * a + b;
                    }
                }

                for (int c0_pad = std::max(c0_hi, validC0); c0_pad < C0; ++c0_pad)
                    for (int h = 0; h < H; ++h) {
                        T* outrow = outbase + h * outStep2;
                        for (int w = 0; w < W; ++w)
                            outrow[w * outStep3 + c0_pad] = 0;
                    }
            }
        }
    });
}

// GroupNorm, plain NCHW layout.
template<typename T>
static void fastNormGroupImpl(const Mat &input, const Mat &scale, const Mat &bias, Mat &output, T epsilon, size_t num_groups) {
    const auto input_shape = shape(input);
    size_t C = input_shape[1];

    size_t N = input_shape[0];
    CV_CheckGE(input.dims, 3, "fastNormGroup: input dimension >= 3");

    size_t channels_per_group = C / num_groups;
    size_t loops = N * num_groups;
    size_t norm_size = static_cast<size_t>(total(input_shape, 2) * channels_per_group);
    size_t step = norm_size / channels_per_group;

    auto fn = [&](const Range &r) {
        const T *input_data = input.ptr<const T>();
        const T *scale_data = scale.ptr<const T>();
        const T *bias_data = bias.ptr<const T>();
        T *output_data = output.ptr<T>();

        for (int i = r.start; i < r.end; i++) {
            const T *x = input_data + norm_size * i;
            T *y = output_data + norm_size * i;

            double dmean = 0., dmean_sq = 0.;
            bool simd_done = false;
            if constexpr (std::is_same<T, float>::value) {
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
                normAccumSumSqSum64f(x, norm_size, dmean, dmean_sq);
                simd_done = true;
#endif
            }
            if (!simd_done) {
                for (size_t j = 0; j < norm_size; j++) {
                    double v = (double)x[j];
                    dmean += v;
                    dmean_sq += v * v;
                }
            }

            T mean = (T)(dmean / norm_size);
            T var = (T)std::max(0., dmean_sq / norm_size - (double)mean * (double)mean);
            T inv_stdev = (T)1 / std::sqrt(var + epsilon);

            // Loop restructured (channel outer, offset inner) instead of j/step per
            // element -- avoids a per-element division, cheap win for T=double too and
            // what lets the T=float instantiation vectorize below.
            size_t group_idx = i % num_groups * channels_per_group;
            size_t j = 0;
            for (size_t c_idx = 0; c_idx < channels_per_group; c_idx++) {
                size_t c = group_idx + c_idx;
                T s = scale_data[c] * inv_stdev, b = bias_data[c];
                const size_t j_end = j + step;
                if constexpr (std::is_same<T, float>::value) {
#if (CV_SIMD || CV_SIMD_SCALABLE)
                    const size_t VEC_SZ = (size_t)VTraits<v_float32>::vlanes();
                    v_float32 vmean = vx_setall_f32(mean), vs = vx_setall_f32(s), vb = vx_setall_f32(b);
                    for (; j + VEC_SZ <= j_end; j += VEC_SZ)
                        vx_store(y + j, v_fma(v_sub(vx_load(x + j), vmean), vs, vb));
#endif
                }
                for (; j < j_end; j++)
                    y[j] = s * (x[j] - mean) + b;
            }
        }
    };

    double nstripes = loops * norm_size * (1 / 1024.0);
    parallel_for_(Range(0, loops), fn, nstripes);
}

void fastNormGroup(const Mat &input, const Mat &scale, const Mat &bias, Mat &output, float epsilon, size_t num_groups) {
    const auto input_shape = shape(input);
    size_t C = input_shape.layout == DATA_LAYOUT_BLOCK ? (size_t)input_shape.C : input_shape[1];
    CV_CheckEQ(scale.total(), bias.total(), "fastNormGroup: scale and bias should have the same shape");
    CV_CheckEQ(scale.total(), C, "fastNormGroup: scale should be a 1d tensor and match the channel of input");

    int type = input.type();
    CV_CheckType(type, type == CV_32F || type == CV_64F, "fastNormGroup: unsupported type");
    CV_CheckTypeEQ(type, output.type(), "fastNormGroup: output must match input type");
    CV_CheckTypeEQ(type, scale.type(), "fastNormGroup: scale must match input type");
    CV_CheckTypeEQ(type, bias.type(), "fastNormGroup: bias must match input type");

    if (input_shape.layout == DATA_LAYOUT_BLOCK) {
        if (type == CV_64F)
            fastNormGroupBlockT<double>(input, scale, bias, output, (double)epsilon, num_groups);
        else
            fastNormGroupBlockF32(input, scale, bias, output, epsilon, num_groups);
        return;
    }

    if (type == CV_64F)
        fastNormGroupImpl<double>(input, scale, bias, output, (double)epsilon, num_groups);
    else
        fastNormGroupImpl<float>(input, scale, bias, output, epsilon, num_groups);
}

}} // cv::dnn
