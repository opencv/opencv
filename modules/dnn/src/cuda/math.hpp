// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA_MATH_HPP
#define OPENCV_DNN_SRC_CUDA_MATH_HPP

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace device {

    template <class T> __device__ T abs(T val) { return (val < T(0) ? -val : val); }
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template <> inline __device__ __half2 abs(__half2 val) {
        val.x = abs(val.x);
        val.y = abs(val.y);
        return val;
    }
#endif
    template <> inline __device__ float abs(float val) { return fabsf(val); }
    template <> inline __device__ double abs(double val) { return fabs(val); }

    template <class T> __device__ T exp(T val);
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template <> inline __device__ __half exp(__half val) { return hexp(val); }
    template <> inline __device__ __half2 exp(__half2 val) { return h2exp(val); }
#endif
    template <> inline __device__ float exp(float val) { return expf(val); }
    template <> inline __device__ double exp(double val) { return ::exp(val); }

    template <class T> __device__ T expm1(T val);
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template <> inline __device__ __half expm1(__half val) { return hexp(val) - __half(1); }
    template <> inline __device__ __half2 expm1(__half2 val) { return h2exp(val) - __half2(1, 1); }
#endif
    template <> inline __device__ float expm1(float val) { return expm1f(val); }
    template <> inline __device__ double expm1(double val) { return ::expm1(val); }

    template <class T> __device__ T max(T x, T y) { return (x > y ? x : y); }
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template <> inline __device__ __half2 max(__half2 a, __half2 b) {
        a.x = max(a.x, a.x);
        a.y = max(a.y, b.y);
        return a;
    }
#endif
    template <> inline __device__ float max(float x, float y) { return fmaxf(x, y); }
    template <> inline __device__ double max(double x, double y) { return fmax(x, y); }

    template <class T> __device__ T min(T x, T y) { return (x > y ? y : x); }
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template <> inline __device__ __half2 min(__half2 a, __half2 b) {
        a.x = min(a.x, a.x);
        a.y = min(a.y, b.y);
        return a;
    }
#endif
    template <> inline __device__ float min(float x, float y) { return fminf(x, y); }
    template <> inline __device__ double min(double x, double y) { return fmin(x, y); }

    template <class T> __device__ T log1p(T val);
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template <> inline __device__ __half log1p(__half val) { return hlog(__half(1) + val); }
    template <> inline __device__ __half2 log1p(__half2 val) { return h2log(__half2(1, 1) + val); }
#endif
    template <> inline __device__ float log1p(float val) { return log1pf(val); }

    template <class T> __device__ T log1pexp(T val);
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template <> inline __device__ __half log1pexp(__half val) {
        if (val <= __half(-4.0))
            return exp(val);
        else if (val <= __half(8.0))
            return log1p(exp(val));
        else if (val <= __half(8.7))
            return val + exp(-val);
        else
            return val;
    }
    template <> inline __device__ __half2 log1pexp(__half2 val) {
        val.x = log1pexp(val.x);
        val.y = log1pexp(val.y);
        return val;
    }
#endif
    template <> inline __device__ float log1pexp(float val) {
        if (val <= -20)
            return expf(val);
        else if (val <= 9.0)
            return log1pf(expf(val));
        else if (val <= 14.6)
            return val + exp(-val);
        else
            return val;
    }
    template <> inline __device__ double log1pexp(double val) {
        if (val <= -37)
            return exp(val);
        else if (val <= 18)
            return log1p(exp(val));
        else if (val <= 33.3)
            return val + exp(-val);
        else
            return val;
    }

    template <class T> __device__ T tanh(T val);
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template <> inline __device__ __half tanh(__half val) { return tanhf(val); }
    template <> inline __device__ __half2 tanh(__half2 val) { return __half2(tanh(val.x), tanh(val.y)); }
#endif
    template <> inline __device__ float tanh(float val) { return tanhf(val); }
    template <> inline __device__ double tanh(double val) { return ::tanh(val); }

    template <class T> __device__ T pow(T val, T exp);
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template <> inline __device__ __half pow(__half val, __half exp) { return powf(val, exp); }
    template <> inline __device__ __half2 pow(__half2 val, __half2 exp) { return __half2(pow(val.x, exp.x), pow(val.y, exp.y)); }
#endif
    template <> inline __device__ float pow(float val, float exp) { return powf(val, exp); }
    template <> inline __device__ double pow(double val, double exp) { return ::pow(val, exp); }

    template <class T> __device__ T sqrt(T val);
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template <> inline __device__ __half sqrt(__half val) { return hsqrt(val); }
    template <> inline __device__ __half2 sqrt(__half2 val) { return h2sqrt(val); }
#endif
    template <> inline __device__ float sqrt(float val) { return sqrtf(val); }
    template <> inline __device__ double sqrt(double val) { return ::sqrt(val); }

    template <class T> __device__ T rsqrt(T val);
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template <> inline __device__ __half rsqrt(__half val) { return hrsqrt(val); }
    template <> inline __device__ __half2 rsqrt(__half2 val) { return h2rsqrt(val); }
#endif
    template <> inline __device__ float rsqrt(float val) { return rsqrtf(val); }
    template <> inline __device__ double rsqrt(double val) { return ::rsqrt(val); }

    template <class T> __device__ T sigmoid(T val) { return T(1) / (T(1) + exp(-val)); }
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template <> inline __device__ __half2 sigmoid(__half2 val) { return __half2(1, 1) / (__half2(1, 1) + exp(__hneg2(val))); }
#endif

    template <class T> __device__ T clamp(T value, T lower, T upper) { return min(max(value, lower), upper); }

    template <class T> __device__ T round(T value);
    template <> inline __device__ double round(double value) { return ::round(value); }
    template <> inline __device__ float round(float value) { return roundf(value); }
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template <> inline __device__ __half round(__half value) { return hrint(value); }
    template <> inline __device__ __half2 round(__half2 value) { return h2rint(value); }
#endif

    template <class T> __device__ T ceil(T value);
    template <> inline __device__ double ceil(double value) { return ::ceil(value); }
    template <> inline __device__ float ceil(float value) { return ceilf(value); }
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template <> inline __device__ __half ceil(__half value) { return hceil(value); }
    template <> inline __device__ __half2 ceil(__half2 value) { return h2ceil(value); }
#endif

}}}}} /* namespace cv::dnn::cuda4dnn::csl::device */

#endif /* OPENCV_DNN_SRC_CUDA_MATH_HPP */
