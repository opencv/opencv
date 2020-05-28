// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA_FUNCTORS_HPP
#define OPENCV_DNN_SRC_CUDA_FUNCTORS_HPP

#include <cuda_runtime.h>

#include "math.hpp"

namespace cv { namespace dnn { namespace cuda4dnn  { namespace kernels {

template <class T>
struct abs_functor {
    __device__ T operator()(T value) {
        using csl::device::abs;
        return abs(value);
    }
};

template <class T>
struct tanh_functor {
    __device__ T operator()(T value) {
        using csl::device::tanh;
        return tanh(value);
    }
};

template <class T>
struct swish_functor {
    __device__ T operator()(T value) {
        // f(x) = x * sigmoid(x)
        using csl::device::fast_divide;
        using csl::device::fast_exp;
        return fast_divide(value, static_cast<T>(1) + fast_exp(-value));
    }
};

template <class T>
struct mish_functor {
    __device__ T operator()(T value) {
        using csl::device::tanh;
        using csl::device::log1pexp;
        return value * tanh(log1pexp(value));
    }
};

template <>
struct mish_functor<float> {
    __device__ float operator()(float value) {
        // f(x) = x * tanh(log1pexp(x));
        using csl::device::fast_divide;
        using csl::device::fast_exp;

        auto e = fast_exp(value);
        if (value <= -18.0f)
            return value * e;

        auto n = e * e + 2 * e;
        if (value <= -5.0f)
            return value * fast_divide(n, n + 2);

        return value - 2 * fast_divide(value, n + 2);
    }
};

template <class T>
struct sigmoid_functor {
    __device__ T operator()(T value) {
        using csl::device::fast_sigmoid;
        return fast_sigmoid(value);
    }
};

template <class T>
struct bnll_functor {
    __device__ T operator()(T value) {
        using csl::device::log1pexp;
        return value > T(0) ? value + log1pexp(-value) : log1pexp(value);
    }
};

template <class T>
struct elu_functor {
    __device__ T operator()(T value) {
        using csl::device::expm1;
        return value >= T(0) ? value : expm1(value);
    }
};

template <class T>
struct relu_functor {
    __device__ relu_functor(T slope_) : slope{slope_} { }
    __device__ T operator()(T value) {
        using csl::device::log1pexp;
        return value >= T(0) ? value : slope * value;
    }

    T slope;
};

template <class T>
struct clipped_relu_functor {
    __device__ clipped_relu_functor(T floor_, T ceiling_) : floor{floor_}, ceiling{ceiling_} { }
    __device__ T operator()(T value) {
        using csl::device::clamp;
        return clamp(value, floor, ceiling);
    }

    T floor, ceiling;
};

template <class T>
struct power_functor {
    __device__ power_functor(T exp_, T scale_, T shift_) : exp{exp_}, scale{scale_}, shift{shift_} { }
    __device__ T operator()(T value) {
        using csl::device::pow;
        return pow(shift + scale * value, exp);
    }

    T exp, scale, shift;
};

template <class T>
struct max_functor {
    __device__ T operator()(T x, T y) {
        using csl::device::max;
        return max(x, y);
    }
};

template <class T>
struct sum_functor {
    __device__ T operator()(T x, T y) { return x + y; }
};

template <class T>
struct scaled_sum_functor {
    __device__ scaled_sum_functor(T scale_x_, T scale_y_)
        : scale_x{scale_x_}, scale_y{scale_y_} { }

    __device__ T operator()(T x, T y) { return scale_x * x + scale_y * y; }

    T scale_x, scale_y;
};

template <class T>
struct product_functor {
    __device__ T operator()(T x, T y) { return x * y; }
};

template <class T>
struct div_functor {
    __device__ T operator()(T x, T y) { return x / y; }
};

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA_FUNCTORS_HPP */