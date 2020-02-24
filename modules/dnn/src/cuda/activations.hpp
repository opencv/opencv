// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

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
            using csl::device::sigmoid;
            return value * sigmoid(value);
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

    template <class T>
    struct sigmoid_functor {
        __device__ T operator()(T value) {
            using csl::device::sigmoid;
            return sigmoid(value);
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

}}}} /* namespace cv::dnn::cuda4dnn::kernels */