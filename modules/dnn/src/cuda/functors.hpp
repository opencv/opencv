// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA_FUNCTORS_HPP
#define OPENCV_DNN_SRC_CUDA_FUNCTORS_HPP

#include <cuda_runtime.h>

#include "math.hpp"

#include "../cuda4dnn/csl/nvcc_defs.hpp"

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

template <class T>
struct IdentityFunctor {
    struct Params {
        CUDA4DNN_HOST_DEVICE Params() { }
    };

    CUDA4DNN_DEVICE IdentityFunctor() { }
    CUDA4DNN_DEVICE IdentityFunctor(const Params& params) { }

    CUDA4DNN_DEVICE T operator()(T value) {
        return value;
    };
};

template <class T>
struct ReLUFunctor {
    struct Params {
        CUDA4DNN_HOST_DEVICE Params() : slope(0) { }
        CUDA4DNN_HOST_DEVICE Params(T slope_) : slope(slope_) { }
        T slope;
    };

    CUDA4DNN_DEVICE ReLUFunctor() : ReLUFunctor(Params{}) { }
    CUDA4DNN_DEVICE ReLUFunctor(const Params& params) : slope(params.slope) { }

    CUDA4DNN_DEVICE T operator()(T value) {
        using csl::device::log1pexp;
        return value >= T(0) ? value : slope * value;
    }

    T slope;
};

template <class T>
struct ClippedReLUFunctor {
    struct Params {
        CUDA4DNN_HOST_DEVICE Params() : floor(0), ceiling(6) { }
        CUDA4DNN_HOST_DEVICE Params(T floor_, T ceiling_) : floor(floor_), ceiling(ceiling_) { }
        T floor, ceiling;
    };

    CUDA4DNN_DEVICE ClippedReLUFunctor() : ClippedReLUFunctor(Params{}) { }
    CUDA4DNN_DEVICE ClippedReLUFunctor(const Params& params) : floor{params.floor}, ceiling{params.ceiling} { }

    CUDA4DNN_DEVICE T operator()(T value) {
        using csl::device::clamp;
        return clamp(value, floor, ceiling);
    }

    T floor, ceiling;
};

template <class T>
struct TanHFunctor {
    struct Params {
        CUDA4DNN_HOST_DEVICE Params() { }
    };

    CUDA4DNN_DEVICE TanHFunctor() { }
    CUDA4DNN_DEVICE TanHFunctor(const Params& params) { }

    CUDA4DNN_DEVICE T operator()(T value) {
        using csl::device::tanh;
        return tanh(value);
    }
};

template <class T>
struct SwishFunctor {
    struct Params {
        CUDA4DNN_HOST_DEVICE Params() { }
    };

    CUDA4DNN_DEVICE SwishFunctor() { }
    CUDA4DNN_DEVICE SwishFunctor(const Params& params) { }

    CUDA4DNN_DEVICE T operator()(T value) {
        // f(x) = x * sigmoid(x)
        using csl::device::fast_divide;
        using csl::device::fast_exp;
        return fast_divide(value, static_cast<T>(1) + fast_exp(-value));
    }
};

template <class T>
struct MishFunctor {
    struct Params {
        CUDA4DNN_HOST_DEVICE Params() { }
    };

    CUDA4DNN_DEVICE MishFunctor() { }
    CUDA4DNN_DEVICE MishFunctor(const Params& params) { }

    CUDA4DNN_DEVICE T operator()(T value) {
        using csl::device::tanh;
        using csl::device::log1pexp;
        return value * tanh(log1pexp(value));
    }
};

template <>
struct MishFunctor<float> {
    struct Params {
        CUDA4DNN_HOST_DEVICE Params() { }
    };

    CUDA4DNN_DEVICE MishFunctor() { }
    CUDA4DNN_DEVICE MishFunctor(const Params& params) { }

    CUDA4DNN_DEVICE float operator()(float value) {
        // f(x) = x * tanh(log1pexp(x));
        using csl::device::fast_divide;
        using csl::device::fast_exp;

        auto e = fast_exp(value);
        auto n = e * e + 2 * e;
        if (value <= -0.6f)
            return value * fast_divide(n, n + 2);
        return value - 2 * fast_divide(value, n + 2);
    }
};

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
template <>
struct MishFunctor<__half> {
    struct Params {
        CUDA4DNN_HOST_DEVICE Params() { }
    };

    CUDA4DNN_DEVICE MishFunctor() { }
    CUDA4DNN_DEVICE MishFunctor(const Params& params) { }

    CUDA4DNN_DEVICE __half operator()(__half value) {
        return MishFunctor<float>()(value);
    }
};
#endif

template <class T>
struct SigmoidFunctor {
    struct Params {
        CUDA4DNN_HOST_DEVICE Params() { }
    };

    CUDA4DNN_DEVICE SigmoidFunctor() { }
    CUDA4DNN_DEVICE SigmoidFunctor(const Params& params) { }

    CUDA4DNN_DEVICE T operator()(T value) {
        using csl::device::fast_sigmoid;
        return fast_sigmoid(value);
    }
};

template <class T>
struct ELUFunctor {
    struct Params {
        CUDA4DNN_HOST_DEVICE Params() { }
    };

    CUDA4DNN_DEVICE ELUFunctor() { }
    CUDA4DNN_DEVICE ELUFunctor(const Params& params) { }

    CUDA4DNN_DEVICE T operator()(T value) {
        using csl::device::expm1;
        return value >= T(0) ? value : expm1(value);
    }
};

template <class T>
struct AbsFunctor {
    struct Params { };

    CUDA4DNN_DEVICE AbsFunctor() { }
    CUDA4DNN_DEVICE AbsFunctor(const Params& params) { }

    CUDA4DNN_DEVICE T operator()(T value) {
        using csl::device::abs;
        return abs(value);
    }
};

template <class T>
struct BNLLFunctor {
    struct Params {
        CUDA4DNN_HOST_DEVICE Params() { }
    };

    CUDA4DNN_DEVICE BNLLFunctor() { }
    CUDA4DNN_DEVICE BNLLFunctor(const Params& params) { }

    CUDA4DNN_DEVICE T operator()(T value) {
        using csl::device::log1pexp;
        return value > T(0) ? value + log1pexp(-value) : log1pexp(value);
    }
};

template <class T>
struct CeilFunctor {
    struct Params {
        CUDA4DNN_HOST_DEVICE Params() { }
    };

    CUDA4DNN_DEVICE CeilFunctor() { }
    CUDA4DNN_DEVICE CeilFunctor(const Params& params) { }

    CUDA4DNN_DEVICE T operator()(T value) {
        using csl::device::ceil;
        return ceil(value);
    }
};

template <class T>
struct FloorFunctor {
    struct Params {
        CUDA4DNN_HOST_DEVICE Params() { }
    };

    CUDA4DNN_DEVICE FloorFunctor() { }
    CUDA4DNN_DEVICE FloorFunctor(const Params& params) { }

    CUDA4DNN_DEVICE T operator()(T value) {
        using csl::device::floor;
        return floor(value);
    }
};

template <class T>
struct LogFunctor {
    struct Params {
        CUDA4DNN_HOST_DEVICE Params() { }
    };

    CUDA4DNN_DEVICE LogFunctor() { }
    CUDA4DNN_DEVICE LogFunctor(const Params& params) { }

    CUDA4DNN_DEVICE T operator()(T value) {
        using csl::device::log;
        return log(value);
    }
};

template <class T>
struct RintFunctor {
    struct Params {
        CUDA4DNN_HOST_DEVICE Params() { }
    };

    CUDA4DNN_DEVICE RintFunctor() { }
    CUDA4DNN_DEVICE RintFunctor(const Params& params) { }

    CUDA4DNN_DEVICE T operator()(T value) {
        using csl::device::rint;
        return rint(value);
    }
};

template <class T>
struct SqrtFunctor {
    struct Params {
        CUDA4DNN_HOST_DEVICE Params() { }
    };

    CUDA4DNN_DEVICE SqrtFunctor() { }
    CUDA4DNN_DEVICE SqrtFunctor(const Params& params) { }

    CUDA4DNN_DEVICE T operator()(T value) {
        using csl::device::sqrt;
        return sqrt(value);
    }
};

template <class T>
struct NotFunctor {
    struct Params {
        CUDA4DNN_HOST_DEVICE Params() { }
    };

    CUDA4DNN_DEVICE NotFunctor() { }
    CUDA4DNN_DEVICE NotFunctor(const Params& params) { }

    CUDA4DNN_DEVICE T operator()(T value) {
        using csl::device::floor;
        return floor(static_cast<T>(1.) - value);
    }
};

template <class T>
struct PowerFunctor {
    struct Params {
        CUDA4DNN_HOST_DEVICE Params() : exp(1), scale(1), shift(0) { }
        CUDA4DNN_HOST_DEVICE Params(T exp_, T scale_, T shift_) : exp(exp_), scale(scale_), shift(shift_) { }
        T exp, scale, shift;
    };

    CUDA4DNN_DEVICE PowerFunctor() : PowerFunctor(Params{}) { }
    CUDA4DNN_DEVICE PowerFunctor(const Params& params) : exp{params.exp}, scale{params.scale}, shift{params.shift} { }

    CUDA4DNN_DEVICE T operator()(T value) {
        using csl::device::pow;
        return pow(shift + scale * value, exp);
    }

    T exp, scale, shift;
};

template <class T>
struct ExpFunctor {
    struct Params {
        CUDA4DNN_HOST_DEVICE Params() : normScale(1), normShift(0) { }
        CUDA4DNN_HOST_DEVICE Params(T nScale_, T nShift_) : normScale(nScale_), normShift(nShift_) { }
        T normScale, normShift;
    };

    CUDA4DNN_DEVICE ExpFunctor() : ExpFunctor(Params{}) { }
    CUDA4DNN_DEVICE ExpFunctor(const Params& params) : normScale{params.normScale}, normShift{params.normShift} { }

    CUDA4DNN_DEVICE T operator()(T value) {
        using csl::device::fast_exp;
        return fast_exp(normShift + normScale * value);
    }

    T normScale, normShift;
};

template <class T>
struct MaxFunctor {
    struct Params {
        CUDA4DNN_HOST_DEVICE Params() { }
    };

    CUDA4DNN_DEVICE MaxFunctor() { }
    CUDA4DNN_DEVICE MaxFunctor(const Params& params) { }

    CUDA4DNN_DEVICE T operator()(T x, T y) {
        using csl::device::max;
        return max(x, y);
    }
};

template <class T>
struct MinFunctor {
    struct Params {
        CUDA4DNN_HOST_DEVICE Params() { }
    };

    CUDA4DNN_DEVICE MinFunctor() { }
    CUDA4DNN_DEVICE MinFunctor(const Params& params) { }

    CUDA4DNN_DEVICE T operator()(T x, T y) {
        using csl::device::min;
        return min(x, y);
    }
};

template <class T>
struct SumFunctor {
    struct Params {
        CUDA4DNN_HOST_DEVICE Params() { }
    };

    CUDA4DNN_DEVICE SumFunctor() { }
    CUDA4DNN_DEVICE SumFunctor(const Params& params) { }

    CUDA4DNN_DEVICE T operator()(T x, T y) { return x + y; }
};

template <class T>
struct ScaledSumFunctor {
    struct Params {
        CUDA4DNN_HOST_DEVICE Params() : scale_x(1), scale_y(1) { }
        CUDA4DNN_HOST_DEVICE Params(T scale_x_, T scale_y_) : scale_x(scale_x_), scale_y(scale_y_) { }
        T scale_x, scale_y;
    };

    CUDA4DNN_DEVICE ScaledSumFunctor() : scale_x(1), scale_y(1) { }
    CUDA4DNN_DEVICE ScaledSumFunctor(const Params& params) : scale_x{params.scale_x}, scale_y{params.scale_y} { }

    CUDA4DNN_DEVICE T operator()(T x, T y) { return scale_x * x + scale_y * y; }

    T scale_x, scale_y;
};

template <class T>
struct ProductFunctor {
    struct Params {
        CUDA4DNN_HOST_DEVICE Params() { }
    };

    CUDA4DNN_DEVICE ProductFunctor() { }
    CUDA4DNN_DEVICE ProductFunctor(const Params& params) { }

    CUDA4DNN_DEVICE T operator()(T x, T y) { return x * y; }
};

template <class T>
struct DivFunctor {
    struct Params {
        CUDA4DNN_HOST_DEVICE Params() { }
    };

    CUDA4DNN_DEVICE DivFunctor() { }
    CUDA4DNN_DEVICE DivFunctor(const Params& params) { }

    CUDA4DNN_DEVICE T operator()(T x, T y) { return x / y; }
};

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA_FUNCTORS_HPP */
