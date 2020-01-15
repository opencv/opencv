// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_CSL_FP16_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_CSL_FP16_HPP

#include "nvcc_defs.hpp"

#include <cuda_fp16.h>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl {

    namespace detail {
        template <class T, class = void>
        struct is_half_convertible : std::false_type { };

        template <class T>
        struct is_half_convertible<T, typename std::enable_if<std::is_integral<T>::value, void>::type> : std::true_type { };

        template <class T>
        struct is_half_convertible<T, typename std::enable_if<std::is_floating_point<T>::value, void>::type> : std::true_type { };
    }

    /* Note: nvcc has a broken overload resolution; it considers host overloads inside device code
    CUDA4DNN_HOST bool operator==(half lhs, half rhs) noexcept { return static_cast<float>(lhs) == static_cast<float>(rhs); }
    CUDA4DNN_HOST bool operator!=(half lhs, half rhs) noexcept { return static_cast<float>(lhs) != static_cast<float>(rhs); }
    CUDA4DNN_HOST bool operator<(half lhs, half rhs) noexcept { return static_cast<float>(lhs) < static_cast<float>(rhs); }
    CUDA4DNN_HOST bool operator>(half lhs, half rhs) noexcept { return static_cast<float>(lhs) > static_cast<float>(rhs); }
    CUDA4DNN_HOST bool operator<=(half lhs, half rhs) noexcept { return static_cast<float>(lhs) <= static_cast<float>(rhs); }
    CUDA4DNN_HOST bool operator>=(half lhs, half rhs) noexcept { return static_cast<float>(lhs) >= static_cast<float>(rhs); }
    */

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template <class T> CUDA4DNN_HOST
    typename std::enable_if<detail::is_half_convertible<T>::value, bool>
    ::type operator==(half lhs, T rhs) noexcept { return static_cast<float>(lhs) == static_cast<float>(rhs); }

    template <class T> CUDA4DNN_HOST
    typename std::enable_if<detail::is_half_convertible<T>::value, bool>
    ::type operator!=(half lhs, T rhs) noexcept { return static_cast<float>(lhs) != static_cast<float>(rhs); }

    template <class T> CUDA4DNN_HOST
    typename std::enable_if<detail::is_half_convertible<T>::value, bool>
    ::type operator<(half lhs, T rhs) noexcept { return static_cast<float>(lhs) < static_cast<float>(rhs); }

    template <class T> CUDA4DNN_HOST
    typename std::enable_if<detail::is_half_convertible<T>::value, bool>
    ::type operator>(half lhs, T rhs) noexcept { return static_cast<float>(lhs) > static_cast<float>(rhs); }

    template <class T> CUDA4DNN_HOST
    typename std::enable_if<detail::is_half_convertible<T>::value, bool>
    ::type operator<=(half lhs, T rhs) noexcept { return static_cast<float>(lhs) <= static_cast<float>(rhs); }

    template <class T> CUDA4DNN_HOST
    typename std::enable_if<detail::is_half_convertible<T>::value, bool>
    ::type operator>=(half lhs, T rhs) noexcept { return static_cast<float>(lhs) >= static_cast<float>(rhs); }

    template <class T> CUDA4DNN_HOST
    typename std::enable_if<detail::is_half_convertible<T>::value, bool>
    ::type operator==(T lhs, half rhs) noexcept { return static_cast<float>(lhs) == static_cast<float>(rhs); }

    template <class T> CUDA4DNN_HOST
    typename std::enable_if<detail::is_half_convertible<T>::value, bool>
    ::type operator!=(T lhs, half rhs) noexcept { return static_cast<float>(lhs) != static_cast<float>(rhs); }

    template <class T> CUDA4DNN_HOST
    typename std::enable_if<detail::is_half_convertible<T>::value, bool>
    ::type operator<(T lhs, half rhs) noexcept { return static_cast<float>(lhs) < static_cast<float>(rhs); }

    template <class T> CUDA4DNN_HOST
    typename std::enable_if<detail::is_half_convertible<T>::value, bool>
    ::type operator>(T lhs, half rhs) noexcept { return static_cast<float>(lhs) > static_cast<float>(rhs); }

    template <class T> CUDA4DNN_HOST
    typename std::enable_if<detail::is_half_convertible<T>::value, bool>
    ::type operator<=(T lhs, half rhs) noexcept { return static_cast<float>(lhs) <= static_cast<float>(rhs); }

    template <class T> CUDA4DNN_HOST
    typename std::enable_if<detail::is_half_convertible<T>::value, bool>
    ::type operator>=(T lhs, half rhs) noexcept { return static_cast<float>(lhs) >= static_cast<float>(rhs); }
#endif

}}}} /* namespace cv::dnn::cuda4dnn::csl */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_CSL_FP16_HPP */
