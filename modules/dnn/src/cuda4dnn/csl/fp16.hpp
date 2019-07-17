// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_FP16_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_FP16_HPP

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

    template <class T>
    typename std::enable_if<detail::is_half_convertible<T>::value, bool>
    ::type operator==(half lhs, T rhs) noexcept { return static_cast<float>(lhs) == static_cast<float>(rhs); }

    template <class T>
    typename std::enable_if<detail::is_half_convertible<T>::value, bool>
    ::type operator!=(half lhs, T rhs) noexcept { return static_cast<float>(lhs) != static_cast<float>(rhs); }

    template <class T>
    typename std::enable_if<detail::is_half_convertible<T>::value, bool>
    ::type operator<(half lhs, T rhs) noexcept { return static_cast<float>(lhs) < static_cast<float>(rhs); }

    template <class T>
    typename std::enable_if<detail::is_half_convertible<T>::value, bool>
    ::type operator>(half lhs, T rhs) noexcept { return static_cast<float>(lhs) > static_cast<float>(rhs); }

    template <class T>
    typename std::enable_if<detail::is_half_convertible<T>::value, bool>
    ::type operator<=(half lhs, T rhs) noexcept { return static_cast<float>(lhs) <= static_cast<float>(rhs); }

    template <class T>
    typename std::enable_if<detail::is_half_convertible<T>::value, bool>
    ::type operator>=(half lhs, T rhs) noexcept { return static_cast<float>(lhs) >= static_cast<float>(rhs); }

    template <class T>
    typename std::enable_if<detail::is_half_convertible<T>::value, bool>
    ::type operator==(T lhs, half rhs) noexcept { return static_cast<float>(lhs) == static_cast<float>(rhs); }

    template <class T>
    typename std::enable_if<detail::is_half_convertible<T>::value, bool>
    ::type operator!=(T lhs, half rhs) noexcept { return static_cast<float>(lhs) != static_cast<float>(rhs); }

    template <class T>
    typename std::enable_if<detail::is_half_convertible<T>::value, bool>
    ::type operator<(T lhs, half rhs) noexcept { return static_cast<float>(lhs) < static_cast<float>(rhs); }

    template <class T>
    typename std::enable_if<detail::is_half_convertible<T>::value, bool>
    ::type operator>(T lhs, half rhs) noexcept { return static_cast<float>(lhs) > static_cast<float>(rhs); }

    template <class T>
    typename std::enable_if<detail::is_half_convertible<T>::value, bool>
    ::type operator<=(T lhs, half rhs) noexcept { return static_cast<float>(lhs) <= static_cast<float>(rhs); }

    template <class T>
    typename std::enable_if<detail::is_half_convertible<T>::value, bool>
    ::type operator>=(T lhs, half rhs) noexcept { return static_cast<float>(lhs) >= static_cast<float>(rhs); }

}}}} /* namespace cv::dnn::cuda4dnn::csl */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_FP16_HPP */
