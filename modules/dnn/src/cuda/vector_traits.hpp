// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA_VECTOR_TYPE_TRAITS_HPP
#define OPENCV_DNN_SRC_CUDA_VECTOR_TYPE_TRAITS_HPP

#include <cuda_runtime.h>
#include <cuda_fp16.hpp>

#include <cstddef>
#include <type_traits>

#include "../cuda4dnn/csl/pointer.hpp"

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl  { namespace kernels {

    namespace detail {
        template <std::size_t N> struct raw_type_ { };
        template <> struct raw_type_<256> { typedef ulonglong4 type; };
        template <> struct raw_type_<128> { typedef uint4 type; };
        template <> struct raw_type_<64> { typedef uint2 type; };
        template <> struct raw_type_<32> { typedef uint1 type; };
        template <> struct raw_type_<16> { typedef uchar2 type; };
        template <> struct raw_type_<8> { typedef uchar1 type; };

        template <std::size_t N> struct raw_type {
            using type = typename raw_type_<N>::type;
            static_assert(sizeof(type) * 8 == N, "");
        };
    }

    template <class T, std::size_t N>
    union vector_type {
        using value_type = T;
        using raw_type = typename detail::raw_type<N * sizeof(T) * 8>::type;

        __device__ vector_type() { }

        __device__ static constexpr auto size() { return N; }

        raw_type raw;
        T data[N];

        template <class U> static __device__
        typename std::enable_if<std::is_const<U>::value, const vector_type*>
        ::type get_pointer(DevicePtr<U> ptr) {
            return reinterpret_cast<const vector_type*>(ptr.get());
        }

        template <class U> static __device__
        typename std::enable_if<!std::is_const<U>::value, vector_type*>
        ::type get_pointer(DevicePtr<U> ptr) {
            return reinterpret_cast<vector_type*>(ptr.get());
        }
    };

    template <class V>
    __device__ void v_load(V& dest, const V& src) {
        dest.raw = src.raw;
    }

    template <class V>
    __device__ void v_load(V& dest, const V* src) {
        dest.raw = src->raw;
    }

    template <class V>
    __device__ void v_store(V* dest, const V& src) {
        dest->raw = src.raw;
    }

    template <class V>
    __device__ void v_store(V& dest, const V& src) {
        dest.raw = src.raw;
    }

    template <class T, std::size_t N>
    struct get_vector_type {
        typedef vector_type<T, N> type;
    };

}}}}} /*  cv::dnn::cuda4dnn::csl::kernels */

#endif /* OPENCV_DNN_SRC_CUDA_VECTOR_TYPE_TRAITS_HPP */
