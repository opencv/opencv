// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA_VECTOR_TRAITS_HPP
#define OPENCV_DNN_SRC_CUDA_VECTOR_TRAITS_HPP

#include <cuda_runtime.h>

#include "types.hpp"

#include "../cuda4dnn/csl/pointer.hpp"

#include <type_traits>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace device {

    /** \file vector_traits.hpp
     *  \brief utility classes and functions for vectorized memory loads/stores
     *
     * Example:
     * using vector_type = get_vector_type_t<float, 4>;
     *
     * auto input_vPtr = type::get_pointer(iptr); // iptr is of type DevicePtr<const float>
     * auto output_vPtr = type::get_pointer(optr);  // optr is of type DevicePtr<float>
     *
     * vector_type vec;
     * v_load(vec, input_vPtr);
     *
     * for(int i = 0; i < vector_type::size(); i++)
     *      vec[i] = do_something(vec[i]);
     *
     * v_store(output_vPtr, vec);
     */

    namespace detail {
        template <size_type N> struct raw_type_ { };
        template <> struct raw_type_<256> { typedef ulonglong4 type; };
        template <> struct raw_type_<128> { typedef uint4 type; };
        template <> struct raw_type_<64> { typedef uint2 type; };
        template <> struct raw_type_<32> { typedef uint1 type; };
        template <> struct raw_type_<16> { typedef uchar2 type; };
        template <> struct raw_type_<8> { typedef uchar1 type; };

        template <size_type N> struct raw_type {
            using type = typename raw_type_<N>::type;
            static_assert(sizeof(type) * 8 == N, "");
        };
    }

    /* \tparam T    type of element in the vector
     * \tparam N    "number of elements" of type T in the vector
     */
    template <class T, size_type N>
    union vector_type {
        using value_type = T;
        using raw_type = typename detail::raw_type<N * sizeof(T) * 8>::type;

        __device__ vector_type() { }

        __device__ static constexpr size_type size() { return N; }

        raw_type raw;
        T data[N];

        template <class U> static __device__
        typename std::enable_if<std::is_const<U>::value, const vector_type*>
        ::type get_pointer(csl::DevicePtr<U> ptr) {
            return reinterpret_cast<const vector_type*>(ptr.get());
        }

        template <class U> static __device__
        typename std::enable_if<!std::is_const<U>::value, vector_type*>
        ::type get_pointer(csl::DevicePtr<U> ptr) {
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

    template <class T, size_type N>
    struct get_vector_type {
        typedef vector_type<T, N> type;
    };

    template <class T, size_type N>
    using get_vector_type_t = typename get_vector_type<T, N>::type;

}}}}} /* namespace cv::dnn::cuda4dnn::csl::device */

#endif /* OPENCV_DNN_SRC_CUDA_VECTOR_TRAITS_HPP */
