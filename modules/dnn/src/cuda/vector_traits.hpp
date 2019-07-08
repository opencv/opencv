// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA_VECTOR_TYPE_TRAITS_HPP
#define OPENCV_DNN_SRC_CUDA_VECTOR_TYPE_TRAITS_HPP

#include <cuda_runtime.h>

#include <cstddef>
#include <type_traits>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl  { namespace kernels {
    /* HOW TO ADD A NEW VECTOR TYPE?
     * - specialize `get_vector_type`
     * - specialize `detail::get_element_type`
     */

    /** returns a vector type in the 'type' field for a given scalar type and vector size
     *
     * if a vector type does not exist for the given combination, the `type` member will not exist
     */
    template <class T, std::size_t N>
    struct get_vector_type {};

    template <> struct get_vector_type<float, 1> { typedef float type; };
    template <> struct get_vector_type<float, 2> { typedef float2 type; };
    template <> struct get_vector_type<float, 4> { typedef float4 type; };

    template <> struct get_vector_type<double, 1> { typedef double type; };
    template <> struct get_vector_type<double, 2> { typedef double2 type; };
    template <> struct get_vector_type<double, 4> { typedef double4 type; };

    namespace detail {
        template <class V, class = void /* SFINAE helper parameter */>
        struct get_element_type { };

        /* only non-const specializations are required; const qualifications are automatically handled */
        template <class X> struct get_element_type<float, X> { typedef float type; };
        template <class X> struct get_element_type<float2, X> { typedef float type; };
        template <class X> struct get_element_type<float4, X> { typedef float type; };

        template <class X> struct get_element_type<double, X> { typedef double type; };
        template <class X> struct get_element_type<double2, X> { typedef double type; };
        template <class X> struct get_element_type<double4, X> { typedef double type; };

        /* handle const qualified types */
        template <class V>
        struct get_element_type<V, typename std::enable_if<std::is_const<V>::value, void>::type>{
            typedef
                typename std::add_const<
                    typename get_element_type<
                        typename std::remove_const<V>::type
                    >::type
                >::type
            type;
        };
    }

    namespace detail {
        template <class V> __host__ __device__
        constexpr std::size_t size() { return sizeof(V) / sizeof(typename get_element_type<V>::type); }
    }

    /** returns a struct with information about a given vector or scalar type
     *
     * - `element_type` gives the scalar type corresponding to the type
     * - `vector_type` gives the type
     * - `size()` returns the number of elements of `element_type` that can exist in the type
     */
    template <class V>
    struct vector_traits {
        typedef typename detail::get_element_type<V>::type element_type;
        typedef V vector_type;

        __host__ __device__
        static constexpr std::size_t size() { return detail::size<V>(); }
    };

    namespace detail {
        template <class V, std::size_t>
        struct accessor { };

        template <class V>
        struct accessor<V, 1> {
            __host__ __device__
            static constexpr typename vector_traits<V>::element_type get(V& v, std::size_t i) { return v; }

            __host__ __device__
            static constexpr void set(V& v, std::size_t i, vector_traits<V>::element_type value) { v = value; }
        };

        template <class V>
        struct accessor<V, 2> {
            __host__ __device__
            static constexpr typename vector_traits<V>::element_type get(V& v, std::size_t i) {
                switch (i) {
                case 0: return v.x;
                case 1: return v.y;
                }
                /* should never end up here */
                return v.x;
            }

            __host__ __device__
            static constexpr void set(V& v, std::size_t i, vector_traits<V>::element_type value) {
                switch (i) {
                case 0: v.x = value;
                case 1: v.y = value;
                }
                /* should never end up here */
            }
        };

        template <class V>
        struct accessor<V, 4> {
            __host__ __device__
            static constexpr typename vector_traits<V>::element_type get(V& v, std::size_t i) {
                switch (i) {
                case 0: return v.w;
                case 1: return v.x;
                case 2: return v.y;
                case 3: return v.z;
                }
                /* should never end up here */
                return v.x;
            }

            __host__ __device__
            static constexpr void set(V& v, std::size_t i, vector_traits<V>::element_type value) {
                switch (i) {
                case 0: v.w = value;
                case 1: v.x = value;
                case 2: v.y = value;
                case 3: v.z = value;
                }
                /* should never end up here */
            }
        };
    }

    /** get a value from a vector type using an index */
    template <class V> __host__ __device__
    constexpr typename vector_traits<V>::element_type get(V& v, std::size_t i) {
        return detail::accessor<V, vector_traits<V>::size()>::get(v, i);
    }

    /** set a value in a vector type using an index */
    template <class V> __host__ __device__
    constexpr void set(V& v, std::size_t i, typename vector_traits<V>::element_type value) {
        return detail::accessor<V, vector_traits<V>::size()>::set(v, i, value);
    }


}}}}} /*  cv::dnn::cuda4dnn::csl::kernels */

#endif /* OPENCV_DNN_SRC_CUDA_VECTOR_TYPE_TRAITS_HPP */
