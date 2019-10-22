// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA_ARRAY_HPP
#define OPENCV_DNN_SRC_CUDA_ARRAY_HPP

#include <cuda_runtime.h>

#include "types.hpp"

#include <cstddef>
#include <type_traits>
#include <iterator>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace device {

    template <class T, std::size_t N>
    struct array {
        using value_type        = T;
        using size_type         = device::size_type;
        using difference_type   = std::ptrdiff_t;
        using reference         = typename std::add_lvalue_reference<value_type>::type;
        using const_reference   = typename std::add_lvalue_reference<typename std::add_const<value_type>::type>::type;
        using pointer           = typename std::add_pointer<value_type>::type;
        using const_pointer     = typename std::add_pointer<typename std::add_const<value_type>::type>::type;
        using iterator          = pointer;
        using const_iterator    = const_pointer;
        using reverse_iterator  = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        __host__ __device__ bool empty() const noexcept { return N == 0; }
        __host__ __device__ size_type size() const noexcept { return N; }

        __host__ __device__ iterator begin() noexcept { return ptr; }
        __host__ __device__ iterator end() noexcept { return ptr + N; }
        __host__ __device__ const_iterator begin() const noexcept { return ptr; }
        __host__ __device__ const_iterator end() const noexcept { return ptr + N; }

        __host__ __device__ const_iterator cbegin() const noexcept { return ptr; }
        __host__ __device__ const_iterator cend() const noexcept { return ptr + N; }

        __host__ __device__ reverse_iterator rbegin() noexcept { return ptr + N; }
        __host__ __device__ reverse_iterator rend() noexcept { return ptr; }
        __host__ __device__ const_reverse_iterator rbegin() const noexcept { return ptr + N; }
        __host__ __device__ const_reverse_iterator rend() const noexcept { return ptr; }

        __host__ __device__ const_reverse_iterator crbegin() const noexcept { return ptr + N; }
        __host__ __device__ const_reverse_iterator crend() const noexcept { return ptr; }

        template <class InputItr>
        __host__ void assign(InputItr first, InputItr last) {
            std::copy(first, last, std::begin(ptr));
        }

        __host__ __device__ reference operator[](int idx) { return ptr[idx]; }
        __host__ __device__ const_reference operator[](int idx) const { return ptr[idx]; }

        __host__ __device__ reference front() { return ptr[0]; }
        __host__ __device__ const_reference front() const { return ptr[0]; }

        __host__ __device__ reference back() { return ptr[N - 1]; }
        __host__ __device__ const_reference back() const { return ptr[N - 1]; }

        __host__ __device__ pointer data() noexcept { return ptr; }
        __host__ __device__ const_pointer data() const noexcept { return ptr; }

        T ptr[N];
    };

}}}}} /* namespace cv::dnn::cuda4dnn::csl::device */

#endif /* OPENCV_DNN_SRC_CUDA_ARRAY_HPP */
