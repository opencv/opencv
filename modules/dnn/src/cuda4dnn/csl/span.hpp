// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_SPAN_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_SPAN_HPP

#include "pointer.hpp"
#include "nvcc_defs.hpp"

#include <cstddef>
#include <type_traits>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl {

    /** @brief provides non-owning mutable view for device arrays
     *
     *  const span<T>/span<T> provides mutable access to the elements unless T is const qualified
     *  const span<T> makes the span immutable but not the elements
     */
    template <class T>
    class span {
        static_assert(std::is_standard_layout<T>::value, "T must satisfy StandardLayoutType");

    public:
        using value_type = T;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        using pointer = DevicePtr<value_type>;
        using const_pointer = DevicePtr<typename std::add_const<value_type>::type>;
        using reference = typename std::add_lvalue_reference<value_type>::type;
        using const_reference = typename std::add_lvalue_reference<typename std::add_const<value_type>::type>;

        using iterator = pointer;
        using const_iterator = const_pointer;

        span() noexcept : ptr{ nullptr }, sz{ 0 } { }
        CUDA4DNN_HOST_DEVICE span(pointer first, pointer last) noexcept : ptr{ first }, sz{ last - first } { }
        CUDA4DNN_HOST_DEVICE span(pointer first, size_type count) noexcept : ptr{ first }, sz{ count } { }

        CUDA4DNN_HOST_DEVICE size_type size() const noexcept { return sz; }
        CUDA4DNN_HOST_DEVICE bool empty() const noexcept { return size() == 0; }

        CUDA4DNN_DEVICE reference operator[](difference_type index) const { return ptr[index]; }
        CUDA4DNN_HOST_DEVICE pointer data() const noexcept { return ptr; }

    private:
        pointer ptr;
        size_type sz;
    };

    /** @brief provides non-owning immutable view for device arrays */
    template <class T>
    using view = span<const T>;

}}}} /* namespace cv::dnn::cuda4dnn::csl */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_SPAN_HPP */
