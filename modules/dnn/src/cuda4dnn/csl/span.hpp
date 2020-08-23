// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_CSL_SPAN_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_CSL_SPAN_HPP

#include "pointer.hpp"
#include "nvcc_defs.hpp"

#include "../../cuda/types.hpp"

#include <cstddef>
#include <type_traits>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl {

    /** @brief provides non-owning mutable access for device arrays
     *
     *  const Span<T>/Span<T> provides mutable access to the elements unless T is const qualified
     *  const Span<T> makes the span immutable but not the elements
     */
    template <class T>
    class Span {
        static_assert(std::is_standard_layout<T>::value, "T must satisfy StandardLayoutType");

    public:
        using value_type = T;
        using size_type = device::size_type;
        using index_type = device::index_type;

        using pointer = DevicePtr<value_type>;
        using const_pointer = DevicePtr<typename std::add_const<value_type>::type>;
        using reference = typename std::add_lvalue_reference<value_type>::type;
        using const_reference = typename std::add_lvalue_reference<typename std::add_const<value_type>::type>;

        Span() noexcept : ptr{ nullptr }, sz{ 0 } { }
        CUDA4DNN_HOST_DEVICE Span(pointer first, pointer last) noexcept : ptr{ first }, sz{ last - first } { }
        CUDA4DNN_HOST_DEVICE Span(pointer first, size_type count) noexcept : ptr{ first }, sz{ count } { }

        CUDA4DNN_HOST_DEVICE size_type size() const noexcept { return sz; }
        CUDA4DNN_HOST_DEVICE bool empty() const noexcept { return size() == 0; }

        CUDA4DNN_DEVICE reference operator[](index_type index) const { return ptr[index]; }
        CUDA4DNN_HOST_DEVICE pointer data() const noexcept { return ptr; }

        template<class U = T, class V = typename std::add_const<U>::type,
            typename std::enable_if<!std::is_const<U>::value, bool>::type = true>
            CUDA4DNN_HOST_DEVICE operator Span<V>() const noexcept { return Span<V>{ptr, sz}; }

    private:
        pointer ptr;
        size_type sz;
    };

    /** @brief provides non-owning immutable view for device arrays */
    template <class T>
    using View = Span<const T>;

    /** returns true if the address of a span/view is aligned to \p alignment number of elements (not bytes) */
    template <class T>
    bool is_address_aligned(View<T> v, std::size_t alignment) {
        return is_aligned(v.data(), alignment * sizeof(T));
    }

    /** returns true if the size of a span/view is a multiple of \p alignment */
    template <class T>
    bool is_size_aligned(View<T> v, std::size_t alignment) {
        return v.size() % alignment == 0;
    }

    /** @brief returns true if the address and the size of the span/view is aligned
     * \p alignment refers to the number of elements (not bytes)
     */
    template <class T>
    bool is_fully_aligned(View<T> v, std::size_t alignment) {
        return is_address_aligned(v, alignment) && is_size_aligned(v, alignment);
    }

}}}} /* namespace cv::dnn::cuda4dnn::csl */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_CSL_SPAN_HPP */
