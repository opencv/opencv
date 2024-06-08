// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_CXX_UTILS_RESIZABLE_STATIC_ARRAY_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_CXX_UTILS_RESIZABLE_STATIC_ARRAY_HPP

#include <cstddef>
#include <array>
#include <cassert>
#include <algorithm>

namespace cv { namespace dnn { namespace cuda4dnn { namespace cxx_utils {

    template <class T, std::size_t maxN>
    class resizable_static_array {
        using container_type = std::array<T, maxN>;

    public:
        using value_type                = typename container_type::value_type;
        using size_type                 = typename container_type::size_type;
        using difference_type           = typename container_type::difference_type;
        using reference                 = typename container_type::reference;
        using const_reference           = typename container_type::const_reference;
        using pointer                   = typename container_type::pointer;
        using const_pointer             = typename container_type::const_pointer;
        using iterator                  = typename container_type::iterator;
        using const_iterator            = typename container_type::const_iterator;
        using reverse_iterator          = typename container_type::reverse_iterator;
        using const_reverse_iterator    = typename container_type::const_reverse_iterator;

        resizable_static_array() noexcept : size_{ 0 } { }
        explicit resizable_static_array(size_type sz) noexcept : size_{ sz } { }

        bool empty() const noexcept { return static_cast<bool>(size_); }
        size_type size() const noexcept { return size_; }
        size_type capacity() const noexcept { return maxN; }

        void resize(size_type sz) noexcept {
            assert(sz <= capacity());
            size_ = sz;
        }

        void clear() noexcept { size_ = 0; }

        template <class ForwardItr>
        void assign(ForwardItr first, ForwardItr last) {
            resize(std::distance(first, last));
            std::copy(first, last, begin());
        }

        iterator begin() noexcept { return std::begin(arr); }
        iterator end() noexcept { return std::begin(arr) + size(); }

        const_iterator begin() const noexcept { return arr.cbegin(); }
        const_iterator end() const noexcept { return arr.cbegin() + size(); }

        const_iterator cbegin() const noexcept { return arr.cbegin(); }
        const_iterator cend() const noexcept { return arr.cbegin() + size(); }

        reverse_iterator rbegin() noexcept { return std::begin(arr) + size(); }
        reverse_iterator rend() noexcept { return std::begin(arr); }

        const_reverse_iterator rbegin() const noexcept { return arr.cbegin()+ size(); }
        const_reverse_iterator rend() const noexcept { return arr.cbegin(); }

        const_reverse_iterator crbegin() const noexcept { return arr.cbegin() + size(); }
        const_reverse_iterator crend() const noexcept { return arr.cbegin(); }

        reference operator[](size_type pos) {
            assert(pos < size());
            return arr[pos];
        }

        const_reference operator[](size_type pos) const {
            assert(pos < size());
            return arr[pos];
        }

        iterator insert(iterator pos, const T& value) {
            resize(size() + 1);
            std::move_backward(pos, end() - 1, end());
            *pos = value;
            return pos;
        }

        iterator insert(iterator pos, T&& value) {
            resize(size() + 1);
            std::move_backward(pos, end() - 1, end());
            *pos = std::move(value);
            return pos;
        }

        iterator erase(iterator pos) {
            std::move(pos + 1, end(), pos);
            resize(size() - 1);
            return pos;
        }

        pointer data() noexcept { return arr.data(); }
        const_pointer data() const noexcept { return arr.data(); }

    private:
        std::size_t size_;
        container_type arr;
    };

}}}} /* namespace cv::dnn::cuda4dnn::csl::cxx_utils */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_CXX_UTILS_RESIZABLE_STATIC_ARRAY_HPP */
