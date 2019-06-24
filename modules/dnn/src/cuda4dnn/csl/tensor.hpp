// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_TENSOR_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_TENSOR_HPP

#include "nvcc_defs.hpp"
#include "memory.hpp"
#include "cublas.hpp"
#include "cudnn.hpp"
#include "span.hpp"
#include "kernels.hpp"
#include "workspace.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <array>
#include <functional>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <vector>
#include <utility>

#ifndef CSL_DEFAULT_TENSOR_RANK
    #define CSL_DEFAULT_TENSOR_RANK 4
#endif

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl {

    /** \file tensor.hpp
     *
     * The tensor library contains three kinds of tensor objects which are summarized
     * in the table below:
     *
     *     TYPE     | OWNERSHIP | MUTABLE | PASS TO KERNELS
     * ------------ + --------- + ------- + ---------------
     *    Tensor    |    Yes    |   Yes   |       No
     *  TensorSpan  |    No     |   Yes   |      Yes
     *  TensorView  |    No     |   No    |      Yes
     *
     * Tensor is implicitly convertible to TensorSpan and TensorView
     * TensorSpan is implicitly convertible to TensorView
     *
     * "TensorType", frequently used as a template parameter, can refer to Tensor, TensorSpan or TensorView.
     */

    /** @brief multi-dimensional contiguous GPU tensor containing elements of a single type
     *
     * \tparam  T       type of data stored by the tensor
     * \tparam  rank_   rank of the tensor
     */
    template <class T, std::size_t rank_ = CSL_DEFAULT_TENSOR_RANK>
    class Tensor {
        static_assert(rank_ > 0, "Scalars are not supported");
        static_assert(std::is_standard_layout<T>::value, "T must staisfy StandardLayoutType");

    public:
        using value_type    = typename ManagedPtr<T>::element_type;
        using pointer       = typename ManagedPtr<value_type>::pointer;
        using const_pointer = typename ManagedPtr<value_type>::const_pointer;
        using size_type     = std::size_t;

        static constexpr auto rank = rank_;

        Tensor() noexcept { std::fill(std::begin(sizes), std::end(sizes), 0); }
        Tensor(const Tensor&) = delete;
        Tensor(Tensor&& other) noexcept {
            data = std::move(other.data);
            sizes = other.sizes;
            std::fill(std::begin(other.sizes), std::end(other.sizes), 0);
        }

        /** @brief constructs a tensor of specific size
         *
         * Whatever arguments are accepted by the resize methods are accepted here.
         */
        template <class ...Args>
        Tensor(Args... sizes) { resize(std::forward<Args>(sizes)...); }

        Tensor& operator=(const Tensor&) = delete;
        Tensor& operator=(Tensor&& other) noexcept {
            data = std::move(other.data);
            sizes = other.sizes;
            std::fill(std::begin(other.sizes), std::end(other.sizes), 0);
            return *this;
        }

        /** returns the total number of elements in the tensor */
        size_type size() const noexcept {
            return std::accumulate(std::begin(sizes), std::end(sizes), 1, std::multiplies<size_type>());
        }

        /** returns a shape array consisting of axis lengths in order starting from zero */
        std::array<size_type, rank> shape() const noexcept { return sizes; }

        /** returns true if the tensor is empty */
        bool empty() const noexcept { return !size(); }

        /** @brief returns the length of the axis
         *
         * Every axis is assigned a zero-based index which can be used to select an axis.
         * Negative index can be used to select an axis from the end.
         *
         * Examples:
         * > -1 represents the last axis
         * > 0 represents the first axis
         * > 1 represents the second axis
         *
         * Pre-conditions:
         * - the axis must be in the range [-rank, rank)
         */
        size_type get_axis_size(int axis) const noexcept {
            axis = clamp_axis(axis, rank);
            CV_Assert(axis >= 0 && axis < rank);
            return sizes[axis];
        }

        /** returns a device pointer to mutable device memory */
        pointer get() noexcept { return data.get(); }

        /** returns a device pointer to immutable device memory */
        const_pointer get() const noexcept { return data.get(); }

        /** @brief resizes the tensor
         *
         * Pre-conditions:
         * - [start, end) represents a range containing length of the axes in order starting from axis zero
         * - number of sizes provided must be less than or equal to the tensor rank
         * - the sizes must be positive integers
         *
         * The length of unspecified axes will be assumed to be one.
         *
         * Exception Guarantee: Strong
         */
        template <class ForwardItr>
        typename std::enable_if<!std::is_integral<ForwardItr>::value, void> // TODO is_iterator
        ::type resize(ForwardItr start, ForwardItr end) {
            CV_Assert(start != end);
            CV_Assert(std::distance(start, end) <= rank);

            using ItrValueType = typename std::iterator_traits<ForwardItr>::value_type;
            auto total = std::accumulate(start, end, 1, std::multiplies<ItrValueType>());
            data.reset(total);

            /* length of the unspecified axes are assumed to be one */
            std::fill(std::begin(sizes), std::end(sizes), 1);
            std::copy_backward(start, end, std::end(sizes));
        }

        /** @brief resizes the tensor
         * constructs a range out of the arguments and invokes range-based resize method
         */
        template <class ...Sizes>
        void resize(Sizes... new_sizes_) {
            static_assert(sizeof...(Sizes) <= rank, "number of axes exceeds the tensor rank");
            std::array<size_type, sizeof...(Sizes)> new_sizes = { static_cast<size_type>(new_sizes_)... };
            resize(std::begin(new_sizes), std::end(new_sizes));
        }

        /** @brief resizes the tensor
         *
         * Pre-conditions:
         * - the reference tensor must be a non-empty tensor
         * - the reference tensor's rank must be lesser than or equal to the rank of target tensor
         *
         * Exception Guarantee: Strong
         */
        template <class TensorType>
        void resize_as(const TensorType& tensor) {
            static_assert(TensorType::rank <= rank, "cannot resize a tensor of lower rank to a tensor of higher rank");
            std::array<size_type, TensorType::rank> new_sizes;
            for (int i = 0; i < TensorType::rank; i++)
                new_sizes[i] = tensor.get_axis_size(i);
            resize(std::begin(new_sizes), std::end(new_sizes));
        }

        /** @brief reshapes the tensor
         *
         * Length deduction:
         * The length of at most one axis can be deduced using the total size constraint. The axis can
         * be marked for deduction by specifying the size as -1.
         *
         * The axes for which no size was provided (excluding -1) will be assumed to be one.
         *
         * Pre-conditions:
         * - [start, end) represents a range containing length of the axes in order starting from axis zero
         * - the number of lengths provided must be less than or equal to the tensor rank
         * - at most one axis length is allowed for length deduction
         * - the lengths provided must ensure that the total number of elements remains unchanged
         *
         * Exception Guarantee: Strong
         */
        template <class ForwardItr>
        typename std::enable_if<!std::is_integral<ForwardItr>::value, void> // TODO is_iterator
        ::type reshape(ForwardItr start, ForwardItr end) {
            CV_Assert(start != end);
            CV_Assert(std::distance(start, end) <= rank);

            using ItrValueType = typename std::iterator_traits<ForwardItr>::value_type;

            /* the user may leave at most one axis size for deduction by specifying -1 */
            auto sizes_to_deduce = std::count(start, end, -1);
            if (sizes_to_deduce > 1) { CV_Error(Error::StsBadArg, "only one axis size can be deduced"); }

            /* sizes must be positive numbers with the exception of -1 */
            auto invalid_sizes = std::count_if(start, end, [](ItrValueType x) {
                return !(x > 0 || x == -1);
            });
            if (invalid_sizes) { CV_Error(Error::StsBadArg, "invalid axis size"); }

            /* compute the total number of elements in the new tensor */
            size_type unknown_size = 0;
            auto total = std::accumulate(start, end, 1, std::multiplies<ItrValueType>());
            if (total < 0) {
                /* there is an unknown size */
                if (std::abs(total) <= size()) {
                    unknown_size = size() / std::abs(total);
                    total = size();
                }
                /* Edge case: if `total` is already more than size(), skip the deduction as it's impossible
                ** Since `total` is negative, the size check which follows will fail and throw an error
                */
            }

            /* the number of elements before and after reshape must be exactly same */
            if (total != size()) {
                CV_Error(Error::StsBadArg, "new axes do not preserve the tensor element count");
            }

            /* we assume the size of the unspecified axes to be one */
            std::fill(std::begin(sizes), std::end(sizes), 1);
            std::copy_backward(start, end, std::end(sizes));

            /* replace the unknown axis with the correct value */
            std::replace(std::begin(sizes), std::end(sizes), size_type(-1), unknown_size);
        }

        /** @brief reshapes the tensor
         * constructs a range out of the arguments and invokes range-based reshape method
         */
        template <class ...Sizes>
        void reshape(Sizes... new_sizes_) {
            static_assert(sizeof...(Sizes) <= rank, "number of axes exceeds the tensor rank");
            std::array<std::int64_t, sizeof...(Sizes)> new_sizes = { static_cast<std::int64_t>(new_sizes_)... };
            reshape(std::begin(new_sizes), std::end(new_sizes));
        }

        /** @brief reshapes the tensor
         *
         * Pre-conditions:
         * - the reference tensor must be a non-empty tensor
         * - the reference tensor's rank must be lesser than or equal to the rank of target tensor
         *
         * Exception Guarantee: Strong
         */
        template <class TensorType>
        void reshape_as(const TensorType& tensor) {
            static_assert(TensorType::rank <= rank, "cannot reshape a tensor of lower rank to a tensor of higher rank");
            std::array<size_type, TensorType::rank> new_sizes;
            for (int i = 0; i < TensorType::rank; i++)
                new_sizes[i] = tensor.get_axis_size(i);
            reshape(std::begin(new_sizes), std::end(new_sizes));
        }

        operator span<T>() noexcept { return span<T>(data.get(), size()); }
        operator view<T>() const noexcept { return view<T>(data.get(), size()); }

        friend void swap(Tensor& lhs, Tensor& rhs) noexcept {
            using std::swap;
            swap(lhs.data, rhs.data);
            swap(lhs.sizes, rhs.sizes);
        }

    private:
        std::array<size_type, rank> sizes;
        ManagedPtr<value_type> data;
    };

    /** @brief provides a non-owning mutable span of a Tensor
     *
     * \tparam  T       type of data stored by the tensor
     * \tparam  rank    rank of the tensor
     *
     * A span is valid if and only if the following hold true:
     * - parent tensor is still alive
     * - parent tensor holds a valid memory block
     * - parent tensor hasn't performed any resizing operation since the span was created
     *
     * A span may be used if and only if it is valid.
     */
    template <class T, std::size_t rank_ = CSL_DEFAULT_TENSOR_RANK>
    class TensorSpan {
    public:
        using tensor_type   = Tensor<T, rank_>;
        using value_type    = typename tensor_type::value_type;
        using pointer       = typename tensor_type::pointer;
        using const_pointer = typename tensor_type::const_pointer;
        using size_type     = typename tensor_type::size_type;

        static constexpr auto rank = rank_;

        TensorSpan() noexcept : ptr{ nullptr } { std::fill(std::begin(sizes), std::end(sizes), 0); }
        TensorSpan(const TensorSpan&) noexcept = default;
        TensorSpan(tensor_type& parent) noexcept : ptr{ parent.get() } {
            for (std::size_t i = 0; i < rank; i++)
                sizes[i] = parent.get_axis_size(i);
        }

        /* returns the total number of elements in the span */
        CUDA4DNN_HOST/*_DEVICE*/ size_type size() const noexcept {
            return std::accumulate(std::begin(sizes), std::end(sizes), 1, std::multiplies<size_type>());
        }

        /** returns a shape array consisting of axis lengths in order starting from zero */
        CUDA4DNN_HOST std::array<size_type, rank> shape() const noexcept {
            std::array<size_type, rank> temp;
            std::copy(std::begin(sizes), std::end(sizes), std::begin(temp));
            return temp;
        }

        /** returns true if the tensor is empty */
        CUDA4DNN_HOST/*_DEVICE*/ bool empty() const noexcept { return !size(); }

        /** @brief returns the length of the axis
         *
         * Negative axis numbers can be used to select axis from the lower order.
         * Examples:
         * > -1 represents the last axis
         * > 0 represents the first axis
         * > 1 represents the second axis
         *
         * Pre-conditions:
         * - the axis must be in the range [-rank, rank)
         */
        CUDA4DNN_HOST_DEVICE size_type get_axis_size(int axis) const noexcept {
            axis = clamp_axis(axis, rank);
            CV_Assert(axis >= 0 && axis < rank);
            return sizes[axis];
        }

        /** returns a device pointer to mutable device memory */
        CUDA4DNN_HOST_DEVICE pointer get() const noexcept { return ptr; }

        /** @brief reshapes the span
         *
         * Length deduction:
         * The length of at most one axis can be deduced using the total size constraint. The axis can
         * be marked for deduction by specifying the corresponding size as -1.
         *
         * The axes for which no size was provided (excluding -1) will be assumed to be one.
         *
         * Pre-conditions:
         * - [start, end) represents a range containing length of the axes in order
         * - the number of axis lengths provided must be less than or equal to the tensor rank
         * - at most one axis length is allowed for length deduction
         * - the lengths provided must ensure that the total number of elements remains unchnged
         *
         * Exception Guarantee: Strong
         */
        template <class ForwardItr> CUDA4DNN_HOST
        typename std::enable_if<!std::is_integral<ForwardItr>::value, void> // TODO is_iterator
        ::type reshape(ForwardItr start, ForwardItr end) {
            CV_Assert(start != end);
            CV_Assert(std::distance(start, end) <= rank);

            using ItrValueType = typename std::iterator_traits<ForwardItr>::value_type;

            /* the user may leave at most one axis size for deduction by specifying -1 */
            auto sizes_to_deduce = std::count(start, end, -1);
            if (sizes_to_deduce > 1) { CV_Error(Error::StsBadArg, "only one axis size can be deduced"); }

            /* sizes must be positive numbers with the exception of -1 */
            auto invalid_sizes = std::count_if(start, end, [](ItrValueType x) {
                return !(x > 0 || x == -1);
            });
            if (invalid_sizes) { CV_Error(Error::StsBadArg, "invalid axis size"); }

            /* compute the total number of elements in the new tensor */
            size_type unknown_size = 0;
            auto total = std::accumulate(start, end, 1, std::multiplies<ItrValueType>());
            if (total < 0) {
                /* there is an unknown size */
                if (std::abs(total) <= size()) {
                    unknown_size = size() / std::abs(total);
                    total = size();
                }
                /* Edge case: if `total` is already more than size(), skip the deduction as it's impossible
                ** Since `total` is negative, the size check which follows will fail and throw an error
                */
            }

            /* the number of elements before and after reshape must be exactly same */
            if (total != size()) {
               CV_Error(Error::StsBadArg, "new axes do not preserve the tensor element count");
            }

            /* we assume the size of the unspecified axes to be one */
            std::fill(std::begin(sizes), std::end(sizes), 1);
            std::copy_backward(start, end, std::end(sizes));

            /* replace the unknown axis with the correct value */
            std::replace(std::begin(sizes), std::end(sizes), size_type(-1), unknown_size);
        }

        /** @brief reshapes the span
         * constructs a range out of the arguments and invokes range-based reshape method
         */
        template <class ...Sizes>
        CUDA4DNN_HOST void reshape(Sizes... new_sizes_) {
            static_assert(sizeof...(Sizes) <= rank, "number of axes exceeds the tensor rank");
            std::array<std::int64_t, sizeof...(Sizes)> new_sizes = { static_cast<std::int64_t>(new_sizes_)... };
            reshape(std::begin(new_sizes), std::end(new_sizes));
        }

        /** @brief reshapes the span
         *
         * Pre-conditions:
         * - the reference tensor must be a non-empty tensor
         * - the reference tensor's rank must be lesser than or equal to the rank of target tensor
         *
         * Exception Guarantee: Strong
         */
        template <class TensorType>
        void reshape_as(const TensorType& tensor) {
            static_assert(TensorType::rank <= rank, "cannot reshape a tensor of lower rank to a tensor of higher rank");
            std::array<size_type, TensorType::rank> new_sizes;
            for (int i = 0; i < TensorType::rank; i++)
                new_sizes[i] = tensor.get_axis_size(i);
            reshape(std::begin(new_sizes), std::end(new_sizes));
        }

        /** @brief obtains a subspan of the span
         *
         * The axes for which no size was provided will be assumed to be one.
         *
         * Pre-conditions:
         * - the `offset` must be less than the size of the span
         * - [start, end) represents a range containing length of the subspan axes in order
         * - the number of axis lengths provided must be less than or equal to the tensor rank
         * - the lengths provided must ensure that the number of elements does not exceed (old size - offset)
         *
         * Exception Guarantee: Strong
         */
        template <class ForwardItr> CUDA4DNN_HOST
        typename std::enable_if<!std::is_integral<ForwardItr>::value, TensorSpan> // TODO is_iterator
        ::type subspan(size_type offset, ForwardItr start, ForwardItr end) const {
            CV_Assert(start != end);
            CV_Assert(std::distance(start, end) <= rank);

            auto cur_size = size();
            CV_Assert(offset < cur_size);

            using ItrValueType = typename std::iterator_traits<ForwardItr>::value_type;

            /* sizes must be positive numbers */
            auto invalid_sizes = std::count_if(start, end, [](ItrValueType x) {
                return !(x > 0);
            });
            if (invalid_sizes) { CV_Error(Error::StsBadArg, "invalid axis size"); }

            /* the number of elements must be equal to the new size */
            auto max_size = (cur_size - offset);
            auto total = std::accumulate(start, end, 1, std::multiplies<ItrValueType>());
            if (total > max_size) {
                CV_Error(Error::StsBadArg, "axis lengths lead to OOB accesses");
            }

            TensorSpan temp;

            /* we assume the size of the unspecified axes to be one */
            std::fill(std::begin(temp.sizes), std::end(temp.sizes), 1);
            std::copy_backward(start, end, std::end(temp.sizes));

            temp.ptr = ptr + offset;
            return temp;
        }

        /** @brief obtains a subspan of the span
         * constructs a range out of the size arguments and invokes the range-based subspan method
         */
        template <class ...Sizes>
        CUDA4DNN_HOST TensorSpan subspan(size_type offset, Sizes... new_sizes_) const {
            static_assert(sizeof...(Sizes) <= rank, "number of axes exceeds the tensor rank");
            std::array<std::int64_t, sizeof...(Sizes)> new_sizes = { static_cast<std::int64_t>(new_sizes_)... };
            return subspan(offset, std::begin(new_sizes), std::end(new_sizes));
        }

        operator span<T>() noexcept { return span<T>(ptr, size()); }
        operator view<T>() const noexcept { return view<T>(ptr, size()); }

        friend void swap(TensorSpan& lhs, TensorSpan& rhs) noexcept {
            using std::swap;
            swap(lhs.ptr, rhs.ptr);
            swap(lhs.sizes, rhs.sizes);
        }

    private:
        size_type sizes[rank];
        pointer ptr;
    };

    /** @brief view of a tensor
     *
     * \tparam  T       type of data stored by the tensor
     * \tparam  rank    rank of the tensor
     *
     * A view is valid if and only if the following hold true:
     * - parent tensor is still alive
     * - parent tensor holds a valid memory block
     * - parent tensor hasn't performed any resizing operation since the view was created
     */
    template <class T, std::size_t rank_ = CSL_DEFAULT_TENSOR_RANK>
    class TensorView {
    public:
        using tensor_type   = Tensor<T, rank_>;
        using value_type    = typename tensor_type::value_type;
        using pointer       = typename tensor_type::pointer;
        using const_pointer = typename tensor_type::const_pointer;
        using size_type     = typename tensor_type::size_type;

        static constexpr auto rank = rank_;

        TensorView() noexcept : ptr{ nullptr } { std::fill_n(sizes, rank, 0); }
        TensorView(const TensorView&) noexcept = default;
        TensorView(const TensorSpan<T, rank_>& other) noexcept : ptr{ other.get() } {
            for (int i = 0; i < rank; i++)
                sizes[i] = other.get_axis_size(i);
        }
        TensorView(const tensor_type& parent) noexcept : ptr{ parent.get() } {
            for (std::size_t i = 0; i < rank; i++)
                sizes[i] = parent.get_axis_size(i);
        }

        TensorView& operator=(const TensorView&) = default;
        TensorView& operator=(const TensorSpan<T, rank_>& other) noexcept {
            TensorView tmp(other);
            swap(*this, tmp);
            return *this;
        }

        /* returns the total number of elements in the view */
        CUDA4DNN_HOST/*_DEVICE*/ size_type size() const noexcept {
            return std::accumulate(std::begin(sizes), std::end(sizes), 1, std::multiplies<size_type>());
        }

        /** returns a shape array consisting of axis lengths in order starting from zero */
        CUDA4DNN_HOST std::array<size_type, rank> shape() const noexcept {
            std::array<size_type, rank> temp;
            std::copy(std::begin(sizes), std::end(sizes), std::begin(temp));
            return temp;
        }

        /** returns true if the tensor is empty */
        CUDA4DNN_HOST/*_DEVICE*/ bool empty() const noexcept { return !size(); }

        /** @brief returns the length of the axis
         *
         * Negative axis numbers can be used to select axis from the lower order.
         * Examples:
         * > -1 represents the last axis
         * > 0 represents the first axis
         * > 1 represents the second axis
         *
         * Pre-conditions:
         * - the axis must be in the range [-rank, rank)
         */
        CUDA4DNN_HOST_DEVICE size_type get_axis_size(int axis) const noexcept {
            axis = clamp_axis(axis, rank);
            CV_Assert(axis >= 0 && axis < rank);
            return sizes[axis];
        }

        /** returns a device pointer to immutable device memory */
        CUDA4DNN_HOST_DEVICE const_pointer get() const noexcept { return ptr; }

        /** @brief reshapes the view
         *
         * Length deduction:
         * The length of at most one axis can be deduced using the total size constraint. The axis can
         * be marked for deduction by specifying the size as -1.
         *
         * The axes for which no size was provided (excluding -1) will be assumed to be one.
         *
         * Pre-conditions:
         * - [start, end) represents a range containing length of the axes in order
         * - the number of axis lengths provided must be less than or equal to the tensor rank
         * - at most one axis length is allowed for length deduction
         * - the lengths provided must ensure that the total number of elements remains unchnged
         *
         * Exception Guarantee: Strong
         */
        template <class ForwardItr> CUDA4DNN_HOST
        typename std::enable_if<!std::is_integral<ForwardItr>::value, void>
        ::type reshape(ForwardItr start, ForwardItr end) {
            CV_Assert(start != end);
            CV_Assert(std::distance(start, end) <= rank);

            using ItrValueType = typename std::iterator_traits<ForwardItr>::value_type;

            /* the user may leave at most one axis size for deduction by specifying -1 */
            auto sizes_to_deduce = std::count(start, end, -1);
            if (sizes_to_deduce > 1) { CV_Error(Error::StsBadArg, "only one axis size can be deduced"); }

            /* sizes must be positive numbers with the exception of -1 */
            auto invalid_sizes = std::count_if(start, end, [](ItrValueType x) {
                return !(x > 0 || x == -1);
            });
            if (invalid_sizes) { CV_Error(Error::StsBadArg, "invalid axis size"); }

            /* compute the total number of elements in the new tensor */
            size_type unknown_size = 0;
            auto total = std::accumulate(start, end, 1, std::multiplies<ItrValueType>());
            if (total < 0) {
                /* there is an unknown size */
                if (std::abs(total) <= size()) {
                    unknown_size = size() / std::abs(total);
                    total = size();
                }
                /* Edge case: if `total` is already more than size(), skip the deduction as it's impossible
                ** Since `total` is negative, the size check which follows will fail and throw an error
                */
            }

            /* the number of elements before and after reshape must be exactly same */
            if (total != size()) {
                CV_Error(Error::StsBadArg, "new axes do not preserve the tensor element count");
            }

            /* we assume the size of the unspecified axes to be one */
            std::fill(std::begin(sizes), std::end(sizes), 1);
            std::copy_backward(start, end, std::end(sizes));

            /* replace the unknown axis with the correct value */
            std::replace(std::begin(sizes), std::end(sizes), size_type(-1), unknown_size);
        }

        /** @brief reshapes the view
         * constructs a range out of the arguments and invokes range-based reshape method
         */
        template <class ...Sizes>
        CUDA4DNN_HOST void reshape(Sizes... new_sizes_) {
            static_assert(sizeof...(Sizes) <= rank, "number of axes exceeds the tensor rank");
            std::array<std::int64_t, sizeof...(Sizes)> new_sizes = { static_cast<std::int64_t>(new_sizes_)... };
            reshape(std::begin(new_sizes), std::end(new_sizes));
        }

        /** @brief reshapes the view
         *
         * Pre-conditions:
         * - the reference tensor must be a non-empty tensor
         * - the reference tensor's rank must be lesser than or equal to the rank of target tensor
         *
         * Exception Guarantee: Strong
         */
        template <class TensorType>
        void reshape_as(const TensorType& tensor) {
            static_assert(TensorType::rank <= rank, "cannot reshape a tensor of lower rank to a tensor of higher rank");
            std::array<size_type, TensorType::rank> new_sizes;
            for (int i = 0; i < TensorType::rank; i++)
                new_sizes[i] = tensor.get_axis_size(i);
            reshape(std::begin(new_sizes), std::end(new_sizes));
        }

        /** @brief obtains a subview of the view
         *
         * The axes for which no size was provided will be assumed to be one.
         *
         * Pre-conditions:
         * - the `offset` must be less than the size of the view
         * - [start, end) represents a range containing length of the subview axes in order
         * - the number of axis lengths provided must be less than or equal to the tensor rank
         * - the lengths provided must ensure that the number of elements does not exceed (old size - offset)
         *
         * Exception Guarantee: Strong
         */
        template <class ForwardItr> CUDA4DNN_HOST
        typename std::enable_if<!std::is_integral<ForwardItr>::value, TensorView> // TODO is_iterator
        ::type subview(size_type offset, ForwardItr start, ForwardItr end) const {
            CV_Assert(start != end);
            CV_Assert(std::distance(start, end) <= rank);

            auto cur_size = size();
            CV_Assert(offset < cur_size);

            using ItrValueType = typename std::iterator_traits<ForwardItr>::value_type;

            /* sizes must be positive numbers */
            auto invalid_sizes = std::count_if(start, end, [](ItrValueType x) {
                return !(x > 0);
            });
            if (invalid_sizes) { CV_Error(Error::StsBadArg, "invalid axis size"); }

            /* the number of elements must be equal to the new size */
            auto max_size = (cur_size - offset);
            auto total = std::accumulate(start, end, 1, std::multiplies<ItrValueType>());
            if (total > max_size) {
                CV_Error(Error::StsBadArg, "axes lengths lead to OOB accesses");
            }

            TensorView temp;

            /* we assume the size of the unspecified axes to be one */
            std::fill(std::begin(temp.sizes), std::end(temp.sizes), 1);
            std::copy_backward(start, end, std::end(temp.sizes));

            temp.ptr = ptr + offset;
            return temp;
        }

        /** @brief obtains a subview of the view
         * constructs a range out of the size arguments and invokes the range-based subview method
         */
        template <class ...Sizes>
        CUDA4DNN_HOST TensorView subview(size_type offset, Sizes... new_sizes_) const {
            static_assert(sizeof...(Sizes) <= rank, "number of axes exceeds the tensor rank");
            std::array<std::int64_t, sizeof...(Sizes)> new_sizes = { static_cast<std::int64_t>(new_sizes_)... };
            return subview(offset, std::begin(new_sizes), std::end(new_sizes));
        }

        operator view<T>() const noexcept { return view<T>(ptr, size()); }

        friend void swap(TensorView& lhs, TensorView& rhs) noexcept {
            using std::swap;
            swap(lhs.ptr, rhs.ptr);
            swap(lhs.sizes, rhs.sizes);
        }

    private:
        size_type sizes[rank];
        const_pointer ptr;
    };

    /** if the \p axis is a negative index, the equivalent postive index is returned; otherwise, returns \p axis */
    template <class T>
    CUDA4DNN_HOST_DEVICE constexpr T clamp_axis(T axis, std::size_t rank) {
        return axis < 0 ? axis + rank : axis;
    }

    /** returns true if the two TensorType objects have the same shape */
    template <class TensorType1, class TensorType2> inline
    bool is_shape_same(const TensorType1& x, const TensorType2& y) noexcept {
        constexpr auto rank1 = TensorType1::rank;
        constexpr auto rank2 = TensorType2::rank;

        if (rank1 != rank2)
            return false;

        for (int i = 0; i < rank1; i++)
            if (x.get_axis_size(i) != y.get_axis_size(i))
                return false;
        return true;
    }

    /** returns true if the two TensorType objects are compatible */
    template <class TensorType1, class TensorType2> inline
    bool is_shape_compatible(const TensorType1& x, const TensorType2& y) noexcept {
        constexpr auto rank1 = TensorType1::rank;
        constexpr auto rank2 = TensorType2::rank;

        if (rank1 != rank2)
            return false;

        for (int i = 0; i < rank1; i++)
            if (x.get_axis_size(i) != y.get_axis_size(i) &&
                x.get_axis_size(i) != 1 && y.get_axis_size(i) != 1)
                return false;
        return true;
    }

    /** returns the rank to which the given tensor can be squeezed to */
    template <class TensorType> inline
    std::size_t get_effective_rank(const TensorType& x) noexcept {
        constexpr auto rank = TensorType::rank;
        std::size_t effective_rank = rank;
        for (int i = 0; i < rank; i++, effective_rank--)
            if (x.get_axis_size(i) != 1)
                break;
        return effective_rank;
    }

    template <class Container, class T = int> inline
    std::vector<T> squeeze_shape(const Container& shape, std::size_t upto_rank = 1) {
        auto start = std::find_if(std::begin(shape), std::end(shape) - upto_rank + 1, [] (T x) { return x != 1; });
        return { start, std::end(shape) };
    }

    namespace tensor_ops {

        /** @brief performs generalized matrix-multiplication
         *
         * Pre-conditions:
         * - \p A and \p B must meet the mathematical requirements for matrix multiplication
         * - \p result must be large enough to hold the result
         *
         * Exception Gaurantee: Basic
         */
        template <class T> inline
        void gemm(const cublas::Handle& handle, T beta, TensorSpan<T> result, T alpha, bool transa, TensorView<T> A, bool transb, TensorView<T> B) {
            /* matrix operations can be performed only on rank two or less tensors */
            CV_Assert(get_effective_rank(A) <= 2 &&
                get_effective_rank(B) <= 2 &&
                get_effective_rank(result) <= 2);

            /* check dimension requirements for matrix multiplication */
            if (!transa && !transb) {
                CV_Assert(A.get_axis_size(-2) == result.get_axis_size(-2));
                CV_Assert(A.get_axis_size(-1) == B.get_axis_size(-2));
                CV_Assert(B.get_axis_size(-1) == result.get_axis_size(-1));
            } else if (!transa && transb) {
                CV_Assert(A.get_axis_size(-2) == result.get_axis_size(-2));
                CV_Assert(A.get_axis_size(-1) == B.get_axis_size(-1));
                CV_Assert(B.get_axis_size(-2) == result.get_axis_size(-1));
            } else if (transa && !transb) {
                CV_Assert(A.get_axis_size(-1) == result.get_axis_size(-2));
                CV_Assert(A.get_axis_size(-2) == B.get_axis_size(-2));
                CV_Assert(B.get_axis_size(-1) == result.get_axis_size(-1));
            } else {
                CV_Assert(A.get_axis_size(-1) == result.get_axis_size(-2));
                CV_Assert(A.get_axis_size(-2) == B.get_axis_size(-1));
                CV_Assert(B.get_axis_size(-2) == result.get_axis_size(-1));
            }

            const auto result_nr = result.get_axis_size(-2);
            const auto result_nc = result.get_axis_size(-1);
            const auto common_dim = A.get_axis_size(transa ? -2 : -1);
            const auto A_nc = A.get_axis_size(-1);
            const auto B_nc = B.get_axis_size(-1);

            /* tensors are stored in row-major but cublas::gemm operates on column-major matrices
             * a row-major matrix when read as column-major matrix gives the transpose of the intended matrix
             *
             * Required: C = AB
             * what cuBLAS sees: C^T = A^TB^T = (BA)^T
             *
             * By reversing operands, we effectively perform:
             * C^T = B^TA^T = (AB)^T
             *
             * which gives C = AB
             */
            cublas::gemm<T>(handle,
                transb, transa,
                result_nc, result_nr, common_dim,
                alpha, B.get(), B_nc,
                A.get(), A_nc,
                beta, result.get(), result_nc);
        }

        /** @brief performs element-wise addition with broadcasting
        *
        * Pre-conditions:
        * - \p A and \p C must be compatible tensors
        *
        * Exception Gaurantee: Basic
        */
        template <class T> inline
        void add(const cudnn::Handle& handle, T beta, TensorSpan<T> C, T alpha, TensorView<T> A) {
            CV_Assert(is_shape_compatible(A, C));

            using cudnn::TensorDescriptor;
            auto aDesc = TensorDescriptor<T>(A.shape());
            auto cDesc = TensorDescriptor<T>(C.shape());
            cudnn::add(handle, alpha, aDesc, A.get(), beta, cDesc, C.get());
        }

        /** @brief performs element-wise addition with broadcasting
        *
        * Pre-conditions:
        * - \p A and \p result must be compatible tensors
        *
        * Exception Gaurantee: Basic
        */
        template <class T> inline
        void softmax(const cudnn::Handle& handle, TensorSpan<T> output, TensorView<T> input, int channel_axis, bool log) {
            CV_Assert(is_shape_same(output, input));

            channel_axis = clamp_axis(channel_axis, input.rank);

            std::size_t outer_size = 1;
            for (int j = 0; j < channel_axis; j++)
                outer_size *= input.get_axis_size(j);

            auto channel_size = input.get_axis_size(channel_axis);

            std::size_t inner_size = 1;
            for (int j = channel_axis + 1; j < input.rank; j++)
                inner_size *= input.get_axis_size(j);

            std::array<std::size_t, 4> shape = { outer_size, channel_size, 1 , inner_size };

            using cudnn::TensorDescriptor;
            auto inputDesc = TensorDescriptor<T>(shape);
            auto outputDesc = TensorDescriptor<T>(shape);
            cudnn::softmax(handle, outputDesc, output.get(), inputDesc, input.get(), log);
        }

        template <class T> inline
        void abs(const Stream& stream, TensorSpan<T> dest, TensorView<T> src) {
            CV_Assert(is_shape_same(dest, src));
            kernels::abs<T>(stream, dest, src);
        }

        template <class T> inline
        void bnll(const Stream& stream, TensorSpan<T> dest, TensorView<T> src) {
            CV_Assert(is_shape_same(dest, src));
            kernels::bnll<T>(stream, dest, src);
        }

        template <class T> inline
        void relu(const Stream& stream, TensorSpan<T> dest, TensorView<T> src, T slope = 0) {
            CV_Assert(is_shape_same(dest, src));
            kernels::relu<T>(stream, dest, src, slope);
        }

        template <class T> inline
        void clipped_relu(const Stream& stream, TensorSpan<T> dest, TensorView<T> src, T min, T max) {
            CV_Assert(is_shape_same(dest, src));
            kernels::clipped_relu<T>(stream, dest, src, min, max);
        }

        template <class T> inline
        void channelwise_relu(const Stream& stream, TensorSpan<T> dest, TensorView<T> src, TensorView<T> slope) {
            CV_Assert(is_shape_same(dest, src));
            CV_Assert(src.get_axis_size(1) == slope.size());
            std::size_t inner_size = src.size() / src.get_axis_size(0);
            std::size_t channel_size = inner_size / src.get_axis_size(1);
            kernels::axiswise_relu<T>(stream, dest, src, slope, inner_size, channel_size);
        }

        template <class T> inline
        void elu(const Stream& stream, TensorSpan<T> dest, TensorView<T> src) {
            CV_Assert(is_shape_same(dest, src));
            kernels::elu<T>(stream, dest, src);
        }

        template <class T> inline
        void power(const Stream& stream, TensorSpan<T> dest, TensorView<T> src, T exp = 1, T scale = 1, T shift = 0) {
            CV_Assert(is_shape_same(dest, src));
            kernels::power<T>(stream, dest, src, exp, scale, shift);
        }

        template <class T> inline
        void sigmoid(const Stream& stream, TensorSpan<T> dest, TensorView<T> src) {
            CV_Assert(is_shape_same(dest, src));
            kernels::sigmoid<T>(stream, dest, src);
        }

        template <class T> inline
        void tanh(const Stream& stream, TensorSpan<T> dest, TensorView<T> src) {
            CV_Assert(is_shape_same(dest, src));
            kernels::tanh<T>(stream, dest, src);
        }
    }

    template <class T>
    class Convolution {
    public:
        struct params_type {
            std::vector<std::size_t> input_shape;
            std::vector<std::size_t> filter_shape;

            std::vector<std::size_t> padding;
            std::vector<std::size_t> stride;
            std::vector<std::size_t> dialation;

            std::size_t groups;
        };

        Convolution() = default;
        Convolution(const Convolution&) = delete;
        Convolution(Convolution&&) = default;
        Convolution(cudnn::Handle handle, const params_type& params) {
            cudnnHandle = std::move(handle);

            inputTensorDesc = TensorDescriptor(params.input_shape);
            filterDesc = FilterDescriptor(params.filter_shape);
            convDesc = ConvolutionDescriptor(params.padding, params.stride, params.dialation, params.groups);

            std::vector<int> output_dims;
            getConvolutionForwardOutputDim(convDesc, filterDesc, inputTensorDesc, output_dims);

            outputTensorDesc = TensorDescriptor(output_dims);

            algo = ConvolutionAlgorithm(cudnnHandle, convDesc, filterDesc, inputTensorDesc, outputTensorDesc);
        }

        Convolution& operator=(const Convolution&) = delete;
        Convolution& operator=(Convolution&&) = default;

        std::size_t get_workspace_size() const noexcept {
            return algo.get_workspace_size();
        }

        void convolve(TensorSpan<T> output, TensorView<T> input, TensorView<T> filters, Workspace& scratchpad) {
            cudnn::convolve<T>(cudnnHandle,
                filterDesc, filters.get(),
                convDesc, algo, WorkspaceAccessor::get(scratchpad),
                inputTensorDesc, input.get(), 1.0,
                0.0, outputTensorDesc, output.get()
            );
        }

    private:
        cudnn::Handle cudnnHandle;

        using TensorDescriptor = cudnn::TensorDescriptor<T>;
        TensorDescriptor inputTensorDesc, outputTensorDesc;

        using FilterDescriptor = cudnn::FilterDescriptor<T>;
        FilterDescriptor filterDesc;

        using ConvolutionDescriptor = cudnn::ConvolutionDescriptor<T>;
        ConvolutionDescriptor convDesc;

        using ConvolutionAlgorithm = cudnn::ConvolutionAlgorithm<T>;
        ConvolutionAlgorithm algo;
    };

}}}} /* namespace cv::dnn::cuda4dnn::csl */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_TENSOR_HPP */
