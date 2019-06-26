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

    /** if the \p axis is a negative index, the equivalent postive index is returned; otherwise, returns \p axis */
    template <class T>
    CUDA4DNN_HOST_DEVICE constexpr T clamp_axis(T axis, std::size_t rank) {
        return axis < 0 ? axis + rank : axis;
    }

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
            assert(axis >= 0 && axis < rank);
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
            assert(axis >= 0 && axis < rank); /* CV_Assert isn't allowed in device code */
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

}}}} /* namespace cv::dnn::cuda4dnn::csl */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_TENSOR_HPP */
