// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_CSL_TENSOR_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_CSL_TENSOR_HPP

#include "nvcc_defs.hpp"
#include "memory.hpp"
#include "cublas.hpp"
#include "cudnn.hpp"
#include "span.hpp"

#include "../cxx_utils/resizable_static_array.hpp"
#include "../cxx_utils/is_iterator.hpp"

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

#ifndef CSL_MAX_TENSOR_RANK
    #define CSL_MAX_TENSOR_RANK 6
#endif

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl {

    /** \file tensor.hpp
     *
     *     TYPE     | OWNERSHIP | MUTABLE
     * ------------ + --------- + --------
     *    Tensor    |    Yes    |   Yes
     *  TensorSpan  |    No     |   Yes
     *  TensorView  |    No     |   No
     *
     * Tensor is implicitly convertible to TensorSpan and TensorView
     * TensorSpan is implicitly convertible to TensorView
     *
     * Concepts and template parameter naming convention:
     * - "MutableTensorType" can refer to a Tensor or TensorSpan
     * - "ImmutableTensorType" can refer to a Tensor, TensorSpan or TensorView
     * - "TensorType" can refer to a Tensor, TensorSpan or TensorView
     *
     * "ImmutableTensorType" is used when the tensor data might be used.
     * "TensorType" is used when only meta-information such as the size or shape is required, i.e. the data won't be touched
     */

    /** if the \p axis is a negative index, the equivalent positive index is returned; otherwise, returns \p axis */
    CUDA4DNN_HOST_DEVICE constexpr std::size_t clamp_axis(int axis, std::size_t rank) {
        return axis < 0 ? axis + rank : axis;
    }

    /** @brief multi-dimensional contiguous non-copyable GPU tensor
     *
     * \tparam  T       type of data stored
     *
     * @note scalars or zero rank tensors are not supported
     * @note the maximum rank supported is controlled by the `CSL_MAX_TENSOR_RANK` preprocessor symbol
     */
    template <class T>
    class Tensor {
        static_assert(std::is_standard_layout<T>::value, "T must satisfy StandardLayoutType");

    public:
        using value_type    = typename ManagedPtr<T>::element_type;
        using pointer       = typename ManagedPtr<value_type>::pointer;
        using const_pointer = typename ManagedPtr<value_type>::const_pointer;
        using size_type     = typename ManagedPtr<value_type>::size_type;

        Tensor() noexcept { }
        Tensor(const Tensor&) = delete;
        Tensor(Tensor&& other) noexcept {
            data = std::move(other.data);
            shape = other.shape;
            other.shape.clear();
        }

        /** @brief constructs a tensor of a specific shape
         *
         * Whatever arguments are accepted by the resize methods are accepted here.
         */
        template <class ...Args>
        Tensor(Args&&... sizes) { resize(std::forward<Args>(sizes)...); }

        Tensor& operator=(const Tensor&) = delete;
        Tensor& operator=(Tensor&& other) noexcept {
            data = std::move(other.data);
            shape = other.shape;
            other.shape.clear();
            return *this;
        }

        /** returns true if the tensor is empty (or uninitialized) */
        bool empty() const noexcept { return shape.size() == 0; }

        /** returns the total number of elements in the tensor
         *
         * Pre-conditions:
         * - tensor must be non-empty
         */
        size_type size() const noexcept {
            CV_Assert(!empty());
            return std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<size_type>());
        }

        /** returns the rank of the tensor
         *
         * Pre-conditions:
         * - tensor must be non-empty
         */
        size_type rank() const noexcept {
            CV_Assert(!empty());
            return shape.size();
        }

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
         * - tensor must be non-empty
         * - the axis must be in the range [-rank(), rank())
         */
        size_type get_axis_size(int axis) const noexcept {
            axis = clamp_axis(axis, rank());
            CV_Assert(axis >= 0 && axis < rank());
            return shape[axis];
        }

        /** @brief returns the combined size of the axes in an axis range
         *
         * if the shape is [3 x 5 x 7 x 11]
         * - `size_range(0, 2)` will return 3 x 5 = 15
         * - `size_range(1, 3)` will return 5 x 7 = 35
         * - `size_range(0, 4)` will return 3 x 5 x 7 x 11 = 1155
         *
         * Pre-conditions:
         * - tensor must be non-empty
         * - `axis_start` must be less than or equal to `axis_end`
         * - `axis_end` must be less than or equal to the rank
         *
         * returns one if the two `axis_start` and `axis_end` are equal
         */
        size_type size_range(size_type axis_start, size_type axis_end) const noexcept {
            CV_Assert(!empty());
            CV_Assert(axis_start <= axis_end);
            CV_Assert(axis_end <= rank());
            auto start = std::begin(shape) + axis_start;
            auto end = std::begin(shape) + axis_end;
            return std::accumulate(start, end, 1, std::multiplies<size_type>());
        }

        /** returns an std::vector containing axis lengths starting from axis zero
         *
         * Pre-conditions:
         * - tensor must be non-empty
         *
         * Exception Guarantee: Strong
         */
        std::vector<size_type> shape_as_vector() const {
            CV_Assert(!empty());
            return std::vector<size_type>(std::begin(shape), std::end(shape));
        }

        /** returns a pointer to mutable device memory owned by the tensor */
        pointer get() noexcept { return data.get(); }

        /** returns a pointer to immutable device memory owned by the tensor */
        const_pointer get() const noexcept { return data.get(); }

        /** @brief releases the memory owned by the tensor
         *
         * Pre-conditions:
         * - tensor must be non-empty
         *
         * Exception Guarantee: Strong
         */
        void clear() {
            CV_Assert(!empty());
            data.reset();
            shape.clear();
        }

        /** @brief resizes the tensor
         *
         * Pre-conditions:
         * - [start, end) represents a forward range containing the length of the axes in order starting from axis zero
         * - number of lengths provided must not exceed the maximum tensor rank (CSL_MAX_TENSOR_RANK)
         * - the sizes must be positive integers
         *
         * Exception Guarantee: Strong
         */
        template <class ForwardItr>
        typename std::enable_if<cxx_utils::is_forward_iterator<ForwardItr>::value, void>
        ::type resize(ForwardItr start, ForwardItr end) {
            CV_Assert(start != end);
            CV_Assert(std::distance(start, end) <= CSL_MAX_TENSOR_RANK);

            using ItrValueType = typename std::iterator_traits<ForwardItr>::value_type;
            auto total = std::accumulate(start, end, 1, std::multiplies<ItrValueType>());
            data.reset(total);

            shape.assign(start, end);
        }

        /** @brief resizes the tensor
         * constructs a range out of the arguments and invokes the range-based resize method
         */
        template <class ...Sizes>
        void resize(Sizes... new_sizes_) {
            static_assert(sizeof...(Sizes) <= CSL_MAX_TENSOR_RANK, "required rank exceeds maximum supported rank");
            static_assert(sizeof...(Sizes) > 0, "no sizes provided");
            std::array<size_type, sizeof...(Sizes)> new_sizes = { static_cast<size_type>(new_sizes_)... };
            resize(std::begin(new_sizes), std::end(new_sizes));
        }

        /** @brief resizes the tensor
         *
         * Pre-conditions:
         * - the reference tensor must be non-empty
         *
         * Exception Guarantee: Strong
         */
        template <class TensorType>
        void resize_as(const TensorType& tensor) {
            CV_Assert(!tensor.empty());
            cxx_utils::resizable_static_array<size_type, CSL_MAX_TENSOR_RANK> new_sizes(tensor.rank());
            for (int i = 0; i < new_sizes.size(); i++)
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
         * - the tensor must be non-empty
         * - [start, end) represents a forward range containing the length of the axes starting from axis zero
         * - the number of lengths provided must be less than or equal to the tensor rank
         * - at most one axis length is allowed for length deduction
         * - the lengths provided must ensure that the total number of elements remains unchanged
         *
         * Exception Guarantee: Strong
         */
        template <class ForwardItr>
        typename std::enable_if<cxx_utils::is_forward_iterator<ForwardItr>::value, void>
        ::type reshape(ForwardItr start, ForwardItr end) {
            CV_Assert(start != end);
            CV_Assert(std::distance(start, end) <= rank());

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
            std::fill(std::begin(shape), std::end(shape), 1);
            std::copy_backward(start, end, std::end(shape));

            /* replace the unknown axis with the correct value */
            std::replace(std::begin(shape), std::end(shape), size_type(-1), unknown_size);
        }

        /** @brief reshapes the tensor
         * constructs a range out of the arguments and invokes range-based reshape method
         */
        template <class ...Sizes>
        void reshape(Sizes... new_sizes_) {
            static_assert(sizeof...(Sizes) <= CSL_MAX_TENSOR_RANK, "required rank exceeds maximum supported rank");
            static_assert(sizeof...(Sizes) > 0, "no sizes provided");
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
            CV_Assert(!tensor.empty());
            cxx_utils::resizable_static_array<size_type, CSL_MAX_TENSOR_RANK> new_sizes(tensor.rank());
            for (int i = 0; i < new_sizes.size(); i++)
                new_sizes[i] = tensor.get_axis_size(i);
            reshape(std::begin(new_sizes), std::end(new_sizes));
        }

        /** @brief squeezes the tensor
         *
         * removes all axes of unit size
         *
         * Pre-conditions:
         * - the tensor must be non-empty
         * - the tensor's rank must be at least two
         *
         * Exception Guarantee: Strong
         */
        void squeeze() {
            CV_Assert(!empty());
            CV_Assert(rank() >= 2);
            auto itr = std::remove(std::begin(shape), std::end(shape), 1);
            shape.resize(itr - std::begin(shape));
        }

        /** @brief squeezes the tensor
         *
         * removes the specified axis if the axis length is one; otherwise, ignores the request
         *
         * Pre-conditions:
         * - the tensor must be non-empty
         * - the tensor's rank must be at least two
         *
         * Exception Guarantee: Strong
         */
        void squeeze(int axis) {
            CV_Assert(!empty());
            CV_Assert(rank() >= 2);
            axis = clamp_axis(axis, rank());
            CV_Assert(axis >= 0 && axis < rank());
            shape.erase(std::begin(shape) + axis);
        }

        /** @brief squeezes the tensor
         *
         * removes leading singleton axes until the tensor's rank is equal to the requested rank
         *
         * Pre-conditions:
         * - the tensor must be non-empty
         * - the tensor's rank must be at least two
         * - the tensor's rank must be at least the requested rank
         * - the tensor must be squeezable up to the requested rank
         *
         * Exception Guarantee: Strong
         */
        void squeeze_to(int r) {
            CV_Assert(!empty());
            CV_Assert(rank() >= r);
            CV_Assert(std::all_of(std::begin(shape), std::end(shape) - r, [](size_type x){ return x == 1; }));
            std::copy(std::end(shape) - r, std::end(shape), std::begin(shape));
            shape.resize(r);
        }

        /** @brief unsqueezes the tensor
         *
         * adds a axis of unit size at the requested before the specified axis
         *
         * Pre-conditions:
         * - the tensor must be non-empty
         * - the tensor's rank must be less than the maximum supported rank (CSL_MAX_TENSOR_RANK)
         *
         * Exception Guarantee: Strong
         */
        void unsqueeze(int axis = 0) {
            CV_Assert(!empty());
            CV_Assert(rank() < CSL_MAX_TENSOR_RANK);
            axis = clamp_axis(axis, rank());
            CV_Assert(axis >= 0 && axis < rank());
            shape.insert(std::begin(shape) + axis, 1);
        }

        operator Span<T>() noexcept { return Span<T>(data.get(), size()); }
        operator View<T>() const noexcept { return View<T>(data.get(), size()); }

        friend void swap(Tensor& lhs, Tensor& rhs) noexcept {
            using std::swap;
            swap(lhs.data, rhs.data);
            swap(lhs.shape, rhs.shape);
        }

    private:
        cxx_utils::resizable_static_array<size_type, CSL_MAX_TENSOR_RANK> shape;
        ManagedPtr<value_type> data;
    };

    /** @brief provides a non-owning mutable span of a Tensor
     *
     * \tparam  T       type of data stored by the tensor
     *
     * A span is valid if and only if the following hold true:
     * - span is non-empty
     * - spanned memory is still allocated
     *
     * A span may be used if and only if it is valid.
     */
    template <class T>
    class TensorSpan {
    public:
        using value_type    = typename Tensor<T>::value_type;
        using pointer       = typename Tensor<T>::pointer;
        using const_pointer = typename Tensor<T>::const_pointer;
        using size_type     = typename Tensor<T>::size_type;

        TensorSpan() noexcept : ptr{ nullptr } { }
        TensorSpan(const TensorSpan&) noexcept = default;
        TensorSpan(Tensor<T>& tensor) noexcept : ptr{ tensor.get() } {
            const auto rank = tensor.rank();
            shape.resize(rank);
            for (int i = 0; i < rank; i++)
                shape[i] = tensor.get_axis_size(i);
        }

        template <class ForwardItr>
        TensorSpan(pointer ptr_, ForwardItr start, ForwardItr end) : ptr{ ptr_ } {
            CV_Assert(start != end);
            CV_Assert(std::distance(start, end) <= CSL_MAX_TENSOR_RANK);

            using ItrValueType = typename std::iterator_traits<ForwardItr>::value_type;
            if (std::any_of(start, end, [](ItrValueType x) { return x <= 0; })) {
                CV_Error(Error::StsBadArg, "the given shape contains negative or zero size");
            }

            shape.assign(start, end);
        }

        /** creates a subspan of a tensor (or span); refer to subspan method for more details */
        template <class... Args>
        TensorSpan(TensorSpan other, size_type offset, Args&&... args)
            : TensorSpan(other.subspan(offset, std::forward<Args>(args)...)) { }

        /** returns true if the span is empty */
        bool empty() const noexcept { return shape.size() == 0; }

        /** returns the total number of elements in the span
         *
         * Pre-conditions:
         * - span must be non-empty
         */
        size_type size() const noexcept {
            CV_Assert(!empty());
            return std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<size_type>());
        }

        /** returns the rank of the span
         *
         * Pre-conditions:
         * - span must be non-empty
         */
        size_type rank() const noexcept {
            CV_Assert(!empty());
            return shape.size();
        }

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
         * - span must be non-empty
         * - the axis must be in the range [-rank(), rank())
         */
        size_type get_axis_size(int axis) const noexcept {
            axis = clamp_axis(axis, rank());
            CV_Assert(axis >= 0 && axis < rank());
            return shape[axis];
        }

        /** @brief returns the combined size of the axes in an axis range
         *
         * if the shape is [3 x 5 x 7 x 11]
         * - `size_range(0, 2)` will return 3 x 5 = 15
         * - `size_range(1, 3)` will return 5 x 7 = 35
         * - `size_range(0, 4)` will return 3 x 5 x 7 x 11 = 1155
         *
         * Pre-conditions:
         * - span must be non-empty
         * - `axis_start` must be less than or equal to `axis_end`
         * - `axis_end` must be less than or equal to the rank
         *
         * returns one if the two `axis_start` and `axis_end` are equal
         */
        size_type size_range(size_type axis_start, size_type axis_end) const noexcept {
            CV_Assert(!empty());
            CV_Assert(axis_start <= axis_end);
            CV_Assert(axis_end <= rank());
            auto start = std::begin(shape) + axis_start;
            auto end = std::begin(shape) + axis_end;
            return std::accumulate(start, end, 1, std::multiplies<size_type>());
        }

        /** returns an std::vector containing axis lengths starting from axis zero
         *
         * Pre-conditions:
         * - span must be non-empty
         *
         * Exception Guarantee: Strong
         */
        std::vector<size_type> shape_as_vector() const {
            CV_Assert(!empty());
            return std::vector<size_type>(std::begin(shape), std::end(shape));
        }

        /** returns a pointer to mutable device memory */
        pointer get() const noexcept { return ptr; }

        /** @brief clears the span
         *
         * Pre-conditions:
         * - span must be non-empty
         *
         * Exception Guarantee: Strong
         */
        void clear() noexcept {
            CV_Assert(!empty());
            ptr = nullptr;
            shape.clear();
        }

        /** @brief reshapes the span
         *
         * Length deduction:
         * The length of at most one axis can be deduced using the total size constraint. The axis can
         * be marked for deduction by specifying the corresponding size as -1.
         *
         * The axes for which no size was provided (excluding -1) will be assumed to be one.
         *
         * Pre-conditions:
         * - the span must be non-empty
         * - [start, end) represents a forward range containing the length of the axes in order
         * - the number of axis lengths must be less than or equal to the rank
         * - at most one axis length is allowed for length deduction
         * - the lengths provided must ensure that the total number of elements remains unchanged
         *
         * Exception Guarantee: Strong
         */
        template <class ForwardItr>
        typename std::enable_if<cxx_utils::is_forward_iterator<ForwardItr>::value, void>
        ::type reshape(ForwardItr start, ForwardItr end) {
            CV_Assert(start != end);
            CV_Assert(std::distance(start, end) <= rank());

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
            std::fill(std::begin(shape), std::end(shape), 1);
            std::copy_backward(start, end, std::end(shape));

            /* replace the unknown axis with the correct value */
            std::replace(std::begin(shape), std::end(shape), size_type(-1), unknown_size);
        }

        /** @brief reshapes the tensor
         * constructs a range out of the arguments and invokes the range-based reshape method
         */
        template <class ...Sizes>
        void reshape(Sizes... new_sizes_) {
            static_assert(sizeof...(Sizes) <= CSL_MAX_TENSOR_RANK, "unsupported tensor rank");
            static_assert(sizeof...(Sizes) > 0, "no sizes provided");
            std::array<std::int64_t, sizeof...(Sizes)> new_sizes = { static_cast<std::int64_t>(new_sizes_)... };
            reshape(std::begin(new_sizes), std::end(new_sizes));
        }

        /** @brief reshapes the span
         *
         * Pre-conditions:
         * - the reference tensor/span/view must be non-empty
         * - the reference tensor/span/view's rank must be less than or equal to the rank of the span
         *
         * Exception Guarantee: Strong
         */
        template <class TensorType>
        void reshape_as(const TensorType& tensor) {
            CV_Assert(!tensor.empty());
            cxx_utils::resizable_static_array<size_type, CSL_MAX_TENSOR_RANK> new_sizes(tensor.rank());
            for (int i = 0; i < new_sizes.size(); i++)
                new_sizes[i] = tensor.get_axis_size(i);
            reshape(std::begin(new_sizes), std::end(new_sizes));
        }

        /** @brief squeezes the tensor
         *
         * removes all axes of unit size
         *
         * Pre-conditions:
         * - the span must be non-empty
         * - the span's rank must be at least two
         *
         * Exception Guarantee: Strong
         */
        void squeeze() {
            CV_Assert(!empty());
            CV_Assert(rank() >= 2);
            auto itr = std::remove(std::begin(shape), std::end(shape), 1);
            shape.resize(itr - std::begin(shape));
        }

        /** @brief squeezes the tensor
         *
         * removes the specified axis if the axis length is one; otherwise, ignores the request
         *
         * Pre-conditions:
         * - the span must be non-empty
         * - the span's rank must be at least two
         *
         * Exception Guarantee: Strong
         */
        void squeeze(int axis) {
            CV_Assert(!empty());
            CV_Assert(rank() >= 2);
            axis = clamp_axis(axis, rank());
            CV_Assert(axis >= 0 && axis < rank());
            shape.erase(std::begin(shape) + axis);
        }

        /** @brief squeezes the tensor
         *
         * removes leading singleton axes until the tensor's rank is equal to the requested rank
         *
         * Pre-conditions:
         * - the tensor must be non-empty
         * - the tensor's rank must be at least two
         * - the tensor's rank must be at least the requested rank
         * - the tensor must be squeezable up to the requested rank
         *
         * Exception Guarantee: Strong
         */
        void squeeze_to(int r) {
            CV_Assert(!empty());
            CV_Assert(rank() >= r);
            CV_Assert(std::all_of(std::begin(shape), std::end(shape) - r, [](size_type x){ return x == 1; }));
            std::copy(std::end(shape) - r, std::end(shape), std::begin(shape));
            shape.resize(r);
        }

        /** @brief unsqueezes the tensor
         *
         * adds a axis of unit size at the requested before the specified axis
         *
         * Pre-conditions:
         * - the span must be non-empty
         * - the span's rank must be less than the maximum supported rank (CSL_MAX_TENSOR_RANK)
         *
         * Exception Guarantee: Strong
         */
        void unsqueeze(int axis = 0) {
            CV_Assert(!empty());
            CV_Assert(rank() < CSL_MAX_TENSOR_RANK);
            axis = clamp_axis(axis, rank());
            CV_Assert(axis >= 0 && axis < rank());
            shape.insert(std::begin(shape) + axis, 1);
        }

        /** @brief obtains a subspan of the span
         *
         * Pre-conditions:
         * - the span must be non-empty
         * - the `offset` must be less than the size of the span
         * - [start, end) represents a forward range containing length of the subspan axes
         * - the lengths provided must ensure that the number of elements does not exceed (old size - offset)
         *
         * Exception Guarantee: Strong
         */
        template <class ForwardItr>
        typename std::enable_if<cxx_utils::is_forward_iterator<ForwardItr>::value, TensorSpan>
        ::type subspan(size_type offset, ForwardItr start, ForwardItr end) const {
            CV_Assert(start != end);
            CV_Assert(std::distance(start, end) <= rank());

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
            temp.shape.assign(start, end);
            temp.ptr = ptr + offset;
            return temp;
        }

        /** @brief obtains a subspan of the span
         * constructs a range out of the size arguments and invokes the range-based subspan method
         */
        template <class ...Sizes>
        TensorSpan subspan(size_type offset, Sizes... new_sizes_) const {
            static_assert(sizeof...(Sizes) <= CSL_MAX_TENSOR_RANK, "required rank exceeds maximum supported rank");
            static_assert(sizeof...(Sizes) > 0, "no sizes provided");
            std::array<std::int64_t, sizeof...(Sizes)> new_sizes = { static_cast<std::int64_t>(new_sizes_)... };
            return subspan(offset, std::begin(new_sizes), std::end(new_sizes));
        }

        operator Span<T>() noexcept { return Span<T>(ptr, size()); }
        operator View<T>() const noexcept { return View<T>(ptr, size()); }

        friend void swap(TensorSpan& lhs, TensorSpan& rhs) noexcept {
            using std::swap;
            swap(lhs.ptr, rhs.ptr);
            swap(lhs.shape, rhs.shape);
        }

    private:
        cxx_utils::resizable_static_array<size_type, CSL_MAX_TENSOR_RANK> shape;
        pointer ptr;
    };

    /** @brief view of a tensor
     *
     * \tparam  T       type of data stored by the tensor
     *
     * A view is valid if and only if the following hold true:
     * - view is non-empty
     * - viewed memory is still allocated
     */
    template <class T>
    class TensorView {
    public:
        using value_type    = typename Tensor<T>::value_type;
        using pointer       = typename Tensor<T>::pointer;
        using const_pointer = typename Tensor<T>::const_pointer;
        using size_type     = typename Tensor<T>::size_type;

        TensorView() noexcept : ptr{ nullptr } { }
        TensorView(const TensorView&) noexcept = default;
        TensorView(TensorSpan<T> other) noexcept : ptr{ other.get() } {
            const auto rank = other.rank();
            shape.resize(rank);
            for (int i = 0; i < rank; i++)
                shape[i] = other.get_axis_size(i);
        }
        TensorView(const Tensor<T>& tensor) noexcept : ptr{ tensor.get() } {
            const auto rank = tensor.rank();
            shape.resize(rank);
            for (int i = 0; i < rank; i++)
                shape[i] = tensor.get_axis_size(i);
        }

        template <class ForwardItr>
        TensorView(const_pointer ptr_, ForwardItr start, ForwardItr end) : ptr{ ptr_ } {
            CV_Assert(start != end);
            CV_Assert(std::distance(start, end) <= CSL_MAX_TENSOR_RANK);

            using ItrValueType = typename std::iterator_traits<ForwardItr>::value_type;
            if (std::any_of(start, end, [](ItrValueType x) { return x <= 0; })) {
                CV_Error(Error::StsBadArg, "the given shape contains negative or zero size");
            }

            shape.assign(start, end);
        }

        /** creates a subview of a tensor (or span or view); refer to subview method for more details */
        template <class... Args>
        TensorView(TensorView other, size_type offset, Args&&... args) noexcept
            : TensorView(other.subview(offset, std::forward<Args>(args)...)) { }

        TensorView& operator=(const TensorView&) = default;
        TensorView& operator=(TensorSpan<T> other) noexcept {
            TensorView tmp(other);
            swap(*this, tmp);
            return *this;
        }

        /** returns true if the view is empty */
        bool empty() const noexcept { return shape.size() == 0; }

        /** returns the total number of elements in the view
         *
         * Pre-conditions:
         * - view must be non-empty
         */
        size_type size() const noexcept {
            CV_Assert(!empty());
            return std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<size_type>());
        }

        /** returns the rank of the view
         *
         * Pre-conditions:
         * - view must be non-empty
         */
        size_type rank() const noexcept {
            CV_Assert(!empty());
            return shape.size();
        }

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
         * - view must be non-empty
         * - the axis must be in the range [-rank(), rank())
         */
        size_type get_axis_size(int axis) const noexcept {
            axis = clamp_axis(axis, rank());
            CV_Assert(axis >= 0 && axis < rank());
            return shape[axis];
        }

        /** @brief returns the combined size of the axes in an axis range
         *
         * if the shape is [3 x 5 x 7 x 11]
         * - `size_range(0, 2)` will return 3 x 5 = 15
         * - `size_range(1, 3)` will return 5 x 7 = 35
         * - `size_range(0, 4)` will return 3 x 5 x 7 x 11 = 1155
         *
         * Pre-conditions:
         * - view must be non-empty
         * - `axis_start` must be less than or equal to `axis_end`
         * - `axis_end` must be less than or equal to the rank
         *
         * returns one if the two `axis_start` and `axis_end` are equal
         */
        size_type size_range(size_type axis_start, size_type axis_end) const noexcept {
            CV_Assert(!empty());
            CV_Assert(axis_start <= axis_end);
            CV_Assert(axis_end <= rank());
            auto start = std::begin(shape) + axis_start;
            auto end = std::begin(shape) + axis_end;
            return std::accumulate(start, end, 1, std::multiplies<size_type>());
        }

        /** returns an std::vector containing axis lengths starting from axis zero
         *
         * Pre-conditions:
         * - view must be non-empty
         *
         * Exception Guarantee: Strong
         */
        std::vector<size_type> shape_as_vector() const {
            CV_Assert(!empty());
            return std::vector<size_type>(std::begin(shape), std::end(shape));
        }

        /** returns a device pointer to immutable device memory */
        const_pointer get() const noexcept { return ptr; }

        /** @brief reshapes the view
         *
         * Length deduction:
         * The length of at most one axis can be deduced using the total size constraint. The axis can
         * be marked for deduction by specifying the size as -1.
         *
         * The axes for which no size was provided (excluding -1) will be assumed to be one.
         *
         * Pre-conditions:
         * - view must be non-empty
         * - [start, end) represents a forward range containing length of the axes in order starting from axis zero
         * - the number of axis lengths must be less than or equal to the tensor rank
         * - at most one axis length is allowed for length deduction
         * - the lengths provided must ensure that the total number of elements remains unchanged
         *
         * Exception Guarantee: Strong
         */
        template <class ForwardItr>
        typename std::enable_if<!std::is_integral<ForwardItr>::value, void>
        ::type reshape(ForwardItr start, ForwardItr end) {
            CV_Assert(start != end);
            CV_Assert(std::distance(start, end) <= rank());

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
            std::fill(std::begin(shape), std::end(shape), 1);
            std::copy_backward(start, end, std::end(shape));

            /* replace the unknown axis with the correct value */
            std::replace(std::begin(shape), std::end(shape), size_type(-1), unknown_size);
        }

        /** @brief reshapes the view
         * constructs a range out of the arguments and invokes the range-based reshape method
         */
        template <class ...Sizes>
        void reshape(Sizes... new_sizes_) {
            static_assert(sizeof...(Sizes) <= CSL_MAX_TENSOR_RANK, "required rank exceeds maximum supported rank");
            static_assert(sizeof...(Sizes) > 0, "no sizes provided");
            std::array<std::int64_t, sizeof...(Sizes)> new_sizes = { static_cast<std::int64_t>(new_sizes_)... };
            reshape(std::begin(new_sizes), std::end(new_sizes));
        }

        /** @brief reshapes the view
         *
         * Pre-conditions:
         * - the reference tensor/span/view must be non-empty
         * - the reference tensor/span/view's rank must be less than or equal to the rank of the view
         *
         * Exception Guarantee: Strong
         */
        template <class TensorType>
        void reshape_as(const TensorType& tensor) {
            CV_Assert(!tensor.empty());
            cxx_utils::resizable_static_array<size_type, CSL_MAX_TENSOR_RANK> new_sizes(tensor.rank());
            for (int i = 0; i < new_sizes.size(); i++)
                new_sizes[i] = tensor.get_axis_size(i);
            reshape(std::begin(new_sizes), std::end(new_sizes));
        }

        /** @brief squeezes the tensor
         *
         * removes all axes of unit size
         *
         * Pre-conditions:
         * - the view must be non-empty
         * - the view's rank must be at least two
         *
         * Exception Guarantee: Strong
         */
        void squeeze() {
            CV_Assert(!empty());
            CV_Assert(rank() >= 2);
            auto itr = std::remove(std::begin(shape), std::end(shape), 1);
            shape.resize(itr - std::begin(shape));
        }

        /** @brief squeezes the tensor
         *
         * removes the specified axis if the axis length is one; otherwise, ignores the request
         *
         * Pre-conditions:
         * - the view must be non-empty
         * - the view's rank must be at least two
         *
         * Exception Guarantee: Strong
         */
        void squeeze(int axis) {
            CV_Assert(!empty());
            CV_Assert(rank() >= 2);
            axis = clamp_axis(axis, rank());
            CV_Assert(axis >= 0 && axis < rank());
            shape.erase(std::begin(shape) + axis);
        }

        /** @brief squeezes the tensor
         *
         * removes leading singleton axes until the tensor's rank is equal to the requested rank
         *
         * Pre-conditions:
         * - the tensor must be non-empty
         * - the tensor's rank must be at least two
         * - the tensor's rank must be at least the requested rank
         * - the tensor must be squeezable up to the requested rank
         *
         * Exception Guarantee: Strong
         */
        void squeeze_to(int r) {
            CV_Assert(!empty());
            CV_Assert(rank() >= r);
            CV_Assert(std::all_of(std::begin(shape), std::end(shape) - r, [](size_type x){ return x == 1; }));
            std::copy(std::end(shape) - r, std::end(shape), std::begin(shape));
            shape.resize(r);
        }

        /** @brief unsqueezes the tensor
         *
         * adds a axis of unit size at the requested before the specified axis
         *
         * Pre-conditions:
         * - the view must be non-empty
         * - the view's rank must be less than the maximum supported rank (CSL_MAX_TENSOR_RANK)
         *
         * Exception Guarantee: Strong
         */
        void unsqueeze(int axis = 0) {
            CV_Assert(!empty());
            CV_Assert(rank() < CSL_MAX_TENSOR_RANK);
            axis = clamp_axis(axis, rank());
            CV_Assert(axis >= 0 && axis < rank());
            shape.insert(std::begin(shape) + axis, 1);
        }

        /** @brief obtains a subview of the view
         *
         * The axes for which no size was provided will be assumed to be one.
         *
         * Pre-conditions:
         * - the view must be non-empty
         * - the `offset` must be less than the size of the view
         * - [start, end) represents a forward range containing length of the subview axes in order
         * - the number of axis lengths provided must be less than or equal to the tensor rank
         * - the lengths provided must ensure that the number of elements does not exceed (old size - offset)
         *
         * Exception Guarantee: Strong
         */
        template <class ForwardItr>
        typename std::enable_if<cxx_utils::is_forward_iterator<ForwardItr>::value, TensorView>
        ::type subview(size_type offset, ForwardItr start, ForwardItr end) const {
            CV_Assert(start != end);
            CV_Assert(std::distance(start, end) <= rank());

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
            temp.shape.assign(start, end);
            temp.ptr = ptr + offset;
            return temp;
        }

        /** @brief obtains a subview of the view
         * constructs a range out of the size arguments and invokes the range-based subview method
         */
        template <class ...Sizes>
        TensorView subview(size_type offset, Sizes... new_sizes_) const {
            static_assert(sizeof...(Sizes) <= CSL_MAX_TENSOR_RANK, "required rank exceeds maximum supported rank");
            static_assert(sizeof...(Sizes) > 0, "no sizes provided");
            std::array<std::int64_t, sizeof...(Sizes)> new_sizes = { static_cast<std::int64_t>(new_sizes_)... };
            return subview(offset, std::begin(new_sizes), std::end(new_sizes));
        }

        operator View<T>() const noexcept { return View<T>(ptr, size()); }

        friend void swap(TensorView& lhs, TensorView& rhs) noexcept {
            using std::swap;
            swap(lhs.ptr, rhs.ptr);
            swap(lhs.shape, rhs.shape);
        }

    private:
        cxx_utils::resizable_static_array<size_type, CSL_MAX_TENSOR_RANK> shape;
        const_pointer ptr;
    };

    /** returns true if the two TensorType objects have the same shape */
    template <class TensorType1, class TensorType2>
    bool is_shape_same(const TensorType1& x, const TensorType2& y) noexcept {
        auto rank1 = x.rank();
        auto rank2 = y.rank();

        if (rank1 != rank2)
            return false;

        for (int i = 0; i < rank1; i++)
            if (x.get_axis_size(i) != y.get_axis_size(i))
                return false;
        return true;
    }

    /** returns true if the two TensorType objects are compatible */
    template <class TensorType1, class TensorType2>
    bool is_shape_compatible(const TensorType1& x, const TensorType2& y) noexcept {
        const auto rank1 = x.rank();
        const auto rank2 = y.rank();

        /* mathematically not required but is a technically required */
        if (rank1 != rank2)
            return false;

        for (int i = 0; i < rank1; i++)
            if (x.get_axis_size(i) != y.get_axis_size(i) &&
                x.get_axis_size(i) != 1 && y.get_axis_size(i) != 1)
                return false;
        return true;
    }

    /** returns the rank to which the given tensor can be squeezed to */
    template <class TensorType>
    std::size_t get_effective_rank(const TensorType& x) noexcept {
        const auto rank = x.rank();
        auto effective_rank = rank;
        for (int i = 0; i < rank; i++, effective_rank--)
            if (x.get_axis_size(i) != 1)
                break;
        return effective_rank;
    }

}}}} /* namespace cv::dnn::cuda4dnn::csl */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_CSL_TENSOR_HPP */
