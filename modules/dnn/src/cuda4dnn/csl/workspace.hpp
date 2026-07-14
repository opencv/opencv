// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_CSL_WORKSPACE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_CSL_WORKSPACE_HPP

#include "pointer.hpp"
#include "span.hpp"
#include "tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <iterator>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl {

    /** @brief maintains a single block of reusable device memory
     *
     * Each Workspace object is intended to be used by a single entity at a time but by
     * different entities at different times. It maintains a single reusable block of memory which
     * is sufficient for the largest consumer.
     */
    class Workspace {
    public:

        /** @brief reserve \p bytes of memory */
        void require(std::size_t bytes) {
            if (bytes > ptr.size())
                ptr.reset(bytes);
        }

        /** @brief number of bytes reserved by the largest consumer */
        std::size_t size() const noexcept {
            return ptr.size();
        }

        /** @brief returns the pointer to the workspace memory */
        DevicePtr<unsigned char> get() {
            return ptr.get();
        }

    private:
        ManagedPtr<unsigned char> ptr;
    };

    /** used to compute total workspace size from several workspace requests */
    class WorkspaceBuilder {
    public:
        WorkspaceBuilder() noexcept : max_size_in_bytes{ 0 } { }

        /** request memory for \p count number of elements of the type \tparam T */
        template <class T = std::int8_t>
        void require(std::size_t count) noexcept {
            auto blocks256 = (count * sizeof(T) + 255) / 256;
            max_size_in_bytes += blocks256 * 256;
        }

        /** returns the total workspace memory that is required */
        std::size_t required_workspace_size() const noexcept { return max_size_in_bytes; }

    private:
        std::size_t max_size_in_bytes;
    };

    /** general memory block from a workspace which can be passed on to the requester */
    class WorkspaceInstance {
    public:

        /** returns a device pointer to the workspace memory */
        template <class T = void>
        DevicePtr<T> get() const noexcept {
            return static_cast<DevicePtr<T>>(ptr);
        }

        /** returnss the size of the workspace memory in bytes */
        std::size_t size_in_bytes() const noexcept {
            return size_in_bytes_;
        }

        /** creates a Span<T> of \p count elements from the workspace memory */
        template <class T>
        Span<T> get_span(std::size_t count = 0) const {
            if (count == 0)
                count = size_in_bytes_ / sizeof(T);

            if (count * sizeof(T) > size_in_bytes_)
                CV_Error(Error::StsNoMem, "memory not sufficient");

            return Span<T>(static_cast<DevicePtr<T>>(ptr), count);
        }

        /** creates a TensorSpan<T> of the given shape from the workspace memory */
        template <class T, class ForwardItr>
        TensorSpan<T> get_tensor_span(ForwardItr shape_begin, ForwardItr shape_end) const {
            using ItrValueType = typename std::iterator_traits<ForwardItr>::value_type;
            auto required_size = std::accumulate(shape_begin, shape_end, 1, std::multiplies<ItrValueType>());
            if (required_size * sizeof(T) > size_in_bytes_)
                CV_Error(Error::StsNoMem, "memory not sufficient");
            return TensorSpan<T>(static_cast<DevicePtr<T>>(ptr), shape_begin, shape_end);
        }

    private:
        DevicePtr<void> ptr;
        std::size_t size_in_bytes_;

        friend class WorkspaceAllocator;
        WorkspaceInstance(DevicePtr<void> ptr_, std::size_t size_in_bytes__)
            : ptr{ ptr_ }, size_in_bytes_{ size_in_bytes__ } { }
    };

    /** used to split a single workspace into constituents */
    class WorkspaceAllocator {
    public:
        WorkspaceAllocator() = default;
        WorkspaceAllocator(Workspace& workspace) noexcept
            : current{ workspace.get() }, bytes_remaining { workspace.size() }
        {
            CV_Assert(is_aligned<void>(current, 256));
            CV_Assert(bytes_remaining % 256 == 0);
        }

        /** allocates a Span<T> of \p count elements from the workspace memory */
        template <class T>
        Span<T> get_span(std::size_t count = 0) {
            return accquire<T>(count);
        }

        /** allocates a TensorSpan<T> of the given shape from the workspace memory */
        template <class T, class ForwardItr>
        TensorSpan<T> get_tensor_span(ForwardItr start, ForwardItr end) {
            using ItrValueType = typename std::iterator_traits<ForwardItr>::value_type;
            auto required_size = std::accumulate(start, end, 1, std::multiplies<ItrValueType>());
            return TensorSpan<T>(accquire<T>(required_size).data(), start, end);
        }

        /** allocates a WorkspaceInstance of size \p bytes from the workspace memory */
        WorkspaceInstance get_instance(std::size_t bytes = 0) {
            auto span = accquire(bytes);
            return WorkspaceInstance(DevicePtr<void>(span.data()), span.size());
        }

    private:
        template <class T = std::int8_t>
        Span<T> accquire(std::size_t count = 0) {
            auto ptr = current;

            if (count == 0)
                count = bytes_remaining / sizeof(T);

            auto blocks256 = (count * sizeof(T) + 255) / 256;
            if (bytes_remaining < blocks256 * 256)
                CV_Error(Error::StsNoMem, "out of workspace memory");

            bytes_remaining -= blocks256 * 256;
            current = static_cast<DevicePtr<std::int8_t>>(current) + blocks256 * 256;
            return Span<T>(static_cast<DevicePtr<T>>(ptr), count);
        }

        DevicePtr<void> current;
        std::size_t bytes_remaining;
    };

}}}} /* namespace cv::dnn::cuda4dnn::csl */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_CSL_WORKSPACE_HPP */
