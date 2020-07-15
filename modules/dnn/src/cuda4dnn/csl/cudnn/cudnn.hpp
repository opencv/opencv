// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_CUDNN_CUDNN_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_CUDNN_CUDNN_HPP

#include "../pointer.hpp"

#include <cudnn.h>

#include <cstddef>
#include <array>
#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>
#include <type_traits>
#include <iterator>

#define CUDA4DNN_CHECK_CUDNN(call) \
    ::cv::dnn::cuda4dnn::csl::cudnn::detail::check((call), CV_Func, __FILE__, __LINE__)

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace cudnn {

    /** @brief exception class for errors thrown by the cuDNN API */
    class cuDNNException : public CUDAException {
    public:
        cuDNNException(cudnnStatus_t code, const std::string& msg, const std::string& func, const std::string& file, int line)
            : CUDAException(Error::GpuApiCallError, msg, func, file, line), cudnnError{code}
        {
        }

        cudnnStatus_t getCUDNNStatus() const noexcept { return cudnnError; }

    private:
        cudnnStatus_t cudnnError;
    };

    namespace detail {
        inline void check(cudnnStatus_t status, const char* func, const char* file, int line) {
            if (status != CUDNN_STATUS_SUCCESS)
                throw cuDNNException(status, cudnnGetErrorString(status), func, file, line);
        }

        /** get_data_type<T> returns the equivalent cudnn enumeration constant for type T */
        using cudnn_data_enum_type = decltype(CUDNN_DATA_FLOAT);
        template <class> cudnn_data_enum_type get_data_type();
        template <> inline cudnn_data_enum_type get_data_type<half>() { return CUDNN_DATA_HALF; }
        template <> inline cudnn_data_enum_type get_data_type<float>() { return CUDNN_DATA_FLOAT; }
    }

    /** @brief noncopyable cuDNN smart handle
     *
     * UniqueHandle is a smart non-sharable wrapper for cuDNN handle which ensures that the handle
     * is destroyed after use.
     */
    class UniqueHandle {
    public:
        UniqueHandle() noexcept : handle{ nullptr } { }
        UniqueHandle(UniqueHandle&) = delete;
        UniqueHandle(UniqueHandle&& other) noexcept {
            stream = std::move(other.stream);
            handle = other.handle;
            other.handle = nullptr;
        }

        /** creates a cuDNN handle and associates it with the stream specified
         *
         * Exception Guarantee: Basic
         */
        UniqueHandle(Stream strm) : stream(std::move(strm)) {
            CV_Assert(stream);
            CUDA4DNN_CHECK_CUDNN(cudnnCreate(&handle));
            try {
                CUDA4DNN_CHECK_CUDNN(cudnnSetStream(handle, stream.get()));
            } catch (...) {
                /* cudnnDestroy won't throw if a valid handle is passed */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroy(handle));
                throw;
            }
        }

        ~UniqueHandle() noexcept {
            if (handle != nullptr) {
                /* cudnnDestroy won't throw if a valid handle is passed */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroy(handle));
            }
        }

        UniqueHandle& operator=(const UniqueHandle&) = delete;
        UniqueHandle& operator=(UniqueHandle&& other) noexcept {
            CV_Assert(other);
            if (&other != this) {
                UniqueHandle(std::move(*this)); /* destroy current handle */
                stream = std::move(other.stream);
                handle = other.handle;
                other.handle = nullptr;
            }
            return *this;
        }

        /** returns the raw cuDNN handle */
        cudnnHandle_t get() const noexcept {
            CV_Assert(handle);
            return handle;
        }

        /** returns true if the handle is valid */
        explicit operator bool() const noexcept { return static_cast<bool>(handle); }

    private:
        Stream stream;
        cudnnHandle_t handle;
    };

    /** @brief sharable cuDNN smart handle
     *
     * Handle is a smart sharable wrapper for cuDNN handle which ensures that the handle
     * is destroyed after all references to the handle are destroyed. The handle must always
     * be associated with a non-default stream. The stream must be specified during construction.
     *
     * @note Moving a Handle object to another invalidates the former
     */
    class Handle {
    public:
        Handle() = default;
        Handle(const Handle&) = default;
        Handle(Handle&&) = default;

        /** creates a cuDNN handle and associates it with the stream specified
         *
         * Exception Guarantee: Basic
         */
        Handle(Stream strm) : handle(std::make_shared<UniqueHandle>(std::move(strm))) { }

        Handle& operator=(const Handle&) = default;
        Handle& operator=(Handle&&) = default;

        /** returns true if the handle is valid */
        explicit operator bool() const noexcept { return static_cast<bool>(handle); }

        /** returns the raw cuDNN handle */
        cudnnHandle_t get() const noexcept {
            CV_Assert(handle);
            return handle->get();
        }

    private:
        std::shared_ptr<UniqueHandle> handle;
    };

    /** describe a tensor
     *
     * @tparam  T   type of elements in the tensor
     */
    template <class T>
    class TensorDescriptor {
    public:
        TensorDescriptor() noexcept : descriptor{ nullptr } { }
        TensorDescriptor(const TensorDescriptor&) = delete;
        TensorDescriptor(TensorDescriptor&& other) noexcept
            : descriptor{ other.descriptor } {
            other.descriptor = nullptr;
        }

        /** constructs a tensor descriptor from the axis lengths provided in \p shape
         *
         * Exception Guarantee: Basic
         */
        template <class SequenceContainer, typename = decltype(std::begin(std::declval<SequenceContainer>()))>
        TensorDescriptor(const SequenceContainer& shape) {
            constructor(shape.begin(), shape.end());
        }

        /** constructs a tensor descriptor from the axis lengths provided in [begin, end)
         *
         * Exception Guarantee: Basic
         */
        template <class ForwardItr, typename = typename std::enable_if<!std::is_integral<ForwardItr>::value, void>::type> // TODO is_iterator
        TensorDescriptor(ForwardItr begin, ForwardItr end) {
            constructor(begin, end);
        }

        /** constructs a tensor descriptor from the axis lengths provided as arguments
         *
         * Exception Guarantee: Basic
         */
        template <class ...Sizes>
        TensorDescriptor(Sizes ...sizes) {
            static_assert(sizeof...(Sizes) <= CUDNN_DIM_MAX, "required rank exceeds maximum supported rank");
            std::array<int, sizeof...(Sizes)> dims = { static_cast<int>(sizes)... };
            constructor(std::begin(dims), std::end(dims));
        }

        ~TensorDescriptor() noexcept {
            if (descriptor != nullptr) {
                /* cudnnDestroyTensorDescriptor will not fail */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroyTensorDescriptor(descriptor));
            }
        }

        TensorDescriptor& operator=(const TensorDescriptor&) = delete;
        TensorDescriptor& operator=(TensorDescriptor&& other) noexcept {
            descriptor = other.descriptor;
            other.descriptor = nullptr;
            return *this;
        };

        cudnnTensorDescriptor_t get() const noexcept { return descriptor; }

    private:
        template <class ForwardItr>
        void constructor(ForwardItr start, ForwardItr end) {
            CV_Assert(start != end);
            CV_Assert(std::distance(start, end) <= CUDNN_DIM_MAX);

            CUDA4DNN_CHECK_CUDNN(cudnnCreateTensorDescriptor(&descriptor));
            try {
                /* cuDNN documentation recommends using the 4d tensor API whenever possible
                 * hence, we create a 4d tensor descriptors for 3d tensor
                 */
                const auto rank = std::distance(start, end);
                if (rank <= 4) {
                    std::array<int, 4> dims;
                    std::fill(std::begin(dims), std::end(dims), 1);

                    /* suppose we have a 3d tensor, the first axis is the batch axis and
                     * the second axis is the channel axis (generally)
                     *
                     * cuDNN frequently assumes that the first axis is the batch axis and the
                     * second axis is the channel axis; hence, we copy the shape of a lower rank
                     * tensor to the beginning of `dims`
                     */
                    std::copy(start, end, std::begin(dims));

                    CUDA4DNN_CHECK_CUDNN(
                        cudnnSetTensor4dDescriptor(descriptor,
                            CUDNN_TENSOR_NCHW, detail::get_data_type<T>(),
                            dims[0], dims[1], dims[2], dims[3]
                        )
                    );
                } else {
                    std::vector<int> stride(rank);
                    stride.back() = 1;
                    /* WHAT WE HAVE NOW:
                     * stride[-1] = 1
                     * stride[-2] = garbage
                     * stride[-3] = garbage
                     * stride[-4] = garbage
                     * ...
                     */

                    std::copy(start + 1, end, stride.begin());
                    /* WHAT WE HAVE NOW:
                     * stride[-1] = 1
                     * stride[-2] = dim[-1]
                     * stride[-3] = dim[-2]
                     * stride[-4] = dim[-3]
                     * ...
                     */

                    std::partial_sum(stride.rbegin(), stride.rend(), stride.rbegin(), std::multiplies<int>());
                    /* WHAT WE HAVE NOW:
                     * stride[-1] = 1
                     * stride[-2] = stride[-1] * dim[-1]
                     * stride[-3] = stride[-2] * dim[-2]
                     * stride[-4] = stride[-3] * dim[-3]
                     * ...
                     */

                    std::vector<int> dims(start, end);
                    CUDA4DNN_CHECK_CUDNN(
                        cudnnSetTensorNdDescriptor(descriptor,
                            detail::get_data_type<T>(), rank,
                            dims.data(), stride.data()
                        )
                    );
                }
            } catch (...) {
                /* cudnnDestroyTensorDescriptor will not fail */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroyTensorDescriptor(descriptor));
                throw;
            }
        }

        cudnnTensorDescriptor_t descriptor;
    };

}}}}} /* namespace cv::dnn::cuda4dnn::csl::cudnn */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_CUDNN_HPP */
