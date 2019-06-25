// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_CUDNN_CUDNN_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_CUDNN_CUDNN_HPP

#include <opencv2/dnn/csl/cudnn.hpp>

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

    namespace detail {
        inline void check(cudnnStatus_t status, const char* func, const char* file, int line) {
            if (status != CUDNN_STATUS_SUCCESS)
                throw cuDNNException(Error::GpuApiCallError, cudnnGetErrorString(status), func, file, line);
        }

        /** get_data_type<T> returns the equivalent cudnn enumeration constant for type T */
        template <class> auto get_data_type()->decltype(CUDNN_DATA_FLOAT);
        template <> inline auto get_data_type<float>()->decltype(CUDNN_DATA_FLOAT) { return CUDNN_DATA_FLOAT; }
        template <> inline auto get_data_type<double>()->decltype(CUDNN_DATA_FLOAT) { return CUDNN_DATA_DOUBLE; }
    }

    /** used to access the raw cuDNN handle held by Handle */
    class HandleAccessor {
    public:
        static cudnnHandle_t get(const Handle& handle);
    };

    /** creates a cuDNN tensor descriptor for a given shape */
    template <class T>
    class TensorDescriptor {
    public:
        TensorDescriptor() noexcept : descriptor{ nullptr } { }
        TensorDescriptor(const TensorDescriptor&) = delete;
        TensorDescriptor(TensorDescriptor&& other) noexcept
            : descriptor{ other.descriptor } {
            other.descriptor = nullptr;
        }

        /** constructs a tensor descriptor from the axis lengths provided in \p shape */
        template <class SequenceContainer, typename = decltype(std::begin(std::declval<SequenceContainer>()))>
        TensorDescriptor(const SequenceContainer& shape) {
            constructor(shape.begin(), shape.end());
        }

        /** constructs a tensor descriptor from the axis lengths provided in [begin, end) */
        template <class ForwardItr, typename = typename std::enable_if<!std::is_integral<ForwardItr>::value, void>::type> // TODO is_iterator
        TensorDescriptor(ForwardItr begin, ForwardItr end) {
            constructor(begin, end);
        }

        /** constructs a tensor descriptor from the axis lengths provided as arguments */
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
                const auto rank = std::distance(start, end);
                if (rank <= 4) {
                    std::array<int, 4> dims;
                    std::fill(std::begin(dims), std::end(dims), 1);

                    /* suppose we have a 3d tensor, the first axis is the batch axis and
                     * the second axis is the channel axis (generally)
                     *
                     * cuDNN frequently assumes that the first axis is the batch axis and the
                     * second axis is the channel axis; hence, we copy the shape of a lower rank
                     * tensor to the begining of `dims`
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

    /** @brief element-wise addition with broadcasting
     *
     * \f$ C = \alpha A + \beta C \f$
     *
     * @tparam          T           matrix element type (must be `float` or `double`)
     *
     * @param           handle      valid cuDNN handle
     * @param           alpha       scale factor for A
     * @param           aDesc       tensor descriptor for A
     * @param[in]       A           pointer to tensor in device memory
     * @param           beta        scale factor for C
     * @param           cDesc       tensor descriptor for C
     * @param[in]       C           pointer to tensor in device memory
     *
     * Exception Guarantee: Basic
     */
    template <class T>
    typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value, void>
    ::type add(const Handle& handle,
        T alpha, const TensorDescriptor<T>& aDesc, DevicePtr<const T> A,
        T beta, const TensorDescriptor<T>& cDesc, DevicePtr<T> C)
    {
        CUDA4DNN_CHECK_CUDNN(
            cudnnAddTensor(HandleAccessor::get(handle),
                &alpha, aDesc.get(), A.get(),
                &beta, cDesc.get(), C.get()
            )
        );
    }

}}}}} /* namespace cv::dnn::cuda4dnn::csl::cudnn */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_CUDNN_HPP */
