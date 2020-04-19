// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_CUDNN_POOLING_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_CUDNN_POOLING_HPP

#include "cudnn.hpp"

#include "../pointer.hpp"

#include <opencv2/core.hpp>

#include <cudnn.h>

#include <cstddef>
#include <array>
#include <algorithm>
#include <vector>
#include <type_traits>
#include <iterator>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace cudnn {

    class PoolingDescriptor {
    public:
        enum class PoolingType {
            MAX,
            MAX_DETERMINISTIC,
            AVERAGE_EXCLUDE_PADDING,
            AVERAGE_INCLUDE_PADDING
        };

        PoolingDescriptor() noexcept : descriptor{ nullptr } { }
        PoolingDescriptor(const PoolingDescriptor&) = delete;
        PoolingDescriptor(PoolingDescriptor&& other) noexcept
            : descriptor{ other.descriptor } {
            other.descriptor = nullptr;
        }

        /** constructs a pooling descriptor
         *
         * Pre-conditions:
         * - \p window_size, \p padding and \p stride must have the same size
         *
         * The length of the containers is interpreted as the order of the pooling operation.
         *
         * Exception Guarantee: Basic
         */
        template <class SequenceContainer, typename = decltype(std::begin(std::declval<SequenceContainer>()))>
        PoolingDescriptor(
            const SequenceContainer& window_size,
            const SequenceContainer& padding,
            const SequenceContainer& stride,
            PoolingType type)
        {
            constructor(window_size, padding, stride, type);
        }

        ~PoolingDescriptor() noexcept {
            if (descriptor != nullptr) {
                /* cudnnDestroyPoolingDescriptor will not fail for a valid descriptor */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroyPoolingDescriptor(descriptor));
            }
        }

        PoolingDescriptor& operator=(const PoolingDescriptor&) = delete;
        PoolingDescriptor& operator=(PoolingDescriptor&& other) noexcept {
            descriptor = other.descriptor;
            other.descriptor = nullptr;
            return *this;
        };

        cudnnPoolingDescriptor_t get() const noexcept { return descriptor; }

    private:
        template <class SequenceContainer>
        void constructor(
            const SequenceContainer& window_size,
            const SequenceContainer& padding,
            const SequenceContainer& stride,
            PoolingType type)
        {
            CV_Assert(window_size.size() == padding.size());
            CV_Assert(window_size.size() == stride.size());

            auto get_pooling_type = [] (PoolingType type) {
                switch (type) {
                case PoolingType::MAX:
                    return CUDNN_POOLING_MAX;
                case PoolingType::MAX_DETERMINISTIC:
                    return CUDNN_POOLING_MAX_DETERMINISTIC;
                case PoolingType::AVERAGE_EXCLUDE_PADDING:
                    return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
                case PoolingType::AVERAGE_INCLUDE_PADDING:
                    return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
                }
                CV_Error(Error::StsBadArg, "unknown pooling type");
            };

            CUDA4DNN_CHECK_CUDNN(cudnnCreatePoolingDescriptor(&descriptor));
            try {
                const auto rank = window_size.size();
                if (rank == 2) {
                    CUDA4DNN_CHECK_CUDNN(
                        cudnnSetPooling2dDescriptor(
                            descriptor,
                            get_pooling_type(type), CUDNN_PROPAGATE_NAN,
                            window_size[0], window_size[1],
                            padding[0], padding[1],
                            stride[0], stride[1]
                        )
                    );
                } else {
                    std::vector<int> iwindow_size(std::begin(window_size), std::end(window_size));
                    std::vector<int> ipadding(std::begin(padding), std::end(padding));
                    std::vector<int> istride(std::begin(stride), std::end(stride));
                    CUDA4DNN_CHECK_CUDNN(
                        cudnnSetPoolingNdDescriptor(
                            descriptor,
                            get_pooling_type(type), CUDNN_PROPAGATE_NAN,
                            rank, iwindow_size.data(), ipadding.data(), istride.data()
                        )
                    );
                }
            } catch (...) {
                /* cudnnDestroyPoolingDescriptor will not fail for a valid descriptor */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroyPoolingDescriptor(descriptor));
                throw;
            }
        }

        cudnnPoolingDescriptor_t descriptor;
    };

    /** gives the shape of the output tensor after pooling
     *
     * @note it's not required to enforce the this shape in the output tensor; slightly different shapes will work
     *
     * Exception Guarantee: Basic
     */
    template <class T> inline
    void getPoolingForwardOutputDim(
        const PoolingDescriptor& poolingDesc,
        const TensorDescriptor<T>& inputDesc,
        std::vector<int>& output_dim)
    {
        output_dim.clear();
        output_dim.resize(CUDNN_DIM_MAX); /* we use `output_dim` to hold temporaries */

        std::vector<int> temp(CUDNN_DIM_MAX);
        cudnnDataType_t tempDataType;
        CUDA4DNN_CHECK_CUDNN(
            cudnnGetTensorNdDescriptor(
                inputDesc.get(),
                CUDNN_DIM_MAX + 1, /* according to docs, this is what we do to get the rank */
                &tempDataType,
                output_dim.data(),
                temp.data(),
                temp.data()
            )
        );

        const auto rank = output_dim[0];
        output_dim.resize(rank);
        CUDA4DNN_CHECK_CUDNN(
            cudnnGetPoolingNdForwardOutputDim(poolingDesc.get(), inputDesc.get(), rank, output_dim.data())
        );
    }

    /** @brief performs pooling operation
     *
     * dstValue = alpha * result + beta * priorDstValue
     *
     * @tparam          T           pooling element type (must be `half` or `float`)
     *
     * @param           handle      valid cuDNN Handle
     * @param           poolingDesc pooling description
     * @param           inputDesc   tensor descriptor describing the input
     * @param[in]       inputPtr    pointer to input tensor in device memory
     * @param           alpha       result scale factor
     * @param           beta        previous value scale factor
     * @param           outputDesc  tensor descriptor describing the output
     * @param[out]      outputPtr   pointer to output tensor in device memory
     *
     * Exception Guarantee: Basic
     */
    template <class T>
    void pool(
        const Handle& handle,
        const PoolingDescriptor& poolingDesc,
        const TensorDescriptor<T>& inputDesc,
        const DevicePtr<const T> inputPtr,
        T alpha, T beta,
        const TensorDescriptor<T>& outputDesc,
        DevicePtr<T> outputPtr)
    {
        CV_Assert(handle);

        CUDA4DNN_CHECK_CUDNN(
            cudnnPoolingForward(
                handle.get(),
                poolingDesc.get(),
                &alpha, inputDesc.get(), inputPtr.get(),
                &beta, outputDesc.get(), outputPtr.get()
            )
        );
    }

    template <> inline
    void pool(
        const Handle& handle,
        const PoolingDescriptor& poolingDesc,
        const TensorDescriptor<half>& inputDesc,
        const DevicePtr<const half> inputPtr,
        half alpha, half beta,
        const TensorDescriptor<half>& outputDesc,
        DevicePtr<half> outputPtr)
    {
        CV_Assert(handle);

        /* we specalize for fp16 as the scaling factors must be provided as `float` */
        float alpha_ = alpha, beta_ = beta;
        CUDA4DNN_CHECK_CUDNN(
            cudnnPoolingForward(
                handle.get(),
                poolingDesc.get(),
                &alpha_, inputDesc.get(), inputPtr.get(),
                &beta_, outputDesc.get(), outputPtr.get()
            )
        );
    }

}}}}} /* namespace cv::dnn::cuda4dnn::csl::cudnn */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_CUDNN_POOLING_HPP */
