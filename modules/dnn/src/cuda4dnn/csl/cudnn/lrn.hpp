// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_CUDNN_LRN_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_CUDNN_LRN_HPP

#include "cudnn.hpp"

#include "../pointer.hpp"
#include "../workspace.hpp"

#include <opencv2/core.hpp>

#include <cudnn.h>

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace cudnn {

    class LRNDescriptor {
    public:
        enum class LRNType {
            ACROSS_CHANNELS,
            WITHIN_CHANNEL
        };

        LRNDescriptor() noexcept : descriptor{ nullptr } { }
        LRNDescriptor(const LRNDescriptor&) = delete;
        LRNDescriptor(LRNDescriptor&& other) noexcept
            : descriptor{ other.descriptor }, type{ other.type } {
            other.descriptor = nullptr;
        }

        /** sets up a LRN descriptor
         *
         * @param local_size    size of the normalization window
         * @param alpha         variance scaling parameter
         * @param beta          power parameter
         * @param k             bias parameter
         *
         * @note \p alpha is divided by the window width in across channels mode
         * @note \p alpha is divided by the (window width)^spatialDimensions in within channel mode
         *
         * @note the \p alpha, \p beta and \p k will be type casted to the tensor datatype during operation
         *
         * Exception Guarantee: Basic
         */
        LRNDescriptor(std::size_t local_size, double alpha, double beta, double k, LRNType type_) {
            constructor(local_size, alpha, beta, k, type_);
        }

        ~LRNDescriptor() noexcept {
            if (descriptor != nullptr) {
                /* cudnnDestroyLRNDescriptor will not fail for a valid descriptor */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroyLRNDescriptor(descriptor));
            }
        }

        LRNDescriptor& operator=(const LRNDescriptor&) = delete;
        LRNDescriptor& operator=(LRNDescriptor&& other) noexcept {
            descriptor = other.descriptor;
            type = other.type;
            other.descriptor = nullptr;
            return *this;
        };

        cudnnLRNDescriptor_t get() const noexcept { return descriptor; }
        LRNType getType() const noexcept { return type; }

    private:
        void constructor(std::size_t local_size, double alpha, double beta, double k, LRNType type_) {
            CV_Assert(CUDNN_LRN_MIN_N <= local_size && local_size <= CUDNN_LRN_MAX_N);

            type = type_;

            CUDA4DNN_CHECK_CUDNN(cudnnCreateLRNDescriptor(&descriptor));
            try {
                CUDA4DNN_CHECK_CUDNN(
                    cudnnSetLRNDescriptor(
                        descriptor,
                        local_size,
                        alpha,
                        beta,
                        k
                    )
               );
            } catch (...) {
                /* cudnnDestroyLRNDescriptor will not fail for a valid descriptor */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroyLRNDescriptor(descriptor));
                throw;
            }
        }

        cudnnLRNDescriptor_t descriptor;
        LRNType type;
    };

    /** @brief performs local response normalization
     *
     * dstValue = alpha * result + beta * priorDstValue
     *
     * @tparam          T           element type (must be `half` or `float`)
     *
     * @param           handle      valid cuDNN Handle
     * @param           lrnDesc     LRN description
     * @param           inputDesc   tensor descriptor describing the input
     * @param[in]       inputPtr    pointer to input tensor in device memory
     * @param           alpha       result scale factor
     * @param           beta        previous value scale factor
     * @param           outputDesc  tensor descriptor describing the output
     * @param[out]      outputPtr   pointer to output tensor in device memory
     * @param           workspace   workspace memory which meets the requirements of \p convAlgo
     *
     * Exception Guarantee: Basic
     */
    template <class T>
    void LRNForward(
        const Handle& handle,
        const LRNDescriptor& lrnDesc,
        const TensorDescriptor<T>& inputDesc,
        DevicePtr<const T> inputPtr,
        T alpha, T beta,
        const TensorDescriptor<T>& outputDesc,
        DevicePtr<T> outputPtr,
        WorkspaceInstance workspace)
    {
        CV_Assert(handle);

        if (lrnDesc.getType() == LRNDescriptor::LRNType::ACROSS_CHANNELS) {
            CUDA4DNN_CHECK_CUDNN(
                cudnnLRNCrossChannelForward(
                    handle.get(),
                    lrnDesc.get(), CUDNN_LRN_CROSS_CHANNEL_DIM1,
                    &alpha, inputDesc.get(), inputPtr.get(),
                    &beta, outputDesc.get(), outputPtr.get()
                )
            );
        } else if (lrnDesc.getType() == LRNDescriptor::LRNType::WITHIN_CHANNEL) {
            std::size_t size;
            CUDA4DNN_CHECK_CUDNN(cudnnGetTensorSizeInBytes(inputDesc.get(), &size));

            DevicePtr<void> temp1 = workspace.get_span<half>(size).data();
            DevicePtr<void> temp2 = workspace.get_span<half>(size).data();

            CUDA4DNN_CHECK_CUDNN(
                cudnnDivisiveNormalizationForward(
                    handle.get(),
                    lrnDesc.get(), CUDNN_DIVNORM_PRECOMPUTED_MEANS,
                    &alpha, inputDesc.get(), inputPtr.get(),
                    NULL,
                    static_cast<void*>(temp1), static_cast<void*>(temp2),
                    &beta, outputDesc.get(), outputPtr.get()
                )
            );
        }
    }

    template <> inline
    void LRNForward(
       const Handle& handle,
       const LRNDescriptor& lrnDesc,
       const TensorDescriptor<half>& inputDesc,
       DevicePtr<const half> inputPtr,
       half alpha, half beta,
       const TensorDescriptor<half>& outputDesc,
       DevicePtr<half> outputPtr,
        WorkspaceInstance workspace)
    {
        CV_Assert(handle);

        /* we specalize for fp16 as the scaling factors must be provided as `float` */
        float alpha_ = alpha, beta_ = beta;
        if (lrnDesc.getType() == LRNDescriptor::LRNType::ACROSS_CHANNELS) {
            CUDA4DNN_CHECK_CUDNN(
                cudnnLRNCrossChannelForward(
                    handle.get(),
                    lrnDesc.get(), CUDNN_LRN_CROSS_CHANNEL_DIM1,
                    &alpha_, inputDesc.get(), inputPtr.get(),
                    &beta_, outputDesc.get(), outputPtr.get()
                )
            );
        } else if (lrnDesc.getType() == LRNDescriptor::LRNType::WITHIN_CHANNEL) {
            std::size_t size;
            CUDA4DNN_CHECK_CUDNN(cudnnGetTensorSizeInBytes(inputDesc.get(), &size));

            DevicePtr<void> temp1 = workspace.get_span<half>(size).data();
            DevicePtr<void> temp2 = workspace.get_span<half>(size).data();

            CUDA4DNN_CHECK_CUDNN(
                cudnnDivisiveNormalizationForward(
                    handle.get(),
                    lrnDesc.get(), CUDNN_DIVNORM_PRECOMPUTED_MEANS,
                    &alpha_, inputDesc.get(), inputPtr.get(),
                    NULL,
                    static_cast<void*>(temp1), static_cast<void*>(temp2),
                    &beta_, outputDesc.get(), outputPtr.get()
                )
            );
        }
    }

}}}}} /* namespace cv::dnn::cuda4dnn::csl::cudnn */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_CUDNN_LRN_HPP */
