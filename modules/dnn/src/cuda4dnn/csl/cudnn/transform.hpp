// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_CUDNN_TRANSFORM_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_CUDNN_TRANSFORM_HPP

#include "../pointer.hpp"

#include "cudnn.hpp"

#include <cudnn.h>
#include <vector>
#include <type_traits>
#include <iterator>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace cudnn {

    /** describes a tensor transform operation
     *
     * Supported transformations:
     * - add or remove asymmetric padding
     */
    class TensorTransformDescriptor {
    public:
        TensorTransformDescriptor() noexcept : descriptor{ nullptr } { }
        TensorTransformDescriptor(const TensorTransformDescriptor&) = delete;
        TensorTransformDescriptor(TensorTransformDescriptor&& other) noexcept
            : descriptor{ other.descriptor } {
            other.descriptor = nullptr;
        }

        /** constructs a convolution descriptor
         *
         * Pre-conditions:
         * - \p padding_left and \p padding_right must have the same size
         *
         * The length of the containers is interpreted as the rank of the tensors which will be given.
         *
         * @note \p padding_left and \p padding_right may have negative values to remove padding
         *
         * Exception Guarantee: Basic
         */
        template <class SequenceContainer, typename = decltype(std::begin(std::declval<SequenceContainer>()))>
        TensorTransformDescriptor(
            const SequenceContainer& padding_left,
            const SequenceContainer& padding_right)
        {
            constructor(padding_left, padding_right);
        }

        ~TensorTransformDescriptor() noexcept {
            if (descriptor != nullptr) {
                /* cudnnDestroyTensorTransformDescriptor will not fail for a valid descriptor */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroyTensorTransformDescriptor(descriptor));
            }
        }

        TensorTransformDescriptor& operator=(const TensorTransformDescriptor&) = delete;
        TensorTransformDescriptor& operator=(TensorTransformDescriptor&& other) noexcept {
            descriptor = other.descriptor;
            other.descriptor = nullptr;
            return *this;
        };

        cudnnTensorTransformDescriptor_t get() const noexcept { return descriptor; }

    private:
        template <class SequenceContainer>
        void constructor(
            const SequenceContainer& padding_left,
            const SequenceContainer& padding_right
        )
        {
            CV_Assert(padding_left.size() == padding_right.size());

            auto ipadding_left  = std::vector<int32_t>(std::begin(padding_left), std::end(padding_left));
            auto ipadding_right = std::vector<int32_t>(std::begin(padding_right), std::end(padding_right));
            CUDA4DNN_CHECK_CUDNN(cudnnCreateTensorTransformDescriptor(&descriptor));
            try {
                CUDA4DNN_CHECK_CUDNN(
                    cudnnSetTensorTransformDescriptor(
                        descriptor,
                        ipadding_left.size(), CUDNN_TENSOR_NCHW,
                        ipadding_left.data(), ipadding_right.data(),
                        NULL, CUDNN_TRANSFORM_FOLD
                    )
                );
            } catch (...) {
                /* cudnnDestroyTensorTransformDescriptor will not fail for a valid descriptor */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroyTensorTransformDescriptor(descriptor));
                throw;
            }
        }

        cudnnTensorTransformDescriptor_t descriptor;
    };

    template <class T>
    void transform(
        const Handle& handle,
        const TensorTransformDescriptor& transDesc,
        const TensorDescriptor<T>& inputDesc,
        DevicePtr<const T> inputPtr,
        const TensorDescriptor<T>& outputDesc,
        DevicePtr<T> outputPtr)
    {
        T alpha = 1.0, beta = 0.0;
        CUDA4DNN_CHECK_CUDNN(
            cudnnTransformTensorEx(
                handle.get(),
                transDesc.get(),
                &alpha, inputDesc.get(), inputPtr.get(),
                &beta, outputDesc.get(), outputPtr.get()
            )
        );
    }

    template <> inline
    void transform(
        const Handle& handle,
        const TensorTransformDescriptor& transDesc,
        const TensorDescriptor<half>& inputDesc,
        DevicePtr<const half> inputPtr,
        const TensorDescriptor<half>& outputDesc,
        DevicePtr<half> outputPtr)
    {
        /* we specalize for fp16 as the scaling factors must be provided as `float` */
        float alpha = 1.0, beta = 0.0;
        CUDA4DNN_CHECK_CUDNN(
            cudnnTransformTensorEx(
                handle.get(),
                transDesc.get(),
                &alpha, inputDesc.get(), inputPtr.get(),
                &beta, outputDesc.get(), outputPtr.get()
            )
        );
    }

}}}}} /* namespace cv::dnn::cuda4dnn::csl::cudnn */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_CUDNN_TRANSFORM_HPP */
