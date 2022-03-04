// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_CUDNN_ACTIVATION_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_CUDNN_ACTIVATION_HPP

#include <cudnn.h>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace cudnn {

    class ActivationDescriptor {
    public:
        enum class ActivationType {
            IDENTITY,
            RELU,
            CLIPPED_RELU,
            TANH,
            SIGMOID,
            ELU
        };

        ActivationDescriptor() noexcept : descriptor{ nullptr } { }
        ActivationDescriptor(const ActivationDescriptor&) = delete;
        ActivationDescriptor(ActivationDescriptor&& other) noexcept
            : descriptor{ other.descriptor } {
            other.descriptor = nullptr;
        }

        /* `relu_ceiling_or_elu_alpha`:
         * - `alpha` coefficient in ELU activation
         * - `ceiling` for CLIPPED_RELU activation
         */
        ActivationDescriptor(ActivationType type, double relu_ceiling_or_elu_alpha = 0.0) {
            CUDA4DNN_CHECK_CUDNN(cudnnCreateActivationDescriptor(&descriptor));
            try {
                const auto mode = [type] {
                    switch(type) {
                        case ActivationType::IDENTITY: return CUDNN_ACTIVATION_IDENTITY;
                        case ActivationType::RELU: return CUDNN_ACTIVATION_RELU;
                        case ActivationType::CLIPPED_RELU: return CUDNN_ACTIVATION_CLIPPED_RELU;
                        case ActivationType::SIGMOID: return CUDNN_ACTIVATION_SIGMOID;
                        case ActivationType::TANH: return CUDNN_ACTIVATION_TANH;
                        case ActivationType::ELU: return CUDNN_ACTIVATION_ELU;
                    }
                    CV_Assert(0);
                    return CUDNN_ACTIVATION_IDENTITY;
                } ();

                CUDA4DNN_CHECK_CUDNN(cudnnSetActivationDescriptor(descriptor, mode, CUDNN_NOT_PROPAGATE_NAN, relu_ceiling_or_elu_alpha));
            } catch(...) {
                /* cudnnDestroyActivationDescriptor will not fail for a valid descriptor object */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroyActivationDescriptor(descriptor));
                throw;
            }
        }

        ~ActivationDescriptor() noexcept {
            if (descriptor != nullptr) {
                /* cudnnDestroyActivationDescriptor will not fail */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroyActivationDescriptor(descriptor));
            }
        }

        ActivationDescriptor& operator=(const ActivationDescriptor&) = delete;
        ActivationDescriptor& operator=(ActivationDescriptor&& other) noexcept {
            descriptor = other.descriptor;
            other.descriptor = nullptr;
            return *this;
        };

        cudnnActivationDescriptor_t get() const noexcept { return descriptor; }

    private:
        cudnnActivationDescriptor_t descriptor;
    };

}}}}} /* namespace cv::dnn::cuda4dnn::csl::cudnn */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_CUDNN_ACTIVATION_HPP */
