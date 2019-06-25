// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_CUDNN_LRN_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_CUDNN_LRN_HPP

#include "cudnn.h"
#include "../pointer.hpp"

#include <opencv2/core.hpp>

#include <cudnn.h>

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace cudnn {

    class LRNDescriptor {
    public:
        enum class lrn_type {
            ACROSS_CHANNELS
        };

        LRNDescriptor() noexcept : descriptor{ nullptr } { }
        LRNDescriptor(const LRNDescriptor&) = delete;
        LRNDescriptor(LRNDescriptor&& other) noexcept
            : descriptor{ other.descriptor } {
            other.descriptor = nullptr;
        }

        LRNDescriptor(std::size_t local_size, double alpha, double beta, double k, lrn_type type)
        {
            constructor(local_size, alpha, beta, k, type);
        }

        ~LRNDescriptor() noexcept {
            if (descriptor != nullptr) {
                /* cudnnDestroyLRNDescriptor will not fail */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroyLRNDescriptor(descriptor));
            }
        }

        LRNDescriptor& operator=(const LRNDescriptor&) = delete;
        LRNDescriptor& operator=(LRNDescriptor&& other) noexcept {
            descriptor = other.descriptor;
            other.descriptor = nullptr;
            return *this;
        };

        cudnnLRNDescriptor_t get() const noexcept { return descriptor; }

    private:
        void constructor(std::size_t local_size, double alpha, double beta, double k, lrn_type type) {
            auto get_lrn_type = [] (lrn_type type) {
                switch (type) {
                case lrn_type::ACROSS_CHANNELS:
                    return CUDNN_LRN_CROSS_CHANNEL_DIM1;
                }
                CV_Error(Error::StsBadArg, "unknown LRN type");
            };
            mode = get_lrn_type(type);

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
                /* cudnnDestroyLRNDescriptor will not fail */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroyLRNDescriptor(descriptor));
                throw;
            }
        }

        cudnnLRNDescriptor_t descriptor;
        cudnnLRNMode_t mode;
    };

    template <class T> inline
    void LRNForward(
        const Handle& handle,
        const LRNDescriptor& lrnDesc,
        const TensorDescriptor<T>& inputDesc,
        DevicePtr<const T> inputPtr,
        T alpha, T beta,
        const TensorDescriptor<T>& outputDesc,
        DevicePtr<T> outputPtr)
    {
        CUDA4DNN_CHECK_CUDNN(
            cudnnLRNCrossChannelForward(
                HandleAccessor::get(handle),
                lrnDesc.get(), CUDNN_LRN_CROSS_CHANNEL_DIM1,
                &alpha, inputDesc.get(), inputPtr.get(),
                &beta, outputDesc.get(), outputPtr.get()
            )
        );
    }

}}}}} /* namespace cv::dnn::cuda4dnn::csl::cudnn */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_CUDNN_LRN_HPP */
