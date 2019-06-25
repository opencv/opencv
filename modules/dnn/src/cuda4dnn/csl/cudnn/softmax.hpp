// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_CUDNN_SOFTMAX_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_CUDNN_SOFTMAX_HPP

#include "cudnn.h"
#include "../pointer.hpp"

#include <cudnn.h>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace cudnn {

    /** @brief computes softmax (or log softmax)
     *
     * @tparam          T           matrix element type (must be `float` or `double`)
     *
     * @param           handle      valid cuDNN handle
     * @param           outputDesc  tensor descriptor for A
     * @param[out]      output      pointer to tensor in device memory
     * @param           inputDesc   tensor descriptor for C
     * @param[in]       input       pointer to tensor in device memory
     * @param           log         apply log on probabilities
     *
     * Exception Guarantee: Basic
     */
    template <class T>
    typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value, void>
    ::type softmax(const cudnn::Handle& handle,
        const TensorDescriptor<T>& outputDesc, DevicePtr<T> output,
        const TensorDescriptor<T>& inputDesc, DevicePtr<const T> input,
        bool log)
    {
        T alpha = 1.0, beta = 0.0;
        cudnnSoftmaxAlgorithm_t algo = log ? CUDNN_SOFTMAX_LOG : CUDNN_SOFTMAX_ACCURATE;
        CUDA4DNN_CHECK_CUDNN(
            cudnnSoftmaxForward(
                HandleAccessor::get(handle),
                algo, CUDNN_SOFTMAX_MODE_CHANNEL,
                &alpha, inputDesc.get(), input.get(),
                &beta, outputDesc.get(), output.get()
            )
        );
    }

}}}}} /* namespace cv::dnn::cuda4dnn::csl::cudnn */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_CUDNN_SOFTMAX_HPP */
