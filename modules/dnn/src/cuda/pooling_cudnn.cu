// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <float.h>
#include "layer_cudnn.hpp"
#include <cudnn.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>

namespace cv { namespace dnn { namespace cuda {

void pool(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t xDesc,
    cudnnTensorDescriptor_t yDesc,
    cudnnPoolingDescriptor_t pDesc,
    const void* d_input,
    void* d_output)
{
    {
        detail::CudnnScalar alpha{}, beta{};
        detail::setCudnnScalars(yDesc, alpha, beta, 1.0f, 0.0f);
        cudnnPoolingForward(handle, pDesc,
                            detail::scalarPtr(alpha), xDesc, d_input,
                            detail::scalarPtr(beta), yDesc, d_output);
        return;
    }
}

}}} // namespace cv::dnn::cuda
