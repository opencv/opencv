// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_CSL_ERROR_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_CSL_ERROR_HPP

#include <opencv2/core.hpp>

#include <cuda_runtime_api.h>

#define CUDA4DNN_CHECK_CUDA(call) \
    ::cv::dnn::cuda4dnn::csl::detail::check((call), CV_Func, __FILE__, __LINE__)

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl {
    /** @brief exception class for errors thrown by the CUDA APIs */
    class CUDAException : public cv::Exception {
    public:
        using cv::Exception::Exception;
    };

    namespace detail {
        inline void check(cudaError_t err, const char* func, const char* file, int line) {
            if (err != cudaSuccess)
                throw CUDAException(Error::GpuApiCallError, cudaGetErrorString(err), func, file, line);
        }
    }
}}}} /* namespace cv::dnn::cuda4dnn::csl */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_CSL_ERROR_HPP */
