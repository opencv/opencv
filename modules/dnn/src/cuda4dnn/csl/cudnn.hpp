// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_CUDNN_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_CUDNN_HPP

#include <opencv2/dnn/csl/cudnn.hpp>

#include <cudnn.h>

#define CUDA4DNN_CHECK_CUDNN(call) \
    ::cv::dnn::cuda4dnn::csl::cudnn::detail::check((call), CV_Func, __FILE__, __LINE__)

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace cudnn {

    namespace detail {
        inline void check(cudnnStatus_t status, const char* func, const char* file, int line) {
            if (status != CUDNN_STATUS_SUCCESS)
                throw cuDNNException(Error::GpuApiCallError, cudnnGetErrorString(status), func, file, line);
        }
    }

}}}}} /* namespace cv::dnn::cuda4dnn::csl::cudnn */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_CUDNN_HPP */
