// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_STREAM_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_STREAM_HPP

#include <opencv2/dnn/csl/stream.hpp>

#include <cuda_runtime_api.h>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl {

    /** used to access the raw CUDA stream handle held by Handle */
    class StreamAccessor {
    public:
        static cudaStream_t get(const Stream& stream);
    };

}}}} /* cv::dnn::cuda4dnn::csl */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_STREAM_HPP */
