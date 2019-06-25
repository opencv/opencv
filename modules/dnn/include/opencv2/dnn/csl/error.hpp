// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CSL_ERROR_HPP
#define OPENCV_DNN_CSL_ERROR_HPP

#include <opencv2/core.hpp>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl {

    /** @brief exception class for errors thrown by the CUDA APIs */
    class CUDAException : public cv::Exception {
    public:
        using cv::Exception::Exception;
    };

}}}} /* namespace cv::dnn::cuda4dnn::csl */

#endif /* OPENCV_DNN_CSL_ERROR_HPP */
