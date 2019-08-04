// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_CXX_UTILS_MAKE_UNIQUE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_CXX_UTILS_MAKE_UNIQUE_HPP

#include <memory>

namespace cv { namespace dnn { namespace cuda4dnn { namespace cxx_utils {

    template<class T, typename... Args>
    std::unique_ptr<T> make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }

}}}} /* cv::dnn::cuda4dnn::csl::cxx_utils */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_CXX_UTILS_MAKE_UNIQUE_HPP */
