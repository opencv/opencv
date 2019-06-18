// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_WORKSPACE_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_WORKSPACE_HPP

#include "pointer.hpp"

#include <opencv2/dnn/csl/workspace.hpp>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl {

    /** used to access the raw CUDA stream handle held by Handle */
    class WorkspaceAccessor {
    public:
        static DevicePtr<unsigned char> get(const Workspace& workspace);
    };

}}}} /* cv::dnn::cuda4dnn::csl */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_WORKSPACE_HPP */
