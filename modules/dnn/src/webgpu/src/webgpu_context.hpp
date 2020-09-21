// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_WEBGPU_CONTEXT_HPP
#define OPENCV_DNN_WEBGPU_CONTEXT_HPP

namespace cv { namespace dnn { namespace webgpu {
#ifdef HAVE_WEBGPU
class Context
{

public:
    Context();
    ~Context();
};

void createContext();

#endif  // HAVE_WEBGPU

}}}  // namespace cv::dnn::webgpu

#endif // OPENCV_DNN_WEBGPU_CONTEXT_HPP
