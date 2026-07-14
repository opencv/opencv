// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_DNN_BACKEND_HPP
#define OPENCV_DNN_BACKEND_HPP

#include <memory>
#include <map>

namespace cv { namespace dnn_backend {

using namespace cv::dnn;

class CV_EXPORTS NetworkBackend
{
public:
    virtual ~NetworkBackend();

    virtual void switchBackend(Net& net) = 0;

    /**
    @param loaderID use empty "" for auto
    @param model see cv::dnn::readNetwork
    @param config see cv::dnn::readNetwork
    */
    virtual Net readNetwork(const std::string& loaderID, const std::string& model, const std::string& config) = 0;

    /** @overload */
    virtual Net readNetwork(
        const std::string& loaderID,
        const uchar* bufferModelConfigPtr, size_t bufferModelConfigSize,
        const uchar* bufferWeightsPtr, size_t bufferWeightsSize
    ) = 0;

    // TODO: target as string + configuration
    virtual bool checkTarget(Target target) = 0;
};


}  // namespace dnn_backend
}  // namespace cv

#endif // OPENCV_DNN_BACKEND_HPP
