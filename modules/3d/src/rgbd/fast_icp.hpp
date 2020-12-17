// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#ifndef __OPENCV_KINFU_FAST_ICP_H__
#define __OPENCV_KINFU_FAST_ICP_H__

#include "precomp.hpp"
#include "kinfu_frame.hpp"

namespace cv {
namespace kinfu {

class ICP
{
public:
    ICP(const cv::kinfu::Intr _intrinsics, const std::vector<int> &_iterations, float _angleThreshold, float _distanceThreshold);

    virtual bool estimateTransform(cv::Affine3f& transform,
                                   InputArray oldPoints, InputArray oldNormals,
                                   InputArray newPoints, InputArray newNormals
                                   ) const = 0;
    virtual ~ICP() { }

protected:

    std::vector<int> iterations;
    float angleThreshold;
    float distanceThreshold;
    cv::kinfu::Intr intrinsics;
};

cv::Ptr<ICP> makeICP(const cv::kinfu::Intr _intrinsics, const std::vector<int> &_iterations,
                     float _angleThreshold, float _distanceThreshold);

} // namespace kinfu
} // namespace cv
#endif
