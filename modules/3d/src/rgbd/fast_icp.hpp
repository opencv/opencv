// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Partially rewritten from https://github.com/Nerei/kinfu_remake
// Copyright(c) 2012, Anatoly Baksheev. All rights reserved.

#ifndef OPENCV_3D_FAST_ICP_HPP
#define OPENCV_3D_FAST_ICP_HPP

#include "../precomp.hpp"
#include "utils.hpp"

namespace cv {
namespace kinfu {

class ICP
{
public:
    ICP(const cv::Matx33f _intrinsics, const std::vector<int> &_iterations, float _angleThreshold, float _distanceThreshold);

    virtual bool estimateTransform(cv::Affine3f& transform,
                                   InputArray oldPoints, InputArray oldNormals,
                                   InputArray newPoints, InputArray newNormals,
                                   InputArray oldPointsMask, InputArray oldNormalsMask,
                                   InputArray newPointsMask, InputArray newNormalsMask
                                   ) const = 0;
    virtual ~ICP() { }

protected:

    std::vector<int> iterations;
    float angleThreshold;
    float distanceThreshold;
    cv::Intr intrinsics;
};

cv::Ptr<ICP> makeICP(const cv::Intr _intrinsics, const std::vector<int> &_iterations,
                     float _angleThreshold, float _distanceThreshold);

} // namespace kinfu
} // namespace cv

#endif // include guard
