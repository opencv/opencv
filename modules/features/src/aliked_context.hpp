// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_FEATURES_ALIKED_CONTEXT_HPP
#define OPENCV_FEATURES_ALIKED_CONTEXT_HPP

#include "opencv2/features.hpp"

namespace cv
{

struct ALIKEDContext
{
    Mat normalizedKeypoints;  // Nx2 float, coordinates in [-1, 1]
    Size imageSize;
};

}  // namespace cv

#endif
