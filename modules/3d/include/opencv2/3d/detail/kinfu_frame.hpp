// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#ifndef __OPENCV_KINFU_FRAME_H__
#define __OPENCV_KINFU_FRAME_H__

#include <opencv2/core/affine.hpp>

namespace cv {
namespace detail {

CV_EXPORTS_W void renderPointsNormals(InputArray _points, InputArray _normals, OutputArray image, cv::Vec3f lightLoc);
CV_EXPORTS_W void renderPointsNormalsColors(InputArray _points, InputArray _normals, InputArray _colors, OutputArray image);

} // namespace detail
} // namespace cv
#endif
