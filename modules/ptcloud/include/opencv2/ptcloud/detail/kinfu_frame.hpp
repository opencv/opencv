// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_3D_DETAIL_KINFU_FRAME_HPP
#define OPENCV_3D_DETAIL_KINFU_FRAME_HPP

#include <opencv2/core/affine.hpp>

namespace cv {
namespace detail {

CV_EXPORTS void renderPointsNormals(InputArray _points, InputArray _normals, OutputArray image, cv::Vec3f lightLoc);
CV_EXPORTS void renderPointsNormalsColors(InputArray _points, InputArray _normals, InputArray _colors, OutputArray image);

} // namespace detail
} // namespace cv

#endif // include guard
