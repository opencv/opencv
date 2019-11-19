// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_RENDER_PRIV_HPP
#define OPENCV_RENDER_PRIV_HPP

#include <opencv2/gapi/render.hpp>

namespace cv
{
namespace gapi
{
namespace wip
{
namespace draw
{

// FIXME only for tests
GAPI_EXPORTS void BGR2NV12(const cv::Mat& bgr, cv::Mat& y_plane, cv::Mat& uv_plane);

} // namespace draw
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // OPENCV_RENDER_PRIV_HPP
