// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#ifndef OPENCV_GAPI_GRENDEROCV_HPP
#define OPENCV_GAPI_GRENDEROCV_HPP

#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include "api/render_priv.hpp"
#include "api/ft_render.hpp"

namespace cv
{
namespace gapi
{
namespace render
{
namespace ocv
{

GAPI_EXPORTS cv::gapi::GBackend backend();

} // namespace ocv
} // namespace render
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_GRENDEROCV_HPP
