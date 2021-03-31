// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <opencv2/gapi/stereo.hpp>

namespace cv { namespace gapi {

GMat stereo(const GMat& left, const GMat& right,
            const cv::gapi::StereoOutputFormat of)
{
    return calib3d::GStereo::on(left, right, of);
}

} // namespace cv
} // namespace gapi
