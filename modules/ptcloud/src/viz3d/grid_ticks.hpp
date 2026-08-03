// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_PTCLOUD_VIZ3D_GRID_TICKS_HPP
#define OPENCV_PTCLOUD_VIZ3D_GRID_TICKS_HPP

namespace cv { namespace viz3d { namespace detail {

// Grid line spacing snapped so dist_scale/tick_step is in [2,4]; GL-free, always terminates.
inline float gridTickStep(float dist_scale)
{
    float tick_step = 1.0f;
    while (dist_scale / tick_step > 4.0f)
        tick_step *= 2.0f;
    while (tick_step > 1e-6f && dist_scale / tick_step < 2.0f)
        tick_step *= 0.5f;
    return tick_step;
}

}}} // namespace cv::viz3d::detail

#endif
