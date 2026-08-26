// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

// target_fps <= 0 must show zero measurable overhead vs. not passing the parameter at all --
// that path bypasses fpsControlGrab() entirely (see VideoCapture::grab()). Enabling it, by
// contrast, makes grab() fully decode (and Mat::clone()) every source frame rather than just
// the ones actually returned, so throughput is expected to be measurably worse than disabled
// even though fewer frames come back to the caller -- this perf test exists to catch a
// regression in either direction (disabled path picking up overhead it shouldn't, or the
// enabled path's overhead growing unexpectedly), not to demonstrate a speedup.

typedef tuple<string, double> VideoCapture_TargetFpsParams;
typedef perf::TestBaseWithParam<VideoCapture_TargetFpsParams> VideoCapture_TargetFps;

PERF_TEST_P(VideoCapture_TargetFps, ReadThroughput,
            testing::Combine(testing::Values(string("highgui/video/big_buck_bunny.mp4")),
                              testing::Values(0.0, 12.0))) // 0 = disabled; 12 = half native 24fps
{
    const string filename = getDataPath(get<0>(GetParam()));
    const double target_fps = get<1>(GetParam());

    TEST_CYCLE()
    {
        VideoCapture cap(filename, CAP_ANY, target_fps);
        Mat frame;
        while (cap.read(frame)) {}
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
