// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {
using namespace perf;
typedef TestBaseWithParam<tuple<int, int> > FarnebackGaussianSIMD;
PERF_TEST_P(FarnebackGaussianSIMD, gaussianAndBoxControl,
            testing::Combine(testing::Values(63, 64, 65, 320, 640), testing::Values(0, OPTFLOW_FARNEBACK_GAUSSIAN)))
{
    const int width = get<0>(GetParam()), flags = get<1>(GetParam());
    Mat src(240, width + 13, CV_8U), secondFrame, flow;
    RNG rng(30388); rng.fill(src, RNG::UNIFORM, 0, 256);
    Mat roi = src(Rect(3, 0, width, 240));
    Mat transform = Mat_<double>({2, 3}, {1., 0., 1., 0., 1., -1.});
    warpAffine(roi, secondFrame, transform, roi.size(), INTER_LINEAR, BORDER_REFLECT_101);
    declare.in(roi, secondFrame);
    TEST_CYCLE() calcOpticalFlowFarneback(roi, secondFrame, flow, .5, 3, 15, 3, 5, 1.1, flags);
    SANITY_CHECK_NOTHING();
}
} // namespace
