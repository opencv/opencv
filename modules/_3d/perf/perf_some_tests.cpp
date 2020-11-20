// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test
{
namespace
{

PERF_TEST(_3DPerfTest, EmptyTest)
{
    TEST_CYCLE() cv::_3d::doNothing();

    SANITY_CHECK(true);
}

} // namespace
} // namespace opencv_test
