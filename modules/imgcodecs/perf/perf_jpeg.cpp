// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "perf_precomp.hpp"

namespace opencv_test
{
    using namespace perf;

    typedef perf::TestBaseWithParam<std::string> JPEG;

    PERF_TEST(JPEG_perf, ReadFile)
    {
        String filename = getDataPath("cv/face/100032540_1.jpg");

        TEST_CYCLE() imread(filename);

        SANITY_CHECK_NOTHING();
    }

} // namespace
