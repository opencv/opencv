// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

#ifdef HAVE_QOI

PERF_TEST(QOI, Decode)
{
    String filename = getDataPath("perf/2560x1600.png");
    cv::Mat src = imread(filename);

    vector<uchar> buf;
    imencode(".qoi", src, buf);

    TEST_CYCLE() imdecode(buf, IMREAD_COLOR);

    SANITY_CHECK_NOTHING();
}

PERF_TEST(QOI, Encode)
{
    String filename = getDataPath("perf/2560x1600.png");
    cv::Mat src = imread(filename);

    vector<uchar> buf;
    TEST_CYCLE() imencode(".qoi", src, buf);

    SANITY_CHECK_NOTHING();
}

#endif

} // namespace
