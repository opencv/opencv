// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "perf_precomp.hpp"

namespace opencv_test
{

#ifdef HAVE_JPEG

using namespace perf;

PERF_TEST(JPEG, Decode)
{
    String filename = getDataPath("stitching/boat1.jpg");

    FILE *f = fopen(filename.c_str(), "rb");
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    vector<uchar> file_buf((size_t)len);
    EXPECT_EQ(len, (long)fread(&file_buf[0], 1, (size_t)len, f));
    fclose(f); f = NULL;

    TEST_CYCLE() imdecode(file_buf, IMREAD_UNCHANGED);

    SANITY_CHECK_NOTHING();
}

PERF_TEST(JPEG, Encode)
{
    String filename = getDataPath("stitching/boat1.jpg");
    cv::Mat src = imread(filename);

    vector<uchar> buf;
    TEST_CYCLE() imencode(".jpg", src, buf);

    SANITY_CHECK_NOTHING();
}

#endif // HAVE_JPEG

} // namespace