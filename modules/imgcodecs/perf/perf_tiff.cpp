// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "perf_precomp.hpp"

namespace opencv_test
{

using namespace perf;

#define TYPICAL_MAT_SIZES_TIFF_DECODE  cv::Size(128, 128), cv::Size(256, 256), cv::Size(512, 512), cv::Size(640, 480), cv::Size(1024, 1024), cv::Size(1280, 720), cv::Size(1920, 1080)
#define TYPICAL_MAT_TYPES_TIFF_DECODE  CV_8UC1, CV_8UC3, CV_16UC1, CV_16UC3
#define TYPICAL_MATS_TIFF_DECODE       testing::Combine( testing::Values( TYPICAL_MAT_SIZES_TIFF_DECODE), testing::Values( TYPICAL_MAT_TYPES_TIFF_DECODE) )

PERF_TEST_P(Size_MatType, tiffDecode, TYPICAL_MATS_TIFF_DECODE)
{
    cv::Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());

    cv::Mat src = Mat(sz, type);
    cv::Mat dst = Mat(sz, type);
    cv::randu(src, 0, (CV_MAT_DEPTH(type) == CV_16U) ? 65535 : 255);
    std::vector<uchar> buf;
    cv::imencode(".tiff", src, buf);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE()
    {
      cv::imdecode(buf, cv::IMREAD_UNCHANGED, &dst);
    }

    SANITY_CHECK(dst);
}

}
