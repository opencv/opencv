// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

#define PERF_QOI
#define PERF_PNG
#undef  PERF_AVIF
#undef  PERF_WEBP

typedef TestBaseWithParam<tuple<ImreadModes,Size,String>> CodecsCommon;

PERF_TEST_P_(CodecsCommon, Decode)
{
    ImreadModes immode = get<0>(GetParam());
    Size dstSize = get<1>(GetParam());
    String codecExt = get<2>(GetParam());

    String filename = getDataPath("perf/2560x1600.png");
    cv::Mat src = imread(filename, immode);
    cv::Mat dst;
    cv::resize(src, dst, dstSize);

    vector<uchar> buf;
    imencode(codecExt.c_str(), dst, buf);

    declare.in(buf).out(dst);

    TEST_CYCLE() imdecode(buf, immode);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(CodecsCommon, Encode)
{
    ImreadModes immode = get<0>(GetParam());
    Size dstSize = get<1>(GetParam());
    String codecExt = get<2>(GetParam());

    String filename = getDataPath("perf/2560x1600.png");
    cv::Mat src = imread(filename, immode);
    cv::Mat dst;
    cv::resize(src, dst, dstSize);

    vector<uchar> buf;
    imencode(codecExt.c_str(), dst, buf); // To recode datasize
    declare.in(dst).out(buf);

    TEST_CYCLE() imencode(codecExt.c_str(), dst, buf);

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/* */,
    CodecsCommon,
    ::testing::Combine(
        ::testing::Values( ImreadModes(IMREAD_COLOR) ,IMREAD_GRAYSCALE),
        ::testing::Values(
            Size(640,480),
            Size(1920,1080),
            Size(3840,2160)
        ),
        ::testing::Values(
#if defined( HAVE_PNG ) && defined( PERF_PNG )
            ".png",
#endif
#if defined( HAVE_QOI ) && defined( PERF_QOI )
            ".qoi",
#endif
#if defined( HAVE_AVIF ) && defined( PERF_AVIF )
            ".avif",
#endif
#if defined( HAVE_WEBP ) && defined( PERF_WEBP )
             ".webp",
#endif
            ".bmp"
        )
    )
);

} // namespace
