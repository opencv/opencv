// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

// To skip test, undef it.
#undef HAVE_AVIF
#undef HAVE_WEBP

typedef TestBaseWithParam<tuple<ImreadModes,Size,string>> CodecsCommon;

PERF_TEST_P_(CodecsCommon, Decode)
{
    ImreadModes immode = get<0>(GetParam());
    Size dstSize = get<1>(GetParam());
    string codecExt = get<2>(GetParam());

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
    string codecExt = get<2>(GetParam());

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

const string all_formats[] =
{
#ifdef HAVE_PNG
    ".png",
#endif
#ifdef HAVE_QOI
    ".qoi",
#endif
#ifdef HAVE_AVIF
    ".avif",
#endif
#ifdef HAVE_WEBP
    ".webp",
#endif
    ".bmp"
};

const string all_formats_tmp[] = { ".bmp" };

INSTANTIATE_TEST_CASE_P(/* */,
    CodecsCommon,
    ::testing::Combine(
        ::testing::Values( ImreadModes(IMREAD_COLOR) ,IMREAD_GRAYSCALE),
        ::testing::Values(
            Size(640,480),
            Size(1920,1080),
            Size(3840,2160)
        ),
        ::testing::ValuesIn( all_formats )
    )
);

} // namespace
