// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "perf_precomp.hpp"

namespace opencv_test
{

#ifdef HAVE_PNG

using namespace perf;

typedef perf::TestBaseWithParam<std::string> Decode;
typedef perf::TestBaseWithParam<std::string> Encode;

const string exts[] = {
#ifdef HAVE_AVIF
    ".avif",
#endif
    ".bmp",
#ifdef HAVE_IMGCODEC_GIF
    ".gif",
#endif
#if (defined(HAVE_JASPER) && defined(OPENCV_IMGCODECS_ENABLE_JASPER_TESTS)) \
    || defined(HAVE_OPENJPEG)
    ".jp2",
#endif
#ifdef HAVE_JPEG
    ".jpg",
#endif
#ifdef HAVE_JPEGXL
    ".jxl",
#endif
    ".png",
#ifdef HAVE_IMGCODEC_PXM
    ".ppm",
#endif
#ifdef HAVE_IMGCODEC_SUNRASTER
    ".ras",
#endif
#ifdef HAVE_TIFF
    ".tiff",
#endif
#ifdef HAVE_WEBP
    ".webp",
#endif
};

const string exts_multi[] = {
#ifdef HAVE_AVIF
    ".avif",
#endif
#ifdef HAVE_IMGCODEC_GIF
    ".gif",
#endif
    ".png",
#ifdef HAVE_TIFF
    ".tiff",
#endif
#ifdef HAVE_WEBP
    ".webp",
#endif
};

PERF_TEST_P(Decode, bgr, testing::ValuesIn(exts))
{
    String filename = getDataPath("perf/1920x1080.png");

    Mat src = imread(filename);
    vector<uchar> buf;
    EXPECT_TRUE(imencode(GetParam(), src, buf));

    TEST_CYCLE() imdecode(buf, IMREAD_UNCHANGED);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Decode, rgb, testing::ValuesIn(exts))
{
    String filename = getDataPath("perf/1920x1080.png");

    Mat src = imread(filename);
    vector<uchar> buf;
    EXPECT_TRUE(imencode(GetParam(), src, buf));

    TEST_CYCLE() imdecode(buf, IMREAD_COLOR_RGB);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Encode, bgr, testing::ValuesIn(exts))
{
    String filename = getDataPath("perf/1920x1080.png");

    Mat src = imread(filename);
    vector<uchar> buf;

    TEST_CYCLE() imencode(GetParam(), src, buf);

    std::cout << "Encoded buffer size: " << buf.size()
        << " bytes, Compression ratio: " << std::fixed << std::setprecision(2)
        << (static_cast<double>(buf.size()) / (src.total() * src.channels())) * 100.0 << "%" << std::endl;

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Encode, multi, testing::ValuesIn(exts_multi))
{
    String filename = getDataPath("perf/1920x1080.png");
    vector<Mat> vec;
    imreadmulti(filename, vec);
    vec.push_back(vec.back().clone());
    circle(vec.back(), Point(100, 100), 45, Scalar(0, 0, 255, 0), 2, LINE_AA);
    vector<uchar> buf;
    imwrite("test" + GetParam(), vec);
    TEST_CYCLE() imencode(GetParam(), vec, buf);

    std::cout << "Encoded buffer size: " << buf.size()
        << " bytes, Compression ratio: " << std::fixed << std::setprecision(2)
        << (static_cast<double>(buf.size()) / (vec[0].total() * vec[0].channels())) * 100.0 << "%" << std::endl;

    SANITY_CHECK_NOTHING();
}
#endif // HAVE_PNG

} // namespace
