// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "perf_precomp.hpp"

namespace opencv_test
{

#if defined(HAVE_PNG) || defined(HAVE_SPNG)

using namespace perf;

CV_ENUM(PNGStrategy, IMWRITE_PNG_STRATEGY_DEFAULT, IMWRITE_PNG_STRATEGY_FILTERED, IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY, IMWRITE_PNG_STRATEGY_RLE, IMWRITE_PNG_STRATEGY_FIXED);
CV_ENUM(PNGFilters, IMWRITE_PNG_FILTER_NONE, IMWRITE_PNG_FILTER_SUB, IMWRITE_PNG_FILTER_UP, IMWRITE_PNG_FILTER_AVG, IMWRITE_PNG_FILTER_PAETH, IMWRITE_PNG_FAST_FILTERS, IMWRITE_PNG_ALL_FILTERS);

typedef perf::TestBaseWithParam<testing::tuple<PNGStrategy, PNGFilters, int>> PNG;

PERF_TEST(PNG, decode)
{
    String filename = getDataPath("perf/2560x1600.png");

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

PERF_TEST(PNG, decode_rgb)
{
    String filename = getDataPath("perf/2560x1600.png");

    FILE *f = fopen(filename.c_str(), "rb");
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    vector<uchar> file_buf((size_t)len);
    EXPECT_EQ(len, (long)fread(&file_buf[0], 1, (size_t)len, f));
    fclose(f); f = NULL;

    TEST_CYCLE() imdecode(file_buf, IMREAD_COLOR_RGB);

    SANITY_CHECK_NOTHING();
}

PERF_TEST(PNG, encode)
{
    String filename = getDataPath("perf/2560x1600.png");
    cv::Mat src = imread(filename);

    vector<uchar> buf;
    TEST_CYCLE() imencode(".png", src, buf);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(PNG, params,
    testing::Combine(
        testing::Values(IMWRITE_PNG_STRATEGY_DEFAULT, IMWRITE_PNG_STRATEGY_FILTERED, IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY, IMWRITE_PNG_STRATEGY_RLE, IMWRITE_PNG_STRATEGY_FIXED),
        testing::Values(IMWRITE_PNG_FILTER_NONE, IMWRITE_PNG_FILTER_SUB, IMWRITE_PNG_FILTER_UP, IMWRITE_PNG_FILTER_AVG, IMWRITE_PNG_FILTER_PAETH, IMWRITE_PNG_FAST_FILTERS, IMWRITE_PNG_ALL_FILTERS),
        testing::Values(1, 6, 9)))
{
    String filename = getDataPath("perf/1920x1080.png");
    const int strategy = get<0>(GetParam());
    const int filter = get<1>(GetParam());
    const int level = get<2>(GetParam());

    Mat src = imread(filename);
    EXPECT_FALSE(src.empty()) << "Cannot open test image perf/1920x1080.png";
    vector<uchar> buf;

    TEST_CYCLE() imencode(".png", src, buf, { IMWRITE_PNG_COMPRESSION, level, IMWRITE_PNG_STRATEGY, strategy, IMWRITE_PNG_FILTER, filter });

    std::cout << "  Encoded buffer size: " << buf.size()
        << " bytes, Compression ratio: " << std::fixed << std::setprecision(2)
        << (static_cast<double>(buf.size()) / (src.total() * src.channels())) * 100.0 << "%" << std::endl;

    SANITY_CHECK_NOTHING();
}

#endif // HAVE_PNG

} // namespace
