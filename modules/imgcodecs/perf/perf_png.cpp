// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "perf_precomp.hpp"

#include <cstring>

namespace opencv_test
{

#if defined(HAVE_PNG) || defined(HAVE_SPNG)

using namespace perf;

static size_t pngChunkLength(const vector<uchar>& buffer, size_t offset)
{
    return (static_cast<size_t>(buffer[offset]) << 24) |
           (static_cast<size_t>(buffer[offset + 1]) << 16) |
           (static_cast<size_t>(buffer[offset + 2]) << 8) |
           static_cast<size_t>(buffer[offset + 3]);
}

static void appendPngUint32(vector<uchar>& buffer, uint32_t value)
{
    for (int shift = 24; shift >= 0; shift -= 8)
        buffer.push_back(static_cast<uchar>(value >> shift));
}

static uint32_t pngCRC(const uchar* data, size_t size)
{
    uint32_t table[256];
    for (uint32_t i = 0; i < 256; ++i)
    {
        uint32_t crc = i;
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ ((crc & 1) ? 0xedb88320U : 0);
        table[i] = crc;
    }
    uint32_t crc = 0xffffffffU;
    for (size_t i = 0; i < size; ++i)
        crc = table[(crc ^ data[i]) & 0xff] ^ (crc >> 8);
    return crc ^ 0xffffffffU;
}

// The encoder limits its IDAT buffer to 1 MiB. Combine the generated chunks
// without changing their compressed bytes to exercise a full-image IDAT.
static vector<uchar> pngSingleIDAT(const vector<uchar>& encoded)
{
    size_t first = 0, last = 0, payloadSize = 0;
    for (size_t offset = 8; offset + 12 <= encoded.size();)
    {
        const size_t length = pngChunkLength(encoded, offset);
        CV_Assert(length <= encoded.size() - offset - 12);
        if (std::memcmp(&encoded[offset + 4], "IDAT", 4) == 0)
        {
            if (first == 0)
                first = offset;
            last = offset + length + 12;
            payloadSize += length;
        }
        offset += length + 12;
    }
    CV_Assert(first != 0 && payloadSize <= 0x7fffffffU);

    vector<uchar> combined;
    combined.reserve(encoded.size());
    combined.insert(combined.end(), encoded.begin(), encoded.begin() + first);
    appendPngUint32(combined, static_cast<uint32_t>(payloadSize));
    combined.insert(combined.end(), encoded.begin() + first + 4, encoded.begin() + first + 8);
    for (size_t offset = first; offset < last;)
    {
        CV_Assert(std::memcmp(&encoded[offset + 4], "IDAT", 4) == 0);
        const size_t length = pngChunkLength(encoded, offset);
        combined.insert(combined.end(), encoded.begin() + offset + 8,
                        encoded.begin() + offset + 8 + length);
        offset += length + 12;
    }
    appendPngUint32(combined, pngCRC(&combined[first + 4], payloadSize + 4));
    combined.insert(combined.end(), encoded.begin() + last, encoded.end());
    return combined;
}

typedef perf::TestBaseWithParam<testing::tuple<Size, bool> > PNGRead;

PERF_TEST_P(PNGRead, idat_layout,
    testing::Combine(testing::Values(Size(512, 512), Size(3840, 2160)),
                     testing::Bool()))
{
    const bool singleIDAT = get<1>(GetParam());
    Mat src(get<0>(GetParam()), CV_8UC3);
    RNG rng(0x12345678);
    rng.fill(src, RNG::UNIFORM, 0, 256);
    vector<uchar> encoded;
    ASSERT_TRUE(imencode(".png", src, encoded,
        { IMWRITE_PNG_COMPRESSION, 0, IMWRITE_PNG_ZLIBBUFFER_SIZE, 8192 }));
    if (singleIDAT)
        encoded = pngSingleIDAT(encoded);

    const String filename = cv::tempfile(".png");
    FILE* file = fopen(filename.c_str(), "wb");
    ASSERT_TRUE(file != NULL);
    const size_t written = fwrite(encoded.data(), 1, encoded.size(), file);
    const int closeResult = fclose(file);
    ASSERT_EQ(encoded.size(), written);
    ASSERT_EQ(0, closeResult);

    Mat dst;
    TEST_CYCLE() dst = imread(filename, IMREAD_UNCHANGED);

    EXPECT_EQ(0, remove(filename.c_str()));
    EXPECT_EQ(0, cv::norm(src, dst, NORM_INF));
    SANITY_CHECK_NOTHING();
}

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
        testing::Values(1, 6)))
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
