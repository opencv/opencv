#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using namespace testing;
using std::tr1::make_tuple;
using std::tr1::get;

typedef tr1::tuple<Size, MatType> Size_Source_t;
typedef TestBaseWithParam<Size_Source_t> Size_Source;
typedef TestBaseWithParam<Size> MatSize;

static const float rangeHight = 256.0f;
static const float rangeLow = 0.0f;

PERF_TEST_P(Size_Source, calcHist1d,
            testing::Combine(testing::Values(sz3MP, sz5MP),
                             testing::Values(CV_8U, CV_16U, CV_32F) )
            )
{
    Size size = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    Mat source(size.height, size.width, type);
    Mat hist;
    int channels [] = {0};
    int histSize [] = {256};
    int dims = 1;
    int numberOfImages = 1;

    const float r[] = {rangeLow, rangeHight};
    const float* ranges[] = {r};

    randu(source, rangeLow, rangeHight);

    declare.in(source);

    TEST_CYCLE()
    {
        calcHist(&source, numberOfImages, channels, Mat(), hist, dims, histSize, ranges);
    }

    SANITY_CHECK(hist);
}

PERF_TEST_P(Size_Source, calcHist2d,
            testing::Combine(testing::Values(sz3MP, sz5MP),
                             testing::Values(CV_8UC2, CV_16UC2, CV_32FC2) )
            )
{
    Size size = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    Mat source(size.height, size.width, type);
    Mat hist;
    int channels [] = {0, 1};
    int histSize [] = {256, 256};
    int dims = 2;
    int numberOfImages = 1;

    const float r[] = {rangeLow, rangeHight};
    const float* ranges[] = {r, r};

    randu(source, rangeLow, rangeHight);

    declare.in(source);
    TEST_CYCLE()
    {
        calcHist(&source, numberOfImages, channels, Mat(), hist, dims, histSize, ranges);
    }

    SANITY_CHECK(hist);
}

PERF_TEST_P(Size_Source, calcHist3d,
            testing::Combine(testing::Values(sz3MP, sz5MP),
                             testing::Values(CV_8UC3, CV_16UC3, CV_32FC3) )
            )
{
    Size size = get<0>(GetParam());
    MatType type = get<1>(GetParam());
    Mat hist;
    int channels [] = {0, 1, 2};
    int histSize [] = {32, 32, 32};
    int dims = 3;
    int numberOfImages = 1;
    Mat source(size.height, size.width, type);

    const float r[] = {rangeLow, rangeHight};
    const float* ranges[] = {r, r, r};

    randu(source, rangeLow, rangeHight);

    declare.in(source);
    TEST_CYCLE()
    {
        calcHist(&source, numberOfImages, channels, Mat(), hist, dims, histSize, ranges);
    }

    SANITY_CHECK(hist);
}

PERF_TEST_P(MatSize, equalizeHist,
            testing::Values(TYPICAL_MAT_SIZES)
            )
{
    Size size = GetParam();
    Mat source(size.height, size.width, CV_8U);
    Mat destination;
    declare.in(source, WARMUP_RNG);

    TEST_CYCLE()
    {
        equalizeHist(source, destination);
    }

    SANITY_CHECK(destination);
}
