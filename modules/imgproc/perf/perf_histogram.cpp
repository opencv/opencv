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


PERF_TEST_P(Size_Source, calcHist,
            testing::Combine(testing::Values(TYPICAL_MAT_SIZES),
                             testing::Values(CV_8U, CV_32F)
                             )
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

    const float r[] = {0.0f, 256.0f};
    const float* ranges[] = {r};

    declare.in(source, WARMUP_RNG).time(20).iterations(1000);
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
