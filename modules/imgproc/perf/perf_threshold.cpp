#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

CV_ENUM(ThreshType, THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV)

typedef std::tr1::tuple<Size, MatType, ThreshType> Size_MatType_ThreshType_t;
typedef perf::TestBaseWithParam<Size_MatType_ThreshType_t> Size_MatType_ThreshType;

PERF_TEST_P(Size_MatType_ThreshType, threshold,
            testing::Combine(
                testing::Values(TYPICAL_MAT_SIZES),
                testing::Values(CV_8UC1, CV_16SC1, CV_32FC1, CV_64FC1),
                ThreshType::all()
                )
            )
{

    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    ThreshType threshType = get<2>(GetParam());

    Mat src(sz, type);
    Mat dst(sz, type);

    double thresh = theRNG().uniform(1, 254);
    double maxval = theRNG().uniform(1, 254);

    declare.in(src, WARMUP_RNG).out(dst);

    int runs = (sz.width <= 640) ? 40 : 1;
    TEST_CYCLE_MULTIRUN(runs) threshold(src, dst, thresh, maxval, threshType);

    SANITY_CHECK(dst);
}

typedef perf::TestBaseWithParam<Size> Size_Only;

PERF_TEST_P(Size_Only, threshold_otsu, testing::Values(TYPICAL_MAT_SIZES))
{
    Size sz = GetParam();

    Mat src(sz, CV_8UC1);
    Mat dst(sz, CV_8UC1);

    double maxval = theRNG().uniform(1, 254);

    declare.in(src, WARMUP_RNG).out(dst);

    int runs = 15;
    TEST_CYCLE_MULTIRUN(runs) threshold(src, dst, 0, maxval, THRESH_BINARY|THRESH_OTSU);

    SANITY_CHECK(dst);
}

CV_ENUM(AdaptThreshType, THRESH_BINARY, THRESH_BINARY_INV)
CV_ENUM(AdaptThreshMethod, ADAPTIVE_THRESH_MEAN_C, ADAPTIVE_THRESH_GAUSSIAN_C)

typedef std::tr1::tuple<Size, AdaptThreshType, AdaptThreshMethod, int, double> Size_AdaptThreshType_AdaptThreshMethod_BlockSize_Delta_t;
typedef perf::TestBaseWithParam<Size_AdaptThreshType_AdaptThreshMethod_BlockSize_Delta_t> Size_AdaptThreshType_AdaptThreshMethod_BlockSize_Delta;

PERF_TEST_P(Size_AdaptThreshType_AdaptThreshMethod_BlockSize_Delta, adaptiveThreshold,
            testing::Combine(
                testing::Values(TYPICAL_MAT_SIZES),
                AdaptThreshType::all(),
                AdaptThreshMethod::all(),
                testing::Values(3, 5),
                testing::Values(0.0, 10.0)
                )
            )
{
    Size sz = get<0>(GetParam());
    AdaptThreshType adaptThreshType = get<1>(GetParam());
    AdaptThreshMethod adaptThreshMethod = get<2>(GetParam());
    int blockSize = get<3>(GetParam());
    double C = get<4>(GetParam());

    double maxValue = theRNG().uniform(1, 254);

    int type = CV_8UC1;

    Mat src_full(cv::Size(sz.width + 2, sz.height + 2), type);
    Mat src = src_full(cv::Rect(1, 1, sz.width, sz.height));
    Mat dst(sz, type);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() adaptiveThreshold(src, dst, maxValue, adaptThreshMethod, adaptThreshType, blockSize, C);

    SANITY_CHECK(dst);
}
