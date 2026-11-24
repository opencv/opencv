// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {

CV_ENUM(ThreshType, THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV)

typedef tuple<Size, MatType, ThreshType> Size_MatType_ThreshType_t;
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
    TEST_CYCLE_MULTIRUN(runs) cv::threshold(src, dst, thresh, maxval, threshType);

    SANITY_CHECK(dst);
}

// values in [0, 1) keep fp16 away from its overflow range
static Mat halfThresholdSrc(Size sz, int type)
{
    Mat src32(sz, CV_MAKETYPE(CV_32F, CV_MAT_CN(type))), src;
    randu(src32, Scalar::all(0), Scalar::all(1));
    src32.convertTo(src, type);
    return src;
}

// the fp32 rows are the reference for the half rows
PERF_TEST_P(Size_MatType_ThreshType, threshold_half,
            testing::Combine(
                testing::Values(TYPICAL_MAT_SIZES),
                testing::Values(CV_16FC1, CV_16FC3, CV_16BFC1, CV_16BFC3, CV_32FC1, CV_32FC3),
                ThreshType::all()
                )
            )
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    ThreshType threshType = get<2>(GetParam());

    Mat src = halfThresholdSrc(sz, type);
    Mat dst(sz, type);

    declare.in(src).out(dst);

    int runs = (sz.width <= 640) ? 40 : 1;
    TEST_CYCLE_MULTIRUN(runs) cv::threshold(src, dst, 0.5, 1.0, threshType);

    SANITY_CHECK_NOTHING();
}

// convert to fp32, threshold, convert back: the route available without native
// half support, so the native half path has to beat it
PERF_TEST_P(Size_MatType_ThreshType, threshold_half_via_fp32,
            testing::Combine(
                testing::Values(TYPICAL_MAT_SIZES),
                testing::Values(CV_16FC1, CV_16FC3, CV_16BFC1, CV_16BFC3),
                ThreshType::all()
                )
            )
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    ThreshType threshType = get<2>(GetParam());

    Mat src = halfThresholdSrc(sz, type);
    Mat dst(sz, type), src32, dst32;

    declare.in(src).out(dst);

    int runs = (sz.width <= 640) ? 40 : 1;
    TEST_CYCLE_MULTIRUN(runs)
    {
        src.convertTo(src32, CV_32F);
        cv::threshold(src32, dst32, 0.5, 1.0, threshType);
        dst32.convertTo(dst, CV_MAT_DEPTH(type));
    }

    SANITY_CHECK_NOTHING();
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
    TEST_CYCLE_MULTIRUN(runs) cv::threshold(src, dst, 0, maxval, THRESH_BINARY|THRESH_OTSU);

    SANITY_CHECK(dst);
}

CV_ENUM(AdaptThreshType, THRESH_BINARY, THRESH_BINARY_INV)
CV_ENUM(AdaptThreshMethod, ADAPTIVE_THRESH_MEAN_C, ADAPTIVE_THRESH_GAUSSIAN_C)

typedef tuple<Size, AdaptThreshType, AdaptThreshMethod, int, double> Size_AdaptThreshType_AdaptThreshMethod_BlockSize_Delta_t;
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

    TEST_CYCLE() cv::adaptiveThreshold(src, dst, maxValue, adaptThreshMethod, adaptThreshType, blockSize, C);

    SANITY_CHECK(dst);
}

} // namespace
