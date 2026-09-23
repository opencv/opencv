#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

#define TYPICAL_MAT_TYPES_ADWEIGHTED  CV_8UC1, CV_8UC4, CV_8SC1, CV_16UC1, CV_16SC1, CV_32SC1
#define TYPICAL_MATS_ADWEIGHTED       testing::Combine(testing::Values(szVGA, sz720p, sz1080p, Size(127, 61)), testing::Values(TYPICAL_MAT_TYPES_ADWEIGHTED))

PERF_TEST_P(Size_MatType, addWeighted, TYPICAL_MATS_ADWEIGHTED)
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());
    int depth = CV_MAT_DEPTH(type);
    Mat src1(size, type);
    Mat src2(size, type);
    double alpha = 3.75;
    double beta = -0.125;
    double gamma = 100.0;

    Mat dst(size, type);

    declare.in(src1, src2, dst, WARMUP_RNG).out(dst);

    if (depth == CV_32S)
    {
        // there might be not enough precision for integers
        src1 /= 2048;
        src2 /= 2048;
    }

    TEST_CYCLE() cv::addWeighted( src1, alpha, src2, beta, gamma, dst, dst.type() );

    // accuracy is covered by the accuracy tests; regression data does not exist for every
    // size in the grid (127x61 guards per-call overhead only)
    SANITY_CHECK_NOTHING();
}


// The same computation written as an expression. cv::texpr() folds a*alpha + b*beta + gamma into
// the single fused OP_ADDW kernel instead of emitting two multiplies, two adds and three temp
// buffers, so this shares the grid above and can be compared against it directly.
PERF_TEST_P(Size_MatType, texpr_addWeighted, TYPICAL_MATS_ADWEIGHTED)
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());
    int depth = CV_MAT_DEPTH(type);
    Mat src1(size, type);
    Mat src2(size, type);

    declare.in(src1, src2, WARMUP_RNG);

    if (depth == CV_32S)
    {
        // there might be not enough precision for integers
        src1 /= 2048;
        src2 /= 2048;
    }

    std::vector<Mat> inputs{ src1, src2 }, outputs;

    TEST_CYCLE() cv::texpr("{0} * 3.75 + {1} * -0.125 + 100.0", inputs, outputs);

    SANITY_CHECK_NOTHING();
}

} // namespace
