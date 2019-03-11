#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

#define TYPICAL_MAT_TYPES_ADWEIGHTED  CV_8UC1, CV_8UC4, CV_8SC1, CV_16UC1, CV_16SC1, CV_32SC1
#define TYPICAL_MATS_ADWEIGHTED       testing::Combine(testing::Values(szVGA, sz720p, sz1080p), testing::Values(TYPICAL_MAT_TYPES_ADWEIGHTED))

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

    SANITY_CHECK(dst, depth == CV_32S ? 4 : 1);
}

} // namespace
