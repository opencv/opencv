/*#include "perf_precomp.hpp"
#include "distransform.cpp"

using namespace std;
using namespace cv;
using namespace perf;

typedef perf::TestBaseWithParam<Size> Size_DistanceTransform;

PERF_TEST_P(Size_DistanceTransform, icvTrueDistTrans, testing::Values(TYPICAL_MAT_SIZES))
{
    Size size = GetParam();
    Mat src(size, CV_8UC1);
    Mat dst(size, CV_32FC1);
    CvMat srcStub = src;
    CvMat dstStub = dst;

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() icvTrueDistTrans(&srcStub, &dstStub);

    SANITY_CHECK(dst, 1);
}*/
