#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;

typedef perf::TestBaseWithParam<cv::Size> MatSize;

/*
 void erode( InputArray src, OutputArray dst, InputArray kernel,
             Point anchor=Point(-1,-1), int iterations=1,
             int borderType=BORDER_CONSTANT,
             const Scalar& borderValue=morphologyDefaultBorderValue() );
*/
PERF_TEST_P( MatSize, erode, ::testing::Values( TYPICAL_MAT_SIZES ))
{
    Size sz = GetParam();
    int type = CV_8UC1;

    Mat src(sz, type);
    Mat dst(sz, type);

    declare.in(src, WARMUP_RNG);

    TEST_CYCLE(100) { erode(src, dst, Mat()); }

    SANITY_CHECK(dst);
}
