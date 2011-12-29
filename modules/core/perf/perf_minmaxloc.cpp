#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

PERF_TEST_P(Size_MatType, minMaxLoc, testing::Combine(
                 testing::Values(TYPICAL_MAT_SIZES),
                 testing::Values(CV_8UC1, CV_8SC1, CV_16UC1, CV_16SC1, CV_32SC1,  CV_32FC1, CV_64FC1)
                 )
             )
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());

    Mat src(sz, matType);
    double minVal, maxVal;
    Point minLoc, maxLoc;

    if (matType == CV_8U)
        randu(src, 1, 254 /*do not include 0 and 255 to avoid early exit on 1 byte data*/);
    else if (matType == CV_8S)
        randu(src, -127, 126);
    else
        warmup(src, WARMUP_RNG);

    declare.in(src);

    TEST_CYCLE() minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc);

    SANITY_CHECK(minVal, 1e-12);
    SANITY_CHECK(maxVal, 1e-12);
}
