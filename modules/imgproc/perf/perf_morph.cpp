#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;

#define TYPICAL_MAT_TYPES_MORPH  CV_8UC1, CV_8UC4
#define TYPICAL_MATS_MORPH       testing::Combine( SZ_ALL_GA, testing::Values( TYPICAL_MAT_TYPES_MORPH) )

/*
 void erode( InputArray src, OutputArray dst, InputArray kernel,
             Point anchor=Point(-1,-1), int iterations=1,
             int borderType=BORDER_CONSTANT,
             const Scalar& borderValue=morphologyDefaultBorderValue() );
*/
PERF_TEST_P(Size_MatType, erode1, TYPICAL_MATS_MORPH)
{
    Size sz = std::tr1::get<0>(GetParam());
    int type = std::tr1::get<1>(GetParam());

    Mat src(sz, type);
    Mat dst(sz, type);

    declare.in(src, WARMUP_RNG);

    TEST_CYCLE(100) 
	{ 
		erode(src, dst, Mat());
	}

    SANITY_CHECK(dst);
}
