#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

CV_ENUM(CmpType, CMP_EQ, CMP_GT, CMP_GE, CMP_LT, CMP_LE, CMP_NE)

typedef std::tr1::tuple<Size, MatType, CmpType> Size_MatType_CmpType_t;
typedef perf::TestBaseWithParam<Size_MatType_CmpType_t> Size_MatType_CmpType;

PERF_TEST_P( Size_MatType_CmpType, compare,
             testing::Combine(
                 testing::Values(TYPICAL_MAT_SIZES),
                 testing::Values(CV_8UC1, CV_8UC4, CV_8SC1, CV_16UC1, CV_16SC1, CV_32SC1, CV_32FC1),
                 testing::ValuesIn(CmpType::all())
                 )
             )
{
    Size sz = get<0>(GetParam());
    int matType1 = get<1>(GetParam());
    CmpType cmpType = get<2>(GetParam());

    Mat src1(sz, matType1);
    Mat src2(sz, matType1);
    Mat dst(sz, CV_8UC(CV_MAT_CN(matType1)));

    declare.in(src1, src2, WARMUP_RNG).out(dst);

    TEST_CYCLE() cv::compare(src1, src2, dst, cmpType);

    SANITY_CHECK(dst);
}

PERF_TEST_P( Size_MatType_CmpType, compareScalar,
             testing::Combine(
                 testing::Values(TYPICAL_MAT_SIZES),
                 testing::Values(TYPICAL_MAT_TYPES),
                 testing::ValuesIn(CmpType::all())
                 )
             )
{
    Size sz = get<0>(GetParam());
    int matType = get<1>(GetParam());
    CmpType cmpType = get<2>(GetParam());

    Mat src1(sz, matType);
    Scalar src2;
    Mat dst(sz, CV_8UC(CV_MAT_CN(matType)));

    declare.in(src1, src2, WARMUP_RNG).out(dst);

    TEST_CYCLE() cv::compare(src1, src2, dst, cmpType);

    SANITY_CHECK(dst);
}
