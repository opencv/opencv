#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;

typedef tr1::tuple<MatType, Size, Size> MatInfo_Size_Size_t;
typedef TestBaseWithParam<MatInfo_Size_Size_t> MatInfo_Size_Size;

PERF_TEST_P(MatInfo_Size_Size, resizeUpLinear,
    testing::Values(
                MatInfo_Size_Size_t(CV_8UC1, szVGA, szqHD),
                MatInfo_Size_Size_t(CV_8UC1, szVGA, sz720p),
                MatInfo_Size_Size_t(CV_8UC4, szVGA, sz720p)
                )
)
{
    int matType = tr1::get<0>(GetParam());
    Size from = tr1::get<1>(GetParam());
    Size to = tr1::get<2>(GetParam());

    cv::Mat src(from, matType);
    cv::Mat dst(to, matType);

    declare.in(src, WARMUP_RNG).out(dst);

    SIMPLE_TEST_CYCLE() cv::resize(src, dst, to);

    SANITY_CHECK(dst, 1 + 1e-6);
}

PERF_TEST_P(MatInfo_Size_Size, resizeDownLinear,
    testing::Values(
                MatInfo_Size_Size_t(CV_8UC1, szVGA, szQVGA),
                MatInfo_Size_Size_t(CV_8UC4, szqHD, szVGA),
                MatInfo_Size_Size_t(CV_8UC1, sz720p, Size(120 * sz720p.width / sz720p.height, 120)),//face detection min_face_size = 20%
                MatInfo_Size_Size_t(CV_8UC4, sz720p, szVGA),
                MatInfo_Size_Size_t(CV_8UC4, sz720p, szQVGA)
                )
)
{
    int matType = tr1::get<0>(GetParam());
    Size from = tr1::get<1>(GetParam());
    Size to = tr1::get<2>(GetParam());

    cv::Mat src(from, matType);
    cv::Mat dst(to, matType);

    declare.in(src, WARMUP_RNG).out(dst);

    SIMPLE_TEST_CYCLE() cv::resize(src, dst, to);

    SANITY_CHECK(dst, 1 + 1e-6);
}


typedef tr1::tuple<MatType, Size, int> MatInfo_Size_Scale_t;
typedef TestBaseWithParam<MatInfo_Size_Scale_t> MatInfo_Size_Scale;

PERF_TEST_P( MatInfo_Size_Scale, resizeAreaFast,
             testing::Combine(
                 testing::Values( CV_8UC1, CV_8UC4 ),
                 testing::Values( szVGA, szqHD, sz720p, sz1080p ),
                 testing::Values( 2, 4 )
                 )
             )
{
    int matType = tr1::get<0>(GetParam());
    Size from = tr1::get<1>(GetParam());
    int scale = tr1::get<2>(GetParam());

    from.width = (from.width/scale)*scale;
    from.height = (from.height/scale)*scale;

    cv::Mat src(from, matType);
    cv::Mat dst(from.height / scale, from.width / scale, matType);

    declare.in(src, WARMUP_RNG).out(dst);

    SIMPLE_TEST_CYCLE() cv::resize(src, dst, dst.size(), 0, 0, INTER_AREA);

    //difference equal to 1 is allowed because of different possible rounding modes: round-to-nearest vs bankers' rounding
    SANITY_CHECK(dst, 1);
}
