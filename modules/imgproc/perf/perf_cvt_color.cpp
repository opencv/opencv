#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

CV_ENUM(CvtMode, CV_YUV2BGR, CV_YUV2RGB, //YUV
        CV_YUV420i2BGR, CV_YUV420i2RGB, CV_YUV420sp2BGR, CV_YUV420sp2RGB, //YUV420
        CV_RGB2GRAY, CV_RGBA2GRAY, CV_BGR2GRAY, CV_BGRA2GRAY, //Gray
        CV_GRAY2RGB, CV_GRAY2RGBA, /*CV_GRAY2BGR, CV_GRAY2BGRA*/ //Gray2
        CV_BGR2HSV, CV_RGB2HSV, CV_BGR2HLS, CV_RGB2HLS, //H
        CV_BGR2YCrCb, CV_RGB2YCrCb
        )

typedef std::tr1::tuple<Size, CvtMode> Size_CvtMode_t;
typedef perf::TestBaseWithParam<Size_CvtMode_t> Size_CvtMode;

typedef std::tr1::tuple<Size, CvtMode, int> Size_CvtMode_OutChNum_t;
typedef perf::TestBaseWithParam<Size_CvtMode_OutChNum_t> Size_CvtMode_OutChNum;

PERF_TEST_P(Size_CvtMode_OutChNum, cvtColorYUV,
            testing::Combine(
                testing::Values(TYPICAL_MAT_SIZES),
                testing::Values((int)CV_YUV2BGR, (int)CV_YUV2RGB),
                testing::Values(3, 4)
                )
            )
{
    Size sz = get<0>(GetParam());
    int mode = get<1>(GetParam());
    int ch = get<2>(GetParam());

    Mat src(sz, CV_8UC3);
    Mat dst(sz, CV_8UC(ch));

    declare.in(src, WARMUP_RNG).out(dst);
    
    TEST_CYCLE() cvtColor(src, dst, mode, ch);
    
    SANITY_CHECK(dst, 1);
}


PERF_TEST_P(Size_CvtMode_OutChNum, cvtColorYUV420,
            testing::Combine(
                testing::Values(szVGA, sz720p, sz1080p, Size(130, 60)),
                testing::Values((int)CV_YUV420i2BGR, (int)CV_YUV420i2RGB, (int)CV_YUV420sp2BGR, (int)CV_YUV420sp2RGB),
                testing::Values(3, 4)
                )
            )
{
    Size sz = get<0>(GetParam());
    int mode = get<1>(GetParam());
    int ch = get<2>(GetParam());

    Mat src(sz.height + sz.height / 2, sz.width, CV_8UC1);
    Mat dst(sz, CV_8UC(ch));

    declare.in(src, WARMUP_RNG).out(dst);
    
    TEST_CYCLE() cvtColor(src, dst, mode, ch);
    
    SANITY_CHECK(dst, 1);
}


PERF_TEST_P(Size_CvtMode, cvtColorGray,
            testing::Combine(
                testing::Values(TYPICAL_MAT_SIZES),
                testing::Values((int)CV_RGB2GRAY, (int)CV_RGBA2GRAY, (int)CV_BGR2GRAY, (int)CV_BGRA2GRAY)
                )
            )
{
    Size sz = get<0>(GetParam());
    int mode = get<1>(GetParam());

    Mat src(sz, CV_8UC((mode==CV_RGBA2GRAY || mode==CV_BGRA2GRAY) ? 4 : 3));
    Mat dst(sz, CV_8UC1);

    declare.in(src, WARMUP_RNG).out(dst);
    
    TEST_CYCLE() cvtColor(src, dst, mode);
    
    SANITY_CHECK(dst, 1);
}


PERF_TEST_P(Size_CvtMode, cvtColorGray2,
            testing::Combine(
                testing::Values(TYPICAL_MAT_SIZES),
                testing::Values((int)CV_GRAY2RGB, (int)CV_GRAY2RGBA)
                )
            )
{
    Size sz = get<0>(GetParam());
    int mode = get<1>(GetParam());

    Mat src(sz, CV_8UC1);
    Mat dst(sz, CV_8UC((mode==CV_GRAY2RGBA || mode==CV_GRAY2BGRA) ? 4 : 3));

    declare.in(src, WARMUP_RNG).out(dst);
    
    TEST_CYCLE() cvtColor(src, dst, mode);
    
    SANITY_CHECK(dst);
}

PERF_TEST_P(Size_CvtMode, cvtColorH,
            testing::Combine(
                testing::Values(TYPICAL_MAT_SIZES),
                testing::Values((int)CV_BGR2HSV, (int)CV_RGB2HSV, (int)CV_BGR2HLS, (int)CV_RGB2HLS)
                )
            )
{
    Size sz = get<0>(GetParam());
    int mode = get<1>(GetParam());

    Mat src(sz, CV_8UC3);
    Mat dst(sz, CV_8UC3);

    declare.in(src, WARMUP_RNG).out(dst);
    
    TEST_CYCLE() cvtColor(src, dst, mode);
    
    SANITY_CHECK(dst, 1);
}

typedef std::tr1::tuple<Size, CvtMode, int> Size_CvtMode_Ch_t;
typedef perf::TestBaseWithParam<Size_CvtMode_Ch_t> Size_CvtMode_Ch;

PERF_TEST_P(Size_CvtMode_Ch, cvtColorYCrCb,
            testing::Combine(
                testing::Values(TYPICAL_MAT_SIZES),
                testing::Values((int)CV_BGR2YCrCb, (int)CV_RGB2YCrCb),
                testing::Values(3, 4)
                )
            )
{
    Size sz = get<0>(GetParam());
    int mode = get<1>(GetParam());
    int ch = get<2>(GetParam());

    Mat src(sz, CV_8UC(ch));
    Mat dst(sz, CV_8UC3);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() cvtColor(src, dst, mode);

    SANITY_CHECK(dst, 1);
}

CV_ENUM(CvtInBGR565Type, CV_RGB2BGR565, CV_RGBA2BGR565, CV_BGR2BGR565, CV_BGRA2BGR565)

typedef std::tr1::tuple<Size, CvtInBGR565Type> Size_CvtInBGR565Type_t;
typedef perf::TestBaseWithParam<Size_CvtInBGR565Type_t> Size_CvtInBGR565Type;

PERF_TEST_P( Size_CvtInBGR565Type, cvtColor_toBGR565,
             testing::Combine
             (
                 testing::Values(TYPICAL_MAT_SIZES),
                 testing::ValuesIn(CvtInBGR565Type::all())
             )
           )
{
    Size sz = get<0>(GetParam());
    CvtInBGR565Type code = get<1>(GetParam());

    Mat src;
    if ( code == CV_RGB2BGR565 || code == CV_BGR2BGR565 )
        src.create(sz, CV_8UC3);
    else
        src.create(sz, CV_8UC4);

    randu(src, 0, 255);

    Mat dst;
    
    TEST_CYCLE() cvtColor( src, dst, code );

    SANITY_CHECK(dst);
}

CV_ENUM(CvtMode2, CV_RGB2BGR, CV_RGB2RGBA, CV_RGB2BGRA, CV_RGBA2RGB, CV_RGBA2BGR, CV_RGBA2BGRA)

typedef std::tr1::tuple<Size, CvtMode2> Size_CvtMode2_t;
typedef perf::TestBaseWithParam<Size_CvtMode2_t> Size_CvtMode2;

PERF_TEST_P( Size_CvtMode2, cvtColor_C3toC4_and_back,
             testing::Combine
             (
                 testing::Values(TYPICAL_MAT_SIZES),
                 testing::ValuesIn(CvtMode2::all())
             )
           )
{
    Size sz = get<0>(GetParam());
    CvtMode2 code = get<1>(GetParam());

    Mat src;
    if ( code == CV_RGB2BGR || code == CV_RGB2RGBA || code == CV_RGB2BGRA )
        src.create(sz, CV_8UC3);
    else
        src.create(sz, CV_8UC4);

    randu(src, 0, 255);

    Mat dst;
    TEST_CYCLE() cvtColor( src, dst, code );

    SANITY_CHECK(dst);
}