#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;

CV_ENUM(CvtMode, CV_YUV2BGR, CV_YUV2RGB, //YUV
                 CV_YUV420i2BGR, CV_YUV420i2RGB, CV_YUV420sp2BGR, CV_YUV420sp2RGB, //YUV420
                 CV_RGB2GRAY, CV_RGBA2GRAY, CV_BGR2GRAY, CV_BGRA2GRAY, //Gray
                 CV_GRAY2RGB, CV_GRAY2RGBA/*, CV_GRAY2BGR, CV_GRAY2BGRA*/ //Gray2
                 )

typedef std::tr1::tuple<Size, CvtMode> Size_CvtMode_t;
typedef perf::TestBaseWithParam<Size_CvtMode_t> Size_CvtMode;

typedef std::tr1::tuple<Size, CvtMode, int> Size_CvtMode_OutChNum_t;
typedef perf::TestBaseWithParam<Size_CvtMode_OutChNum_t> Size_CvtMode_OutChNum;


/*
// void cvtColor(InputArray src, OutputArray dst, int code, int dstCn=0 )
*/


PERF_TEST_P( Size_CvtMode_OutChNum, cvtColorYUV,
    testing::Combine( 
    testing::Values( TYPICAL_MAT_SIZES ), 
        testing::Values( (int)CV_YUV2BGR, (int)CV_YUV2RGB ),
        testing::Values( 3, 4 )
    )
)
{
    Size sz = std::tr1::get<0>(GetParam());
    int mode = std::tr1::get<1>(GetParam());
    int ch = std::tr1::get<2>(GetParam());

    Mat src(sz, CV_8UC3);
    Mat dst(sz, CV_8UC(ch));

    declare.in(src, WARMUP_RNG).out(dst);
    
    TEST_CYCLE(100) { cvtColor(src, dst, mode, ch);  }
    
    SANITY_CHECK(dst);
}


PERF_TEST_P( Size_CvtMode_OutChNum, cvtColorYUV420,
    testing::Combine( 
        testing::Values( szVGA, sz720p, sz1080p, Size(130, 60) ), 
        testing::Values( (int)CV_YUV420i2BGR, (int)CV_YUV420i2RGB, (int)CV_YUV420sp2BGR, (int)CV_YUV420sp2RGB ),
        testing::Values( 3, 4 )
    )
)
{
    Size sz = std::tr1::get<0>(GetParam());
    int mode = std::tr1::get<1>(GetParam());
    int ch = std::tr1::get<2>(GetParam());

    Mat src(sz.height+sz.height/2, sz.width, CV_8UC1);
    Mat dst(sz, CV_8UC(ch));

    declare.in(src, WARMUP_RNG).out(dst);
    
    TEST_CYCLE(100) { cvtColor(src, dst, mode, ch);  }
    
    SANITY_CHECK(dst);
}


PERF_TEST_P( Size_CvtMode, cvtColorGray,
    testing::Combine( 
        testing::Values( TYPICAL_MAT_SIZES ), 
        testing::Values( (int)CV_RGB2GRAY, (int)CV_RGBA2GRAY, (int)CV_BGR2GRAY, (int)CV_BGRA2GRAY )
    )
)
{
    Size sz = std::tr1::get<0>(GetParam());
    int mode = std::tr1::get<1>(GetParam());

    Mat src(sz, CV_8UC((mode==CV_RGBA2GRAY || mode==CV_BGRA2GRAY)?4:3));
    Mat dst(sz, CV_8UC1);

    declare.in(src, WARMUP_RNG).out(dst);
    
    TEST_CYCLE(100) { cvtColor(src, dst, mode);  }
    
    SANITY_CHECK(dst);
}


PERF_TEST_P( Size_CvtMode, cvtColorGray2,
    testing::Combine( 
        testing::Values( TYPICAL_MAT_SIZES ), 
        testing::Values( (int)CV_GRAY2RGB, (int)CV_GRAY2RGBA/*, CV_GRAY2BGR, CV_GRAY2BGRA*/ )
    )
)
{
    Size sz = std::tr1::get<0>(GetParam());
    int mode = std::tr1::get<1>(GetParam());

    Mat src(sz, CV_8UC1);
    Mat dst(sz, CV_8UC((mode==CV_GRAY2RGBA || mode==CV_GRAY2BGRA)?4:3));

    declare.in(src, WARMUP_RNG).out(dst);
    
    TEST_CYCLE(100) { cvtColor(src, dst, mode);  }
    
    SANITY_CHECK(dst);
}

