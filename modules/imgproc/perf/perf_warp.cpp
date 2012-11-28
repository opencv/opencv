#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using namespace testing;
using std::tr1::make_tuple;
using std::tr1::get;

enum{HALF_SIZE=0, UPSIDE_DOWN, REFLECTION_X, REFLECTION_BOTH};

CV_ENUM(BorderMode, BORDER_CONSTANT, BORDER_REPLICATE)
CV_ENUM(InterType, INTER_NEAREST, INTER_LINEAR)
CV_ENUM(RemapMode, HALF_SIZE, UPSIDE_DOWN, REFLECTION_X, REFLECTION_BOTH)

typedef TestBaseWithParam< tr1::tuple<Size, InterType, BorderMode> > TestWarpAffine;
typedef TestBaseWithParam< tr1::tuple<Size, InterType, BorderMode> > TestWarpPerspective;
typedef TestBaseWithParam< tr1::tuple<Size, InterType, BorderMode, MatType> > TestWarpPerspectiveNear_t;
typedef TestBaseWithParam< tr1::tuple<MatType, Size, InterType, BorderMode, RemapMode> > TestRemap;

void update_map(const Mat& src, Mat& map_x, Mat& map_y, const int remapMode );

PERF_TEST_P( TestWarpAffine, WarpAffine,
             Combine(
                Values( szVGA, sz720p, sz1080p ),
                ValuesIn( InterType::all() ),
                ValuesIn( BorderMode::all() )
             )
)
{
    Size sz;
    int borderMode, interType;
    sz         = get<0>(GetParam());
    interType  = get<1>(GetParam());
    borderMode = get<2>(GetParam());

    Mat src, img = imread(getDataPath("cv/shared/fruits.png"));
    cvtColor(img, src, COLOR_BGR2RGBA, 4);
    Mat warpMat = getRotationMatrix2D(Point2f(src.cols/2.f, src.rows/2.f), 30., 2.2);
    Mat dst(sz, CV_8UC4);

    declare.in(src).out(dst);

    TEST_CYCLE() warpAffine( src, dst, warpMat, sz, interType, borderMode, Scalar::all(150) );

    SANITY_CHECK(dst, 1);

}

PERF_TEST_P( TestWarpPerspective, WarpPerspective,
             Combine(
                Values( szVGA, sz720p, sz1080p ),
                ValuesIn( InterType::all() ),
                ValuesIn( BorderMode::all() )
             )
)
{
    Size sz;
    int borderMode, interType;
    sz         = get<0>(GetParam());
    interType  = get<1>(GetParam());
    borderMode = get<2>(GetParam());


    Mat src, img = imread(getDataPath("cv/shared/fruits.png"));
    cvtColor(img, src, COLOR_BGR2RGBA, 4);
    Mat rotMat = getRotationMatrix2D(Point2f(src.cols/2.f, src.rows/2.f), 30., 2.2);
    Mat warpMat(3, 3, CV_64FC1);
    for(int r=0; r<2; r++)
        for(int c=0; c<3; c++)
            warpMat.at<double>(r, c) = rotMat.at<double>(r, c);
    warpMat.at<double>(2, 0) = .3/sz.width;
    warpMat.at<double>(2, 1) = .3/sz.height;
    warpMat.at<double>(2, 2) = 1;
    Mat dst(sz, CV_8UC4);

    declare.in(src).out(dst);

    TEST_CYCLE() warpPerspective( src, dst, warpMat, sz, interType, borderMode, Scalar::all(150) );

    SANITY_CHECK(dst, 1);
}

PERF_TEST_P( TestWarpPerspectiveNear_t, WarpPerspectiveNear,
             Combine(
                 Values( Size(176,144), Size(320,240), Size(352,288), Size(480,480),
                         Size(640,480), Size(704,576), Size(720,408), Size(720,480),
                         Size(720,576), Size(768,432), Size(800,448), Size(960,720),
                         Size(1024,768), Size(1280,720), Size(1280,960), Size(1360,720),
                         Size(1600,1200), Size(1920,1080), Size(2048,1536), Size(2592,1920),
                         Size(2592,1944), Size(3264,2448), Size(4096,3072), Size(4208,3120) ),
                 ValuesIn( InterType::all() ),
                 ValuesIn( BorderMode::all() ),
                 Values( CV_8UC1, CV_8UC4 )
                 )
             )
{
    Size size;
    int borderMode, interType, type;
    size       = get<0>(GetParam());
    interType  = get<1>(GetParam());
    borderMode = get<2>(GetParam());
    type       = get<3>(GetParam());

    Mat src, img = imread(getDataPath("cv/shared/5MP.png"));

    if( type == CV_8UC1 )
    {
        cvtColor(img, src, COLOR_BGR2GRAY, 1);
    }
    else if( type == CV_8UC4 )
    {
        cvtColor(img, src, COLOR_BGR2BGRA, 4);
    }
    else
    {
        FAIL();
    }

    resize(src, src, size);

    int shift = static_cast<int>(src.cols*0.04);
    Mat srcVertices = (Mat_<Vec2f>(1, 4) << Vec2f(0, 0),
                                            Vec2f(static_cast<float>(size.width-1), 0),
                                            Vec2f(static_cast<float>(size.width-1), static_cast<float>(size.height-1)),
                                            Vec2f(0, static_cast<float>(size.height-1)));
    Mat dstVertices = (Mat_<Vec2f>(1, 4) << Vec2f(0, static_cast<float>(shift)),
                                            Vec2f(static_cast<float>(size.width-shift/2), 0),
                                            Vec2f(static_cast<float>(size.width-shift), static_cast<float>(size.height-shift)),
                                            Vec2f(static_cast<float>(shift/2), static_cast<float>(size.height-1)));
    Mat warpMat = getPerspectiveTransform(srcVertices, dstVertices);

    Mat dst(size, type);

    declare.in(src).out(dst);
    declare.time(100);

    TEST_CYCLE()
    {
        warpPerspective( src, dst, warpMat, size, interType, borderMode, Scalar::all(150) );
    }

    SANITY_CHECK(dst, 1);
}

PERF_TEST_P( TestRemap, remap,
             Combine(
                 Values( TYPICAL_MAT_TYPES ),
                 Values( szVGA, sz720p, sz1080p ),
                 ValuesIn( InterType::all() ),
                 ValuesIn( BorderMode::all() ),
                 ValuesIn( RemapMode::all() )
                 )
             )
{
    int type = get<0>(GetParam());
    Size size = get<1>(GetParam());
    int interpolationType = get<2>(GetParam());
    int borderMode = get<3>(GetParam());
    int remapMode = get<4>(GetParam());
    unsigned int height = size.height;
    unsigned int width = size.width;
    Mat source(height, width, type);
    Mat destination;
    Mat map_x(height, width, CV_32F);
    Mat map_y(height, width, CV_32F);

    declare.in(source, WARMUP_RNG);

    update_map(source, map_x, map_y, remapMode);

    TEST_CYCLE()
    {
        remap(source, destination, map_x, map_y, interpolationType, borderMode);
    }

    SANITY_CHECK(destination, 1);
}

void update_map(const Mat& src, Mat& map_x, Mat& map_y, const int remapMode )
{
    for( int j = 0; j < src.rows; j++ )
    {
        for( int i = 0; i < src.cols; i++ )
        {
            switch( remapMode )
            {
            case HALF_SIZE:
                if( i > src.cols*0.25 && i < src.cols*0.75 && j > src.rows*0.25 && j < src.rows*0.75 )
                {
                    map_x.at<float>(j,i) = 2*( i - src.cols*0.25f ) + 0.5f ;
                    map_y.at<float>(j,i) = 2*( j - src.rows*0.25f ) + 0.5f ;
                }
                else
                {
                    map_x.at<float>(j,i) = 0 ;
                    map_y.at<float>(j,i) = 0 ;
                }
                break;
            case UPSIDE_DOWN:
                map_x.at<float>(j,i) = static_cast<float>(i) ;
                map_y.at<float>(j,i) = static_cast<float>(src.rows - j) ;
                break;
            case REFLECTION_X:
                map_x.at<float>(j,i) = static_cast<float>(src.cols - i) ;
                map_y.at<float>(j,i) = static_cast<float>(j) ;
                break;
            case REFLECTION_BOTH:
                map_x.at<float>(j,i) = static_cast<float>(src.cols - i) ;
                map_y.at<float>(j,i) = static_cast<float>(src.rows - j) ;
                break;
            } // end of switch
        }
    }
}

PERF_TEST(Transform, getPerspectiveTransform)
{
    unsigned int size = 8;
    Mat source(1, size/2, CV_32FC2);
    Mat destination(1, size/2, CV_32FC2);
    Mat transformCoefficient;

    declare.in(source, destination, WARMUP_RNG);

    TEST_CYCLE()
    {
        transformCoefficient = getPerspectiveTransform(source, destination);
    }

    SANITY_CHECK(transformCoefficient, 1e-5);
}

