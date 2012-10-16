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
    borderMode = get<1>(GetParam());
    interType  = get<2>(GetParam());

    Mat src, img = imread(getDataPath("cv/shared/fruits.png"));
    cvtColor(img, src, COLOR_BGR2RGBA, 4);
    Mat warpMat = getRotationMatrix2D(Point2f(src.cols/2.f, src.rows/2.f), 30., 2.2);
    Mat dst(sz, CV_8UC4);

    declare.in(src).out(dst);

    TEST_CYCLE() warpAffine( src, dst, warpMat, sz, interType, borderMode, Scalar::all(150) );

    SANITY_CHECK(dst);

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
    borderMode = get<1>(GetParam());
    interType  = get<2>(GetParam());


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

    SANITY_CHECK(dst);
}

PERF_TEST_P( TestWarpPerspective, WarpPerspectiveLarge,
             Combine(
                Values( sz3MP, sz5MP ),
                ValuesIn( InterType::all() ),
                ValuesIn( BorderMode::all() )
             )
)
{
    Size sz;
    int borderMode, interType;
    sz         = get<0>(GetParam());
    borderMode = get<1>(GetParam());
    interType  = get<2>(GetParam());

    string resolution;
    if (sz == sz3MP)
        resolution = "3MP";
    else if (sz == sz5MP)
        resolution = "5MP";
    else
        FAIL();

    Mat src, img = imread(getDataPath("cv/shared/" + resolution + ".png"));
    cvtColor(img, src, COLOR_BGR2BGRA, 4);

    int shift = 103;
    Mat srcVertices = (Mat_<Vec2f>(1, 4) << Vec2f(0, 0), Vec2f(sz.width-1, 0),
                                            Vec2f(sz.width-1, sz.height-1), Vec2f(0, sz.height-1));
    Mat dstVertices = (Mat_<Vec2f>(1, 4) << Vec2f(0, shift), Vec2f(sz.width-shift/2, 0),
                                            Vec2f(sz.width-shift, sz.height-shift), Vec2f(shift/2, sz.height-1));
    Mat warpMat = getPerspectiveTransform(srcVertices, dstVertices);

    Mat dst(sz, CV_8UC4);

    declare.in(src).out(dst);

    TEST_CYCLE()
        warpPerspective( src, dst, warpMat, sz, interType, borderMode, Scalar::all(150) );

    SANITY_CHECK(dst);
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

