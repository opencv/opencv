#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using namespace testing;

CV_ENUM(BorderMode, BORDER_CONSTANT, BORDER_REPLICATE);
CV_ENUM(InterType, INTER_NEAREST, INTER_LINEAR);

typedef TestBaseWithParam< tr1::tuple<Size, InterType, BorderMode> > TestWarpAffine;
typedef TestBaseWithParam< tr1::tuple<Size, InterType, BorderMode> > TestWarpPerspective;


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
    //tr1::tie(sz, borderMode, interType) = GetParam();
    sz         = get<0>(GetParam());
    borderMode = get<1>(GetParam());
    interType  = get<2>(GetParam());

    Mat src, img = imread(getDataPath("cv/shared/fruits.jpg"));
    cvtColor(img, src, COLOR_BGR2RGBA, 4);
    Mat warpMat = getRotationMatrix2D(Point2f(src.cols/2.f, src.rows/2.f), 30., 2.2);
    Mat dst(sz, CV_8UC4);

    //declare.in(src).out(dst);

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
    //tr1::tie(sz, borderMode, interType) = GetParam();
    sz         = get<0>(GetParam());
    borderMode = get<1>(GetParam());
    interType  = get<2>(GetParam());


    Mat src, img = imread(getDataPath("cv/shared/fruits.jpg"));
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

    //declare.in(src).out(dst);

    TEST_CYCLE() warpPerspective( src, dst, warpMat, sz, interType, borderMode, Scalar::all(150) );

    SANITY_CHECK(dst);

}

