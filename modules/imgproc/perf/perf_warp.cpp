// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {

enum{HALF_SIZE=0, UPSIDE_DOWN, REFLECTION_X, REFLECTION_BOTH};

CV_ENUM(BorderMode, BORDER_CONSTANT, BORDER_REPLICATE)
CV_ENUM(InterType, INTER_NEAREST, INTER_LINEAR)
CV_ENUM(RemapMode, HALF_SIZE, UPSIDE_DOWN, REFLECTION_X, REFLECTION_BOTH)

typedef TestBaseWithParam< tuple<Size, InterType, BorderMode> > TestWarpAffine;
typedef TestBaseWithParam< tuple<Size, InterType, BorderMode> > TestWarpPerspective;
typedef TestBaseWithParam< tuple<Size, InterType, BorderMode, MatType> > TestWarpPerspectiveNear_t;
typedef TestBaseWithParam< tuple<MatType, Size, InterType, BorderMode, RemapMode> > TestRemap;

void update_map(const Mat& src, Mat& map_x, Mat& map_y, const int remapMode );

PERF_TEST_P( TestWarpAffine, WarpAffine,
             Combine(
                Values( szVGA, sz720p, sz1080p ),
                InterType::all(),
                BorderMode::all()
             )
)
{
    Size sz, szSrc(512, 512);
    int borderMode, interType;
    sz         = get<0>(GetParam());
    interType  = get<1>(GetParam());
    borderMode = get<2>(GetParam());
    Scalar borderColor = Scalar::all(150);

    Mat src(szSrc,CV_8UC4), dst(sz, CV_8UC4);
    cvtest::fillGradient(src);
    if(borderMode == BORDER_CONSTANT) cvtest::smoothBorder(src, borderColor, 1);
    Mat warpMat = getRotationMatrix2D(Point2f(src.cols/2.f, src.rows/2.f), 30., 2.2);
    declare.in(src).out(dst);

    TEST_CYCLE() warpAffine( src, dst, warpMat, sz, interType, borderMode, borderColor );

#ifdef __ANDROID__
    SANITY_CHECK(dst, interType==INTER_LINEAR? 5 : 10);
#else
    SANITY_CHECK(dst, 1);
#endif
}

PERF_TEST_P(TestWarpAffine, WarpAffine_ovx,
    Combine(
        Values(szVGA, sz720p, sz1080p),
        InterType::all(),
        BorderMode::all()
    )
)
{
    Size sz, szSrc(512, 512);
    int borderMode, interType;
    sz = get<0>(GetParam());
    interType = get<1>(GetParam());
    borderMode = get<2>(GetParam());
    Scalar borderColor = Scalar::all(150);

    Mat src(szSrc, CV_8UC1), dst(sz, CV_8UC1);
    cvtest::fillGradient(src);
    if (borderMode == BORDER_CONSTANT) cvtest::smoothBorder(src, borderColor, 1);
    Mat warpMat = getRotationMatrix2D(Point2f(src.cols / 2.f, src.rows / 2.f), 30., 2.2);
    declare.in(src).out(dst);

    TEST_CYCLE() warpAffine(src, dst, warpMat, sz, interType, borderMode, borderColor);

#ifdef __ANDROID__
    SANITY_CHECK(dst, interType == INTER_LINEAR ? 5 : 10);
#else
    SANITY_CHECK(dst, 1);
#endif
}

PERF_TEST_P( TestWarpPerspective, WarpPerspective,
             Combine(
                Values( szVGA, sz720p, sz1080p ),
                InterType::all(),
                BorderMode::all()
             )
)
{
    Size sz, szSrc(512, 512);
    int borderMode, interType;
    sz         = get<0>(GetParam());
    interType  = get<1>(GetParam());
    borderMode = get<2>(GetParam());
    Scalar borderColor = Scalar::all(150);

    Mat src(szSrc,CV_8UC4), dst(sz, CV_8UC4);
    cvtest::fillGradient(src);
    if(borderMode == BORDER_CONSTANT) cvtest::smoothBorder(src, borderColor, 1);
    Mat rotMat = getRotationMatrix2D(Point2f(src.cols/2.f, src.rows/2.f), 30., 2.2);
    Mat warpMat(3, 3, CV_64FC1);
    for(int r=0; r<2; r++)
        for(int c=0; c<3; c++)
            warpMat.at<double>(r, c) = rotMat.at<double>(r, c);
    warpMat.at<double>(2, 0) = .3/sz.width;
    warpMat.at<double>(2, 1) = .3/sz.height;
    warpMat.at<double>(2, 2) = 1;

    declare.in(src).out(dst);

    TEST_CYCLE() warpPerspective( src, dst, warpMat, sz, interType, borderMode, borderColor );

#ifdef __ANDROID__
    SANITY_CHECK(dst, interType==INTER_LINEAR? 5 : 10);
#else
    SANITY_CHECK(dst, 1);
#endif
}

PERF_TEST_P(TestWarpPerspective, WarpPerspective_ovx,
    Combine(
        Values(szVGA, sz720p, sz1080p),
        InterType::all(),
        BorderMode::all()
    )
)
{
    Size sz, szSrc(512, 512);
    int borderMode, interType;
    sz = get<0>(GetParam());
    interType = get<1>(GetParam());
    borderMode = get<2>(GetParam());
    Scalar borderColor = Scalar::all(150);

    Mat src(szSrc, CV_8UC1), dst(sz, CV_8UC1);
    cvtest::fillGradient(src);
    if (borderMode == BORDER_CONSTANT) cvtest::smoothBorder(src, borderColor, 1);
    Mat rotMat = getRotationMatrix2D(Point2f(src.cols / 2.f, src.rows / 2.f), 30., 2.2);
    Mat warpMat(3, 3, CV_64FC1);
    for (int r = 0; r<2; r++)
        for (int c = 0; c<3; c++)
            warpMat.at<double>(r, c) = rotMat.at<double>(r, c);
    warpMat.at<double>(2, 0) = .3 / sz.width;
    warpMat.at<double>(2, 1) = .3 / sz.height;
    warpMat.at<double>(2, 2) = 1;

    declare.in(src).out(dst);

    TEST_CYCLE() warpPerspective(src, dst, warpMat, sz, interType, borderMode, borderColor);

#ifdef __ANDROID__
    SANITY_CHECK(dst, interType == INTER_LINEAR ? 5 : 10);
#else
    SANITY_CHECK(dst, 1);
#endif
}

PERF_TEST_P( TestWarpPerspectiveNear_t, WarpPerspectiveNear,
             Combine(
                 Values( Size(640,480), Size(1920,1080), Size(2592,1944) ),
                 InterType::all(),
                 BorderMode::all(),
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
    Scalar borderColor = Scalar::all(150);

    Mat src(size, type), dst(size, type);
    cvtest::fillGradient(src);
    if(borderMode == BORDER_CONSTANT) cvtest::smoothBorder(src, borderColor, 1);
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

    declare.in(src).out(dst);
    declare.time(100);

    TEST_CYCLE()
    {
        warpPerspective( src, dst, warpMat, size, interType, borderMode, borderColor );
    }

#ifdef __ANDROID__
    SANITY_CHECK(dst, interType==INTER_LINEAR? 5 : 10);
#else
    SANITY_CHECK(dst, 1);
#endif
}

PERF_TEST_P( TestRemap, remap,
             Combine(
                 Values( CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1 ),
                 Values( szVGA, sz1080p ),
                 InterType::all(),
                 BorderMode::all(),
                 RemapMode::all()
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

    SANITY_CHECK_NOTHING();
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

PERF_TEST(Transform, getPerspectiveTransform_1000)
{
    unsigned int size = 8;
    Mat source(1, size/2, CV_32FC2);
    Mat destination(1, size/2, CV_32FC2);
    Mat transformCoefficient;

    declare.in(source, destination, WARMUP_RNG);

    PERF_SAMPLE_BEGIN()
    for (int i = 0; i < 1000; i++)
    {
        transformCoefficient = getPerspectiveTransform(source, destination);
    }
    PERF_SAMPLE_END()

    SANITY_CHECK_NOTHING();
}

} // namespace
