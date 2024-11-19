// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {

CV_ENUM(BorderMode, BORDER_CONSTANT, BORDER_REPLICATE)
CV_ENUM(InterType, INTER_NEAREST, INTER_LINEAR)
CV_ENUM(InterTypeExtended, INTER_NEAREST, INTER_LINEAR, WARP_RELATIVE_MAP)

typedef TestBaseWithParam< tuple<Size, InterType, BorderMode, MatType> > TestWarpAffine;
typedef TestBaseWithParam< tuple<Size, InterType, BorderMode, MatType> > TestWarpPerspective;
typedef TestBaseWithParam< tuple<Size, InterType, BorderMode, MatType> > TestWarpPerspectiveNear_t;
typedef TestBaseWithParam< tuple<Size, InterTypeExtended, BorderMode, MatType> > TestRemap;

void update_map(const Mat& src, Mat& map_x, Mat& map_y, bool relative = false );

PERF_TEST_P( TestWarpAffine, WarpAffine,
             Combine(
                Values( szVGA, sz720p, sz1080p ),
                InterType::all(),
                BorderMode::all(),
                Values(CV_8UC3, CV_16UC3, CV_32FC3, CV_8UC1, CV_16UC1, CV_32FC1, CV_8UC4, CV_16UC4, CV_32FC4)
             )
)
{
    Size sz, szSrc(512, 512);
    int type, borderMode, interType;
    sz         = get<0>(GetParam());
    interType  = get<1>(GetParam());
    borderMode = get<2>(GetParam());
    type       = get<3>(GetParam());
    Scalar borderColor = Scalar::all(150);

    Mat src(szSrc,type), dst(sz, type);
    switch (src.depth()) {
        case CV_8U: {
            cvtest::fillGradient<uint8_t>(src);
            if(borderMode == BORDER_CONSTANT) cvtest::smoothBorder<uint8_t>(src, borderColor, 1);
            break;
        }
        case CV_16U: {
            cvtest::fillGradient<uint16_t>(src);
            if(borderMode == BORDER_CONSTANT) cvtest::smoothBorder<uint16_t>(src, borderColor, 1);
            break;
        }
        case CV_32F: {
            cvtest::fillGradient<float>(src);
            if(borderMode == BORDER_CONSTANT) cvtest::smoothBorder<float>(src, borderColor, 1);
            break;
        }
    }
    Mat warpMat = getRotationMatrix2D(Point2f(src.cols/2.f, src.rows/2.f), 30., 2.2);
    declare.in(src).out(dst);

    TEST_CYCLE() warpAffine( src, dst, warpMat, sz, interType, borderMode, borderColor );

    SANITY_CHECK(dst, 1);
}

PERF_TEST_P( TestWarpPerspective, WarpPerspective,
             Combine(
                Values( szVGA, sz720p, sz1080p ),
                InterType::all(),
                BorderMode::all(),
                Values(CV_8UC3, CV_16UC3, CV_32FC3, CV_8UC1, CV_16UC1, CV_32FC1, CV_8UC4, CV_16UC4, CV_32FC4)
             )
)
{
    Size sz, szSrc(512, 512);
    int type, borderMode, interType;
    sz         = get<0>(GetParam());
    interType  = get<1>(GetParam());
    borderMode = get<2>(GetParam());
    type       = get<3>(GetParam());
    Scalar borderColor = Scalar::all(150);

    Mat src(szSrc, type), dst(sz, type);
    switch (src.depth()) {
        case CV_8U: {
            cvtest::fillGradient<uint8_t>(src);
            if(borderMode == BORDER_CONSTANT) cvtest::smoothBorder<uint8_t>(src, borderColor, 1);
            break;
        }
        case CV_16U: {
            cvtest::fillGradient<uint16_t>(src);
            if(borderMode == BORDER_CONSTANT) cvtest::smoothBorder<uint16_t>(src, borderColor, 1);
            break;
        }
        case CV_32F: {
            cvtest::fillGradient<float>(src);
            if(borderMode == BORDER_CONSTANT) cvtest::smoothBorder<float>(src, borderColor, 1);
            break;
        }
    }

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

    SANITY_CHECK(dst, 1);
}

PERF_TEST_P( TestRemap, map1_32fc1,
             Combine(
                 Values( szVGA, sz1080p ),
                 InterTypeExtended::all(),
                 BorderMode::all(),
                 Values(CV_8UC3, CV_16UC3, CV_32FC3, CV_8UC1, CV_16UC1, CV_32FC1, CV_8UC4, CV_16UC4, CV_32FC4)
                 )
             )
{
    Size size = get<0>(GetParam());
    int interpolationType = get<1>(GetParam());
    int borderMode = get<2>(GetParam());
    int type = get<3>(GetParam());
    unsigned int height = size.height;
    unsigned int width = size.width;
    Mat source(height, width, type);
    Mat destination;
    Mat map_x(height, width, CV_32F);
    Mat map_y(height, width, CV_32F);

    declare.in(source, WARMUP_RNG);

    update_map(source, map_x, map_y, ((interpolationType & WARP_RELATIVE_MAP) != 0));

    TEST_CYCLE()
    {
        remap(source, destination, map_x, map_y, interpolationType, borderMode);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P( TestRemap, map1_32fc2,
             Combine(
                 Values( szVGA, sz1080p ),
                 InterTypeExtended::all(),
                 BorderMode::all(),
                 Values(CV_8UC3, CV_16UC3, CV_32FC3, CV_8UC1, CV_16UC1, CV_32FC1, CV_8UC4, CV_16UC4, CV_32FC4)
                 )
             )
{
    Size size = get<0>(GetParam());
    int interpolationType = get<1>(GetParam());
    int borderMode = get<2>(GetParam());
    int type = get<3>(GetParam());
    unsigned int height = size.height;
    unsigned int width = size.width;
    Mat source(height, width, type);
    Mat destination;
    Mat map_x(height, width, CV_32FC2);
    Mat map_y;

    declare.in(source, WARMUP_RNG);

    update_map(source, map_x, map_y, ((interpolationType & WARP_RELATIVE_MAP) != 0));

    TEST_CYCLE()
    {
        remap(source, destination, map_x, map_y, interpolationType, borderMode);
    }

    SANITY_CHECK_NOTHING();
}

void update_map(const Mat& src, Mat& map_x, Mat& map_y, bool relative )
{
    if (map_y.empty()) {
        float *ptr_x = map_x.ptr<float>();
        for (int j = 0; j < src.rows; j++) {
            for (int i = 0; i < src.cols; i++) {
                size_t offset = 2 * j * src.cols + 2 * i;
                if( i > src.cols*0.25 && i < src.cols*0.75 && j > src.rows*0.25 && j < src.rows*0.75 )
                {
                    ptr_x[offset]   = 2*( i - src.cols*0.25f ) + 0.5f ;
                    ptr_x[offset+1] = 2*( j - src.rows*0.25f ) + 0.5f ;
                }
                else
                {
                    ptr_x[offset]   = 0 ;
                    ptr_x[offset+1] = 0 ;
                }

                if( relative )
                {
                    ptr_x[offset]   -= static_cast<float>(i) ;
                    ptr_x[offset+1] -= static_cast<float>(j) ;
                }
            }
        }
    } else {
        for( int j = 0; j < src.rows; j++ )
        {
            for( int i = 0; i < src.cols; i++ )
            {
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

                if( relative )
                {
                    map_x.at<float>(j,i) -= static_cast<float>(i);
                    map_y.at<float>(j,i) -= static_cast<float>(j);
                }
            }
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

PERF_TEST(Transform, getPerspectiveTransform_QR_1000)
{
    unsigned int size = 8;
    Mat source(1, size/2, CV_32FC2);
    Mat destination(1, size/2, CV_32FC2);
    Mat transformCoefficient;

    declare.in(source, destination, WARMUP_RNG);

    PERF_SAMPLE_BEGIN()
    for (int i = 0; i < 1000; i++)
    {
        transformCoefficient = getPerspectiveTransform(source, destination, DECOMP_QR);
    }
    PERF_SAMPLE_END()

    SANITY_CHECK_NOTHING();
}

} // namespace
