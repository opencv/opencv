// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {

CV_ENUM(BorderMode, BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT_101)

typedef TestBaseWithParam< tuple<Size, int, BorderMode> > TestFilter2d;
typedef TestBaseWithParam< tuple<string, int> > Image_KernelSize;

PERF_TEST_P( TestFilter2d, Filter2d,
             Combine(
                Values( Size(320, 240), sz1080p ),
                Values( 3, 5 ),
                BorderMode::all()
             )
)
{
    Size sz;
    int borderMode, kSize;
    sz         = get<0>(GetParam());
    kSize      = get<1>(GetParam());
    borderMode = get<2>(GetParam());

    Mat src(sz, CV_8UC4);
    Mat dst(sz, CV_8UC4);

    Mat kernel(kSize, kSize, CV_32FC1);
    randu(kernel, -3, 10);
    double s = fabs( sum(kernel)[0] );
    if(s > 1e-3) kernel /= s;

    declare.in(src, WARMUP_RNG).out(dst).time(20);

    TEST_CYCLE() cv::filter2D(src, dst, CV_8UC4, kernel, Point(1, 1), 0., borderMode);

    SANITY_CHECK(dst, 1);
}

PERF_TEST_P(TestFilter2d, DISABLED_Filter2d_ovx,
            Combine(
                Values(Size(320, 240), sz1080p),
                Values(3, 5),
                Values(BORDER_CONSTANT, BORDER_REPLICATE)
            )
)
{
    Size sz;
    int borderMode, kSize;
    sz = get<0>(GetParam());
    kSize = get<1>(GetParam());
    borderMode = get<2>(GetParam());

    Mat src(sz, CV_8UC1);
    Mat dst(sz, CV_16SC1);

    Mat kernel(kSize, kSize, CV_16SC1);
    randu(kernel, -3, 10);

    declare.in(src, WARMUP_RNG).out(dst).time(20);

    TEST_CYCLE() cv::filter2D(src, dst, CV_16SC1, kernel, Point(kSize / 2, kSize / 2), 0., borderMode);

    SANITY_CHECK(dst, 1);
}

PERF_TEST_P( Image_KernelSize, GaborFilter2d,
             Combine(
                 Values("stitching/a1.png", "cv/shared/pic5.png"),
                 Values(16, 32, 64) )
             )
{
    string fileName = getDataPath(get<0>(GetParam()));
    Mat sourceImage = imread(fileName, IMREAD_GRAYSCALE);
    if( sourceImage.empty() )
    {
        FAIL() << "Unable to load source image" << fileName;
    }

    int kernelSize = get<1>(GetParam());
    double sigma = 4;
    double lambda = 11;
    double theta = 47;
    double gamma = 0.5;
    Mat gaborKernel = getGaborKernel(Size(kernelSize, kernelSize), sigma, theta, lambda, gamma);
    Mat filteredImage;

    declare.in(sourceImage);

    TEST_CYCLE()
    {
        cv::filter2D(sourceImage, filteredImage, CV_32F, gaborKernel);
    }

    SANITY_CHECK(filteredImage, 1e-6, ERROR_RELATIVE);
}

// Performance test for the tiled parallel FilterEngine path (images >= 1MP).
// Exercises filter2D and sepFilter2D separately across common types and border modes.
typedef TestBaseWithParam< tuple<Size, int, BorderMode, bool> > ImgProc_ParallelFilter_Perf;

PERF_TEST_P( ImgProc_ParallelFilter_Perf, filter2D_parallel,
             Combine(
                 Values( Size(1280, 1024), sz1080p ),
                 Values( CV_8UC1, CV_8UC3, CV_32FC1 ),
                 Values( BORDER_DEFAULT, BORDER_CONSTANT ),
                 Values( false, true )   // false = filter2D, true = sepFilter2D
             )
)
{
    const Size  sz         = get<0>(GetParam());
    const int   type       = get<1>(GetParam());
    const int   borderMode = get<2>(GetParam());
    const bool  isSep      = get<3>(GetParam());

    Mat src(sz, type);
    Mat dst(sz, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if (isSep)
    {
        Mat kx = (Mat_<float>(1, 3) << 0.25f, 0.5f, 0.25f);
        Mat ky = (Mat_<float>(3, 1) << 0.25f, 0.5f, 0.25f);
        TEST_CYCLE() cv::sepFilter2D(src, dst, -1, kx, ky,
                                     Point(-1, -1), 0, borderMode);
    }
    else
    {
        Mat kernel = (Mat_<float>(3, 3) <<
            1/16.f, 2/16.f, 1/16.f,
            2/16.f, 4/16.f, 2/16.f,
            1/16.f, 2/16.f, 1/16.f);
        TEST_CYCLE() cv::filter2D(src, dst, -1, kernel,
                                  Point(-1, -1), 0, borderMode);
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
