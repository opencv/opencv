#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using namespace testing;
using std::tr1::make_tuple;
using std::tr1::get;


CV_ENUM(BorderMode, BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT_101)

typedef TestBaseWithParam< tr1::tuple<Size, int, BorderMode> > TestFilter2d;
typedef TestBaseWithParam< tr1::tuple<string, int> > Image_KernelSize;

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

    TEST_CYCLE() filter2D(src, dst, CV_8UC4, kernel, Point(1, 1), 0., borderMode);

    SANITY_CHECK(dst, 1);
}

PERF_TEST_P(TestFilter2d, Filter2d_ovx,
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

    TEST_CYCLE() filter2D(src, dst, CV_16SC1, kernel, Point(kSize / 2, kSize / 2), 0., borderMode);

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
        filter2D(sourceImage, filteredImage, CV_32F, gaborKernel);
    }

    SANITY_CHECK(filteredImage, 1e-6, ERROR_RELATIVE);
}
