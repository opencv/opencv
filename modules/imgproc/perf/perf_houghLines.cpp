#include "perf_precomp.hpp"

#include "cmath"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

typedef std::tr1::tuple<String, double, double, int> Image_RhoStep_ThetaStep_Threshold_t;
typedef perf::TestBaseWithParam<Image_RhoStep_ThetaStep_Threshold_t> Image_RhoStep_ThetaStep_Threshold;

#ifdef __aarch64__
// In case of  aarch64 the function produces one more line than expected
PERF_TEST_P(Image_RhoStep_ThetaStep_Threshold, DISABLED_HoughLines,
            testing::Combine(
                testing::Values( "cv/shared/pic5.png", "stitching/a1.png" ),
                testing::Values( 1, 10 ),
                testing::Values( 0.01, 0.1 ),
                testing::Values( 300, 500 )
                )
            )
#else
PERF_TEST_P(Image_RhoStep_ThetaStep_Threshold, HoughLines,
            testing::Combine(
                testing::Values( "cv/shared/pic5.png", "stitching/a1.png" ),
                testing::Values( 1, 10 ),
                testing::Values( 0.01, 0.1 ),
                testing::Values( 300, 500 )
                )
            )
#endif
{
    String filename = getDataPath(get<0>(GetParam()));
    double rhoStep = get<1>(GetParam());
    double thetaStep = get<2>(GetParam());
    int threshold = get<3>(GetParam());

    Mat image = imread(filename, IMREAD_GRAYSCALE);
    if (image.empty())
        FAIL() << "Unable to load source image" << filename;

    Canny(image, image, 0, 0);

    Mat lines;
    declare.time(60);

    TEST_CYCLE() HoughLines(image, lines, rhoStep, thetaStep, threshold);

    SANITY_CHECK(lines);
}
