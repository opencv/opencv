#include "perf_precomp.hpp"

#include "cmath"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

typedef std::tr1::tuple<string, double, double, int> Image_RhoStep_ThetaStep_Threshold_t;
typedef perf::TestBaseWithParam<Image_RhoStep_ThetaStep_Threshold_t> Image_RhoStep_ThetaStep_Threshold;

PERF_TEST_P(Image_RhoStep_ThetaStep_Threshold, HoughLines,
            testing::Combine(
                testing::Values( "cv/shared/pic5.png", "stitching/a1.png" ),
                testing::Values( 1, 10 ),
                testing::Values( 0.05, 0.1 ),
                testing::Values( 80 , 150 )
                )
            )
{
    string filename = getDataPath(get<0>(GetParam()));
    double rhoStep = get<1>(GetParam());
    double thetaStep = get<2>(GetParam());
    int threshold = get<3>(GetParam());

    Mat image = imread(filename, IMREAD_GRAYSCALE);
    if (image.empty())
        FAIL() << "Unable to load source image" << filename;

    Canny(image, image, 100, 150, 3);

    Mat lines;
    declare.time(60);

    TEST_CYCLE() HoughLines(image, lines, rhoStep, thetaStep, threshold);

    EXPECT_FALSE(lines.empty());
    SANITY_CHECK_NOTHING();
}
