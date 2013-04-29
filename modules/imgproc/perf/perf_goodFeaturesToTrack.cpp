#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

typedef std::tr1::tuple<string, int, double, int, bool> Image_MaxCorners_QualityLevel_MinDistance_BlockSize_UseHarris_t;
typedef perf::TestBaseWithParam<Image_MaxCorners_QualityLevel_MinDistance_BlockSize_UseHarris_t> Image_MaxCorners_QualityLevel_MinDistance_BlockSize_UseHarris;

PERF_TEST_P(Image_MaxCorners_QualityLevel_MinDistance_BlockSize_UseHarris, goodFeaturesToTrack,
            testing::Combine(
                testing::Values( "stitching/a1.png", "cv/shared/pic5.png"),
                testing::Values( 100, 500 ),
                testing::Values( 0.1, 0.01 ),
                testing::Values( 3, 5 ),
                testing::Bool()
                )
          )
{
    string filename = getDataPath(get<0>(GetParam()));
    int maxCorners = get<1>(GetParam());
    double qualityLevel = get<2>(GetParam());
    int blockSize = get<3>(GetParam());
    bool useHarrisDetector = get<4>(GetParam());

    Mat image = imread(filename, IMREAD_GRAYSCALE);
    if (image.empty())
        FAIL() << "Unable to load source image" << filename;

    std::vector<Point2f> corners;

    double minDistance = 1;
    TEST_CYCLE() goodFeaturesToTrack(image, corners, maxCorners, qualityLevel, minDistance, noArray(), blockSize, useHarrisDetector);

    if (corners.size() > 50)
        corners.erase(corners.begin() + 50, corners.end());

    SANITY_CHECK(corners);
}
