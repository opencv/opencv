#include "perf_precomp.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"

using namespace std;
using namespace cv;
using namespace perf;

PERF_TEST(PerfHoughCircles, Basic)
{
    string filename = getDataPath("cv/imgproc/stuff.jpg");
    const double dp = 1.0;
    double minDist = 20;
    double edgeThreshold = 20;
    double accumThreshold = 30;
    int minRadius = 20;
    int maxRadius = 200;

    Mat img = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty()) << "Unable to load source image " << filename;

    GaussianBlur(img, img, Size(9, 9), 2, 2);

    vector<Vec3f> circles;
    declare.in(img);

    TEST_CYCLE()
    {
        HoughCircles(img, circles, CV_HOUGH_GRADIENT, dp, minDist, edgeThreshold, accumThreshold, minRadius, maxRadius);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST(PerfHoughCircles2, ManySmallCircles)
{
    string filename = getDataPath("cv/imgproc/beads.jpg");
    const double dp = 1.0;
    double minDist = 10;
    double edgeThreshold = 90;
    double accumThreshold = 11;
    int minRadius = 7;
    int maxRadius = 18;

    Mat img = imread(filename, IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty()) << "Unable to load source image " << filename;

    vector<Vec3f> circles;
    declare.in(img);

    TEST_CYCLE()
    {
        HoughCircles(img, circles, CV_HOUGH_GRADIENT, dp, minDist, edgeThreshold, accumThreshold, minRadius, maxRadius);
    }

    SANITY_CHECK_NOTHING();
}
