// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {

typedef tuple<string, double, double, double> Image_RhoStep_ThetaStep_Threshold_t;
typedef perf::TestBaseWithParam<Image_RhoStep_ThetaStep_Threshold_t> Image_RhoStep_ThetaStep_Threshold;

PERF_TEST_P(Image_RhoStep_ThetaStep_Threshold, HoughLines,
            testing::Combine(
                testing::Values( "cv/shared/pic5.png", "stitching/a1.png" ),
                testing::Values( 1, 10 ),
                testing::Values( 0.01, 0.1 ),
                testing::Values( 0.5, 1.1 )
                )
            )
{
    string filename = getDataPath(get<0>(GetParam()));
    double rhoStep = get<1>(GetParam());
    double thetaStep = get<2>(GetParam());
    double threshold_ratio = get<3>(GetParam());

    Mat image = imread(filename, IMREAD_GRAYSCALE);
    if (image.empty())
        FAIL() << "Unable to load source image" << filename;

    Canny(image, image, 32, 128);

    // add some synthetic lines:
    line(image, Point(0, 0), Point(image.cols, image.rows), Scalar::all(255), 3);
    line(image, Point(image.cols, 0), Point(image.cols/2, image.rows), Scalar::all(255), 3);

    vector<Vec2f> lines;
    declare.time(60);

    int hough_threshold = (int)(std::min(image.cols, image.rows) * threshold_ratio);

    TEST_CYCLE() HoughLines(image, lines, rhoStep, thetaStep, hough_threshold);

    printf("%dx%d: %d lines\n", image.cols, image.rows, (int)lines.size());

    if (threshold_ratio < 1.0)
    {
        EXPECT_GE(lines.size(), 2u);
    }

    EXPECT_LT(lines.size(), 3000u);

#if 0
    cv::cvtColor(image,image,cv::COLOR_GRAY2BGR);
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line(image, pt1, pt2, Scalar(0,0,255), 1, cv::LINE_AA);
    }
    cv::imshow("result", image);
    cv::waitKey();
#endif

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Image_RhoStep_ThetaStep_Threshold, HoughLines3f,
            testing::Combine(
                testing::Values( "cv/shared/pic5.png", "stitching/a1.png" ),
                testing::Values( 1, 10 ),
                testing::Values( 0.01, 0.1 ),
                testing::Values( 0.5, 1.1 )
                )
            )
{
    string filename = getDataPath(get<0>(GetParam()));
    double rhoStep = get<1>(GetParam());
    double thetaStep = get<2>(GetParam());
    double threshold_ratio = get<3>(GetParam());

    Mat image = imread(filename, IMREAD_GRAYSCALE);
    if (image.empty())
        FAIL() << "Unable to load source image" << filename;

    Canny(image, image, 32, 128);

    // add some synthetic lines:
    line(image, Point(0, 0), Point(image.cols, image.rows), Scalar::all(255), 3);
    line(image, Point(image.cols, 0), Point(image.cols/2, image.rows), Scalar::all(255), 3);

    vector<Vec3f> lines;
    declare.time(60);

    int hough_threshold = (int)(std::min(image.cols, image.rows) * threshold_ratio);

    TEST_CYCLE() HoughLines(image, lines, rhoStep, thetaStep, hough_threshold);

    printf("%dx%d: %d lines\n", image.cols, image.rows, (int)lines.size());

    if (threshold_ratio < 1.0)
    {
        EXPECT_GE(lines.size(), 2u);
    }

    EXPECT_LT(lines.size(), 3000u);

    SANITY_CHECK_NOTHING();
}

typedef tuple<Size, int> Size_Density_t;
typedef perf::TestBaseWithParam<Size_Density_t> Size_Density;

PERF_TEST_P(Size_Density, GeneralizedHoughGuil,
            testing::Combine(
                testing::Values( szQVGA, szVGA ),
                testing::Values( 2, 6, 16 )
                )
            )
{
    Size sz = get<0>(GetParam());
    int density = get<1>(GetParam());

    Mat templ = Mat::zeros(96, 96, CV_8UC1);
    const Point poly[] = { Point(10, 20), Point(40, 6), Point(84, 18), Point(88, 60),
                           Point(60, 88), Point(30, 80), Point(8, 56) };
    fillConvexPoly(templ, poly, 7, Scalar(200));
    circle(templ, Point(48, 46), 18, Scalar(90), FILLED);
    rectangle(templ, Point(40, 60), Point(70, 72), Scalar(30), FILLED);

    Mat image = Mat::zeros(sz, CV_8UC1);
    RNG rng(0x1234);
    int nseg = (int)((double)sz.area() * density / 10000);
    for (int i = 0; i < nseg; i++)
    {
        Point p1(rng.uniform(0, sz.width), rng.uniform(0, sz.height));
        double a = rng.uniform(0., CV_2PI);
        Point p2(p1.x + cvRound(12 * cos(a)), p1.y + cvRound(12 * sin(a)));
        line(image, p1, p2, Scalar(rng.uniform(120, 256)), 1);
    }
    Mat rot = getRotationMatrix2D(Point2f(48, 48), 10, 1.0);
    rot.at<double>(0, 2) += sz.width / 2 - 48;
    rot.at<double>(1, 2) += sz.height / 2 - 48;
    Mat placed;
    warpAffine(templ, placed, rot, sz);
    image = max(image, placed);

    Ptr<GeneralizedHoughGuil> guil = createGeneralizedHoughGuil();
    guil->setMinDist(10);
    guil->setLevels(360);
    guil->setDp(2);
    guil->setMaxBufferSize(1000);
    guil->setMinAngle(0);
    guil->setMaxAngle(20);
    guil->setAngleStep(1);
    guil->setAngleThresh(1);
    guil->setMinScale(0.9);
    guil->setMaxScale(1.1);
    guil->setScaleStep(0.05);
    guil->setScaleThresh(1);
    guil->setPosThresh(20);
    guil->setTemplate(templ);

    vector<Vec4f> positions;
    vector<Vec3i> votes;
    declare.in(image).time(60);

    TEST_CYCLE() guil->detect(image, positions, votes);

    EXPECT_FALSE(positions.empty());

    SANITY_CHECK_NOTHING();
}

} // namespace
