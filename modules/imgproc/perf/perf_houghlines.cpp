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

    // add some syntetic lines:
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

    // add some syntetic lines:
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

} // namespace
