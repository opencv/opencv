// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

///////////// HoughLines //////////////////////

struct Vec2fComparator
{
    bool operator()(const Vec2f& a, const Vec2f b) const
    {
        if(a[0] != b[0]) return a[0] < b[0];
        else return a[1] < b[1];
    }
};

typedef tuple<Size, double, double> ImageSize_RhoStep_ThetaStep_t;
typedef TestBaseWithParam<ImageSize_RhoStep_ThetaStep_t> HoughLinesFixture;

OCL_PERF_TEST_P(HoughLinesFixture, HoughLines, Combine(OCL_TEST_SIZES,
                                                       Values( 0.1, 1 ),
                                                       Values( CV_PI / 180.0, 0.1 )))
{
    const Size srcSize = get<0>(GetParam());
    double rhoStep = get<1>(GetParam());
    double thetaStep = get<2>(GetParam());
    int threshold = 250;

    UMat usrc(srcSize, CV_8UC1), lines(1, 1, CV_32FC2);
    Mat src(srcSize, CV_8UC1);
    src.setTo(Scalar::all(0));
    line(src, Point(0, 100), Point(src.cols, 100), Scalar::all(255), 1);
    line(src, Point(0, 200), Point(src.cols, 200), Scalar::all(255), 1);
    line(src, Point(0, 400), Point(src.cols, 400), Scalar::all(255), 1);
    line(src, Point(100, 0), Point(100, src.rows), Scalar::all(255), 1);
    line(src, Point(200, 0), Point(200, src.rows), Scalar::all(255), 1);
    line(src, Point(400, 0), Point(400, src.rows), Scalar::all(255), 1);
    src.copyTo(usrc);

    declare.in(usrc).out(lines);

    OCL_TEST_CYCLE() cv::HoughLines(usrc, lines, rhoStep, thetaStep, threshold);

    Mat result;
    lines.copyTo(result);
    std::sort(result.begin<Vec2f>(), result.end<Vec2f>(), Vec2fComparator());

    SANITY_CHECK(result, 1e-6);
}

///////////// HoughLinesP /////////////////////

typedef tuple<string, double, double> Image_RhoStep_ThetaStep_t;
typedef TestBaseWithParam<Image_RhoStep_ThetaStep_t> HoughLinesPFixture;

OCL_PERF_TEST_P(HoughLinesPFixture, HoughLinesP, Combine(Values("cv/shared/pic5.png", "stitching/a1.png"),
                                                         Values( 0.1, 1 ),
                                                         Values( CV_PI / 180.0, 0.1 )))
{
    string filename = get<0>(GetParam());
    double rhoStep = get<1>(GetParam());
    double thetaStep = get<2>(GetParam());
    int threshold = 100;
    double minLineLength = 50, maxGap = 5;

    Mat image = imread(getDataPath(filename), IMREAD_GRAYSCALE);
    Canny(image, image, 50, 200, 3);
    UMat usrc, lines(1, 1, CV_32SC4);
    image.copyTo(usrc);

    declare.in(usrc).out(lines);

    OCL_TEST_CYCLE() cv::HoughLinesP(usrc, lines, rhoStep, thetaStep, threshold, minLineLength, maxGap);

    EXPECT_NE((int) lines.total(), 0);
    SANITY_CHECK_NOTHING();
}

} } // namespace opencv_test::ocl

#endif // HAVE_OPENCL