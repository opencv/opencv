// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, Intel Corporation, all rights reserved.
#include "perf_precomp.hpp"

namespace opencv_test {

typedef tuple<string, int, bool, tuple<double, double> > Img_Aperture_L2_thresholds_t;
typedef perf::TestBaseWithParam<Img_Aperture_L2_thresholds_t> Img_Aperture_L2_thresholds;

PERF_TEST_P(Img_Aperture_L2_thresholds, canny,
            testing::Combine(
                testing::Values( "cv/shared/lena.png", "stitching/b1.png", "cv/detectors_descriptors_evaluation/images_datasets/leuven/img1.png" ),
                testing::Values( 3, 5 ),
                testing::Bool(),
                testing::Values( make_tuple(50.0, 100.0), make_tuple(0.0, 50.0), make_tuple(100.0, 120.0) )
                )
            )
{
    string filename = getDataPath(get<0>(GetParam()));
    int aperture = get<1>(GetParam());
    bool useL2 = get<2>(GetParam());
    double thresh_low = get<0>(get<3>(GetParam()));
    double thresh_high = get<1>(get<3>(GetParam()));

    Mat img = imread(filename, IMREAD_GRAYSCALE);
    if (img.empty())
        FAIL() << "Unable to load source image " << filename;
    Mat edges(img.size(), img.type());

    declare.in(img).out(edges);

    PERF_SAMPLE_BEGIN();
        Canny(img, edges, thresh_low, thresh_high, aperture, useL2);
    PERF_SAMPLE_END();

    SANITY_CHECK(edges);
}

typedef tuple<string, bool> Img_L2_t;
typedef perf::TestBaseWithParam<Img_L2_t> Img_L2;

PERF_TEST_P(Img_L2, cannyDeriv,
            testing::Combine(
                testing::Values("cv/shared/lena.png", "stitching/b1.png"),
                testing::Bool()
                ))
{
    string filename = getDataPath(get<0>(GetParam()));
    bool useL2 = get<1>(GetParam());

    Mat img = imread(filename, IMREAD_GRAYSCALE);
    if (img.empty())
        FAIL() << "Unable to load source image " << filename;

    Mat dx, dy;
    cv::Sobel(img, dx, CV_16S, 1, 0);
    cv::Sobel(img, dy, CV_16S, 0, 1);

    Mat edges(img.size(), CV_8UC1);
    double low = 50, high = 100;

    declare.in(dx, dy).out(edges);

    TEST_CYCLE() cv::Canny(dx, dy, edges, low, high, useL2);

    SANITY_CHECK_NOTHING();
}

} // namespace
