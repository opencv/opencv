/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

// Seeds are eroded copies of the stored reference regions, so watershed has to re-grow the
// boundaries. The neighbouring watershed/comp.xml is not used: it is an
// "opencv-sequence-tree" holding CvSeq contours and needs the removed C API to read.
TEST(Imgproc_Watershed, regression)
{
    const string folder = string(cvtest::TS::ptr()->get_data_path());
    const string expPath = folder + "watershed/wshed_exp.png";

    Mat image = imread(folder + "inpaint/orig.png", IMREAD_COLOR);
    Mat expLabels8 = imread(expPath, IMREAD_GRAYSCALE);
    ASSERT_FALSE(image.empty()) << "Could not read " << folder << "inpaint/orig.png";
    ASSERT_FALSE(expLabels8.empty()) << "Could not read " << expPath;
    ASSERT_EQ(image.size(), expLabels8.size());

    // wshed_exp.png stores the expected labels offset by +1, so that the -1 used for
    // watershed boundaries survives being written to an 8-bit PNG.
    Mat expected;
    expLabels8.convertTo(expected, CV_32S);
    expected -= 1;                       // -1 = boundary, 1..nLabels = regions

    double maxLabel = 0;
    minMaxLoc(expected, 0, &maxLabel);
    const int nLabels = cvRound(maxLabel);
    ASSERT_GT(nLabels, 1) << "reference image does not contain several regions";

    Mat markers = Mat::zeros(expected.size(), CV_32S);
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
    for (int label = 1; label <= nLabels; label++)
    {
        Mat seed;
        erode(expected == label, seed, kernel);
        ASSERT_GT(countNonZero(seed), 0) << "region " << label << " vanished when eroded";
        markers.setTo(label, seed);
    }

    Mat result = markers.clone();
    watershed(image, result);

    ASSERT_EQ(expected.size(), result.size());
    ASSERT_EQ(CV_32S, result.type());

    EXPECT_EQ(0, countNonZero(result == 0)) << "watershed left pixels unlabelled";

    double minVal = 0, maxVal = 0;
    minMaxLoc(result, &minVal, &maxVal);
    EXPECT_GE(minVal, -1);
    EXPECT_LE(maxVal, nLabels) << "watershed produced a label that was never seeded";

    // Measured: ~99.6% for a correct implementation, ~78.7% if watershed does nothing.
    Mat interior = expected > 0;
    const int totalPx = countNonZero(interior);
    ASSERT_GT(totalPx, 0);
    Mat agreeMask;
    bitwise_and(result == expected, interior, agreeMask);
    const double agreement = (double)countNonZero(agreeMask) / totalPx;
    EXPECT_GT(agreement, 0.95)
        << "only " << agreement * 100 << "% of interior pixels match " << expPath;

    Mat again = markers.clone();
    watershed(image, again);
    EXPECT_EQ(0, countNonZero(again != result)) << "watershed is not deterministic";
}

}} // namespace
