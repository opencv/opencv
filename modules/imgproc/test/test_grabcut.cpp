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

class CV_GrabcutTest : public cvtest::BaseTest
{
public:
    CV_GrabcutTest();
    ~CV_GrabcutTest();
protected:
    bool verify(const Mat& mask, const Mat& exp);
    void run(int);
};

CV_GrabcutTest::CV_GrabcutTest() {}
CV_GrabcutTest::~CV_GrabcutTest() {}

bool CV_GrabcutTest::verify(const Mat& mask, const Mat& exp)
{
    const float maxDiffRatio = 0.005f;
    int expArea = countNonZero( exp );
    int nonIntersectArea = countNonZero( mask != exp );

    float curRatio = (float)nonIntersectArea / (float)expArea;
    ts->printf( cvtest::TS::LOG, "nonIntersectArea/expArea = %f\n", curRatio );
    return curRatio < maxDiffRatio;
}

void CV_GrabcutTest::run( int /* start_from */)
{
    cvtest::DefaultRngAuto defRng;

    Mat img = imread(string(ts->get_data_path()) + "shared/airplane.png");
    Mat mask_prob = imread(string(ts->get_data_path()) + "grabcut/mask_prob.png", 0);
    Mat exp_mask1 = imread(string(ts->get_data_path()) + "grabcut/exp_mask1.png", 0);
    Mat exp_mask2 = imread(string(ts->get_data_path()) + "grabcut/exp_mask2.png", 0);

    if (img.empty() || (!mask_prob.empty() && img.size() != mask_prob.size()) ||
                       (!exp_mask1.empty() && img.size() != exp_mask1.size()) ||
                       (!exp_mask2.empty() && img.size() != exp_mask2.size()) )
    {
         ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
         return;
    }

    Rect rect(Point(24, 126), Point(483, 294));
    Mat exp_bgdModel, exp_fgdModel;

    Mat mask;
    Mat bgdModel, fgdModel;
    grabCut( img, mask, rect, bgdModel, fgdModel, 0, GC_INIT_WITH_RECT );
    bgdModel.copyTo(exp_bgdModel);
    fgdModel.copyTo(exp_fgdModel);
    grabCut( img, mask, rect, bgdModel, fgdModel, 2, GC_EVAL_FREEZE_MODEL );

    // Multiply images by 255 for more visuality of test data.
    if( mask_prob.empty() )
    {
        mask.copyTo( mask_prob );
        imwrite(string(ts->get_data_path()) + "grabcut/mask_prob.png", mask_prob);
    }
    if( exp_mask1.empty() )
    {
        exp_mask1 = (mask & 1) * 255;
        imwrite(string(ts->get_data_path()) + "grabcut/exp_mask1.png", exp_mask1);
    }
    if (!verify((mask & 1) * 255, exp_mask1))
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        return;
    }
    // The model should not be changed after calling with GC_EVAL_FREEZE_MODEL
    double sumBgdModel = cv::sum(cv::abs(bgdModel) - cv::abs(exp_bgdModel))[0];
    double sumFgdModel = cv::sum(cv::abs(fgdModel) - cv::abs(exp_fgdModel))[0];
    if (sumBgdModel >= 0.1 || sumFgdModel >= 0.1)
    {
        ts->printf(cvtest::TS::LOG, "sumBgdModel = %f, sumFgdModel = %f\n", sumBgdModel, sumFgdModel);
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        return;
    }

    mask = mask_prob;
    bgdModel.release();
    fgdModel.release();
    rect = Rect();
    grabCut( img, mask, rect, bgdModel, fgdModel, 0, GC_INIT_WITH_MASK );
    grabCut( img, mask, rect, bgdModel, fgdModel, 1, GC_EVAL );

    if( exp_mask2.empty() )
    {
        exp_mask2 = (mask & 1) * 255;
        imwrite(string(ts->get_data_path()) + "grabcut/exp_mask2.png", exp_mask2);
    }
    if (!verify((mask & 1) * 255, exp_mask2))
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        return;
    }
    ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Imgproc_GrabCut, regression) { CV_GrabcutTest test; test.safe_run(); }

TEST(Imgproc_GrabCut, repeatability)
{
    cvtest::TS& ts = *cvtest::TS::ptr();

    Mat image_1 = imread(string(ts.get_data_path()) + "grabcut/image1652.ppm", IMREAD_COLOR);
    Mat mask_1 = imread(string(ts.get_data_path()) + "grabcut/mask1652.ppm", IMREAD_GRAYSCALE);
    Rect roi_1(0, 0, 150, 150);

    Mat image_2 = image_1.clone();
    Mat mask_2 = mask_1.clone();
    Rect roi_2 = roi_1;

    Mat image_3 = image_1.clone();
    Mat mask_3 = mask_1.clone();
    Rect roi_3 = roi_1;

    Mat bgdModel_1, fgdModel_1;
    Mat bgdModel_2, fgdModel_2;
    Mat bgdModel_3, fgdModel_3;

    theRNG().state = 12378213;
    grabCut(image_1, mask_1, roi_1, bgdModel_1, fgdModel_1, 1, GC_INIT_WITH_MASK);
    theRNG().state = 12378213;
    grabCut(image_2, mask_2, roi_2, bgdModel_2, fgdModel_2, 1, GC_INIT_WITH_MASK);
    theRNG().state = 12378213;
    grabCut(image_3, mask_3, roi_3, bgdModel_3, fgdModel_3, 1, GC_INIT_WITH_MASK);

    EXPECT_EQ(0, countNonZero(mask_1 != mask_2));
    EXPECT_EQ(0, countNonZero(mask_1 != mask_3));
    EXPECT_EQ(0, countNonZero(mask_2 != mask_3));
}

TEST(Imgproc_GrabCut, multi_iter_lut_and_direct_equivalence)
{
    RNG& rng = theRNG();
    // Test both small (inline exp) and medium (table LUT: 4 * 200 * 200 * 3 = 480000 >= 390152) sizes
    for (int side : {64, 200})
    {
        Mat img(side, side, CV_8UC3);
        rng.fill(img, RNG::UNIFORM, 0, 40);
        Rect fgRect(side / 4, side / 4, side / 2, side / 2);
        Mat fgRoi = img(fgRect);
        rng.fill(fgRoi, RNG::UNIFORM, 200, 255);

        Mat maskMulti, bgd1, fgd1;
        uint64 savedState = rng.state;
        grabCut(img, maskMulti, fgRect, bgd1, fgd1, 3, GC_INIT_WITH_RECT);

        // Step-by-step 1-iter init + 2 x 1-iter GC_EVAL must match 3-iter GC_INIT_WITH_RECT exactly
        rng.state = savedState;
        Mat maskStep, bgd2, fgd2;
        grabCut(img, maskStep, fgRect, bgd2, fgd2, 1, GC_INIT_WITH_RECT);
        grabCut(img, maskStep, Rect(), bgd2, fgd2, 1, GC_EVAL);
        grabCut(img, maskStep, Rect(), bgd2, fgd2, 1, GC_EVAL);

        EXPECT_EQ(0, countNonZero(maskMulti != maskStep)) << "side=" << side;
        Mat fgExtracted = (maskMulti == GC_PR_FGD) | (maskMulti == GC_FGD);
        EXPECT_GT(countNonZero(fgExtracted(fgRect)), fgRect.area() * 8 / 10);
    }
}

TEST(Imgproc_GrabCut, pruned_graph_edge_cases)
{
    RNG rng(20261113ULL);
    Mat img(120, 160, CV_8UC3);
    rng.fill(img, RNG::UNIFORM, 10, 45);
    Rect fgRect(35, 25, 90, 70);
    rng.fill(img(fgRect), RNG::UNIFORM, 200, 250);

    // Case 1: GC_INIT_WITH_RECT recovers planted foreground accurately
    {
        theRNG().state = 1234567ULL;
        Mat mask, bgdModel, fgdModel;
        grabCut(img, mask, fgRect, bgdModel, fgdModel, 3, GC_INIT_WITH_RECT);
        for (int y = 0; y < img.rows; ++y)
        {
            const uchar* m = mask.ptr<uchar>(y);
            for (int x = 0; x < img.cols; ++x)
            {
                if (!fgRect.contains(Point(x, y)))
                    EXPECT_EQ((int)GC_BGD, (int)m[x]);
            }
        }
        Mat fg = (mask == GC_PR_FGD) | (mask == GC_FGD);
        EXPECT_GT(countNonZero(fg(fgRect)), fgRect.area() * 9 / 10);
    }

    // Case 2: 1x1 rect (single uncertain pixel, edgePairCount == 0) + isolated uncertain pixels in INIT_WITH_MASK
    {
        theRNG().state = 1234567ULL;
        Mat mask1x1Same, bgdModel, fgdModel;
        // Same-color neighbors all marked GC_BGD pull the isolated pixel to GC_PR_BGD
        grabCut(img, mask1x1Same, Rect(80, 60, 1, 1), bgdModel, fgdModel, 2, GC_INIT_WITH_RECT);
        EXPECT_EQ((int)GC_PR_BGD, (int)mask1x1Same.at<uchar>(60, 80));

        // Distinct bright isolated pixel inside dark GC_BGD neighborhood stays GC_PR_FGD
        Mat imgSpot(40, 40, CV_8UC3, Scalar(20, 20, 20));
        imgSpot.at<Vec3b>(20, 20) = Vec3b(245, 245, 245);
        Mat maskSpot, bgdSpot, fgdSpot;
        theRNG().state = 1234567ULL;
        grabCut(imgSpot, maskSpot, Rect(20, 20, 1, 1), bgdSpot, fgdSpot, 2, GC_INIT_WITH_RECT);
        EXPECT_EQ((int)GC_PR_FGD, (int)maskSpot.at<uchar>(20, 20));

        // Mask with uncertain interior + isolated uncertain border pixel (5, 5)
        Mat maskIso(img.size(), CV_8UC1, Scalar(GC_BGD));
        maskIso(Rect(50, 40, 60, 40)).setTo(GC_PR_BGD);
        maskIso(Rect(60, 45, 40, 30)).setTo(GC_FGD);
        maskIso.at<uchar>(5, 5) = GC_PR_FGD;
        grabCut(img, maskIso, Rect(), bgdModel, fgdModel, 2, GC_INIT_WITH_MASK);
        EXPECT_EQ((int)GC_FGD, (int)maskIso.at<uchar>(55, 75));
        EXPECT_EQ((int)GC_BGD, (int)maskIso.at<uchar>(0, 0));
    }

    // Case 3: All definite mask (vtxCount == 0) must leave mask unchanged without error
    {
        theRNG().state = 1234567ULL;
        Mat maskFixed(img.size(), CV_8UC1, Scalar(GC_BGD));
        maskFixed(fgRect).setTo(GC_FGD);
        Mat orig = maskFixed.clone();
        Mat bgdModel, fgdModel;
        grabCut(img, maskFixed, Rect(), bgdModel, fgdModel, 2, GC_INIT_WITH_MASK);
        EXPECT_EQ(0, countNonZero(maskFixed != orig));
    }
}

}} // namespace
