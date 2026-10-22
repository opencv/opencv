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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#define OUTPUT_SAVING 0
#if OUTPUT_SAVING
#define SAVE(x) std::vector<int> params;\
                params.push_back(16);\
                params.push_back(0);\
                imwrite(folder + "output.png", x ,params);
#else
#define SAVE(x)
#endif

static const double numerical_precision = 0.05; // 95% of pixels should have exact values

typedef testing::TestWithParam<tuple<int, bool> > Photo_SeamlessClone;

TEST_P(Photo_SeamlessClone, input_mask_is_preserved)
{
    const int operation = get<0>(GetParam());
    const bool useROI = get<1>(GetParam());
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "cloning/Normal_Cloning/";
    Mat source = imread(folder + "source1.png", IMREAD_COLOR);
    Mat destination = imread(folder + "destination1.png", IMREAD_COLOR);
    Mat originalMask = imread(folder + "mask.png", IMREAD_GRAYSCALE);
    ASSERT_FALSE(source.empty()) << "Could not load source image " << folder + "source1.png";
    ASSERT_FALSE(destination.empty()) << "Could not load destination image "
                                     << folder + "destination1.png";
    ASSERT_FALSE(originalMask.empty()) << "Could not load mask image " << folder + "mask.png";

    const Mat sourceBefore = source.clone();
    const Mat destinationBefore = destination.clone();
    const Point center(destination.cols / 2, destination.rows / 2);
    const char* operations[] = {"NORMAL_CLONE", "NORMAL_CLONE_WIDE", "colorChange",
                                "illuminationChange", "textureFlattening"};

    SCOPED_TRACE(testing::Message() << operations[operation] << " useROI=" << useROI);
    Mat maskParent(useROI ? Size(originalMask.cols + 8, originalMask.rows + 4)
                         : originalMask.size(), CV_8UC1, Scalar(17));
    Mat mask = useROI ? maskParent(Rect(Point(3, 2), originalMask.size())) : maskParent;
    originalMask.copyTo(mask);
    const Mat maskBefore = maskParent.clone();
    Mat colorMask;
    cvtColor(mask, colorMask, COLOR_GRAY2BGR);
    const Mat colorMaskBefore = colorMask.clone();

    const auto run = [&](const Mat& inputMask, Mat& result)
    {
        switch (operation)
        {
        case 0:
        case 1:
            seamlessClone(source, destination, inputMask, center, result,
                          operation == 0 ? NORMAL_CLONE : NORMAL_CLONE_WIDE);
            break;
        case 2:
            colorChange(source, inputMask, result, 1.5f, 0.5f, 0.5f);
            break;
        case 3:
            illuminationChange(source, inputMask, result, 0.2f, 0.4f);
            break;
        case 4:
            textureFlattening(source, inputMask, result, 30, 120, 3);
            break;
        }
        EXPECT_EQ(0.0, cvtest::norm(sourceBefore, source, NORM_INF)) << "source";
        EXPECT_EQ(0.0, cvtest::norm(destinationBefore, destination, NORM_INF))
            << "destination";
    };

    Mat first, second, control;
    run(mask, first);
    EXPECT_EQ(0.0, cvtest::norm(maskBefore, maskParent, NORM_INF))
        << "mask after first call";
    run(mask, second);
    EXPECT_EQ(0.0, cvtest::norm(maskBefore, maskParent, NORM_INF))
        << "mask after second call";
    run(colorMask, control);
    EXPECT_EQ(0.0, cvtest::norm(colorMaskBefore, colorMask, NORM_INF)) << "color mask";
    EXPECT_EQ(0.0, cvtest::norm(first, control, NORM_INF)) << "equivalent masks";
    EXPECT_EQ(0.0, cvtest::norm(first, second, NORM_INF)) << "repeated call";
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Photo_SeamlessClone,
    testing::Combine(testing::Values(0, 1, 2, 3, 4), testing::Bool()));

TEST(Photo_SeamlessClone_normal, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "cloning/Normal_Cloning/";
    string original_path1 = folder + "source1.png";
    string original_path2 = folder + "destination1.png";
    string original_path3 = folder + "mask.png";
    string reference_path = folder + "reference.png";

    Mat source = imread(original_path1, IMREAD_COLOR);
    Mat destination = imread(original_path2, IMREAD_COLOR);
    Mat mask = imread(original_path3, IMREAD_COLOR);

    ASSERT_FALSE(source.empty()) << "Could not load source image " << original_path1;
    ASSERT_FALSE(destination.empty()) << "Could not load destination image " << original_path2;
    ASSERT_FALSE(mask.empty()) << "Could not load mask image " << original_path3;

    Mat result;
    Point p;
    p.x = destination.size().width/2;
    p.y = destination.size().height/2;
    seamlessClone(source, destination, mask, p, result, NORMAL_CLONE);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    SAVE(result);

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();

    mask = Scalar(0, 0, 0);
    seamlessClone(source, destination, mask, p, result, NORMAL_CLONE);

    reference = destination;
    errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}

TEST(Photo_SeamlessClone_mixed, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "cloning/Mixed_Cloning/";
    string original_path1 = folder + "source1.png";
    string original_path2 = folder + "destination1.png";
    string original_path3 = folder + "mask.png";
    string reference_path = folder + "reference.png";

    Mat source = imread(original_path1, IMREAD_COLOR);
    Mat destination = imread(original_path2, IMREAD_COLOR);
    Mat mask = imread(original_path3, IMREAD_COLOR);

    ASSERT_FALSE(source.empty()) << "Could not load source image " << original_path1;
    ASSERT_FALSE(destination.empty()) << "Could not load destination image " << original_path2;
    ASSERT_FALSE(mask.empty()) << "Could not load mask image " << original_path3;

    Mat result;
    Point p;
    p.x = destination.size().width/2;
    p.y = destination.size().height/2;
    seamlessClone(source, destination, mask, p, result, MIXED_CLONE);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}

TEST(Photo_SeamlessClone_featureExchange, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "cloning/Monochrome_Transfer/";
    string original_path1 = folder + "source1.png";
    string original_path2 = folder + "destination1.png";
    string original_path3 = folder + "mask.png";
    string reference_path = folder + "reference.png";

    Mat source = imread(original_path1, IMREAD_COLOR);
    Mat destination = imread(original_path2, IMREAD_COLOR);
    Mat mask = imread(original_path3, IMREAD_COLOR);

    ASSERT_FALSE(source.empty()) << "Could not load source image " << original_path1;
    ASSERT_FALSE(destination.empty()) << "Could not load destination image " << original_path2;
    ASSERT_FALSE(mask.empty()) << "Could not load mask image " << original_path3;

    Mat result;
    Point p;
    p.x = destination.size().width/2;
    p.y = destination.size().height/2;
    seamlessClone(source, destination, mask, p, result, MONOCHROME_TRANSFER);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}

TEST(Photo_SeamlessClone_colorChange, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "cloning/color_change/";
    string original_path1 = folder + "source1.png";
    string original_path2 = folder + "mask.png";
    string reference_path = folder + "reference.png";

    Mat source = imread(original_path1, IMREAD_COLOR);
    Mat mask = imread(original_path2, IMREAD_COLOR);

    ASSERT_FALSE(source.empty()) << "Could not load source image " << original_path1;
    ASSERT_FALSE(mask.empty()) << "Could not load mask image " << original_path2;

    Mat result;
    colorChange(source, mask, result, 1.5, .5, .5);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}

TEST(Photo_SeamlessClone_illuminationChange, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "cloning/Illumination_Change/";
    string original_path1 = folder + "source1.png";
    string original_path2 = folder + "mask.png";
    string reference_path = folder + "reference.png";

    Mat source = imread(original_path1, IMREAD_COLOR);
    Mat mask = imread(original_path2, IMREAD_COLOR);

    ASSERT_FALSE(source.empty()) << "Could not load source image " << original_path1;
    ASSERT_FALSE(mask.empty()) << "Could not load mask image " << original_path2;

    Mat result;
    illuminationChange(source, mask, result, 0.2f, 0.4f);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}

TEST(Photo_SeamlessClone_textureFlattening, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "cloning/Texture_Flattening/";
    string original_path1 = folder + "source1.png";
    string original_path2 = folder + "mask.png";
    string reference_path = folder + "reference.png";

    Mat source = imread(original_path1, IMREAD_COLOR);
    Mat mask = imread(original_path2, IMREAD_COLOR);

    ASSERT_FALSE(source.empty()) << "Could not load source image " << original_path1;
    ASSERT_FALSE(mask.empty()) << "Could not load mask image " << original_path2;

    Mat result;
    textureFlattening(source, mask, result, 30, 45, 3);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}

}} // namespace
