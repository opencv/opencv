// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

#ifdef HAVE_OPENCV_DNN

TEST(Features2d_ALIKED, create_from_model)
{
    std::string modelPath = cvtest::findDataFile("dnn/onnx/models/aliked-n16rot-top1k-640.onnx", false);
    Ptr<ALIKED> aliked = ALIKED::create(modelPath);
    ASSERT_FALSE(aliked.empty());
    ASSERT_FALSE(aliked->empty());
}

TEST(Features2d_ALIKED, detectAndCompute)
{
    std::string modelPath = cvtest::findDataFile("dnn/onnx/models/aliked-n16rot-top1k-640.onnx", false);
    Ptr<ALIKED> aliked = ALIKED::create(modelPath);

    Mat image = imread(cvtest::findDataFile("features2d/tsukuba.png"));
    ASSERT_FALSE(image.empty());

    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    aliked->detectAndCompute(image, Mat(), keypoints, descriptors);

    ASSERT_GT((int)keypoints.size(), 0);
    ASSERT_EQ(descriptors.cols, 128);
    ASSERT_EQ(descriptors.type(), CV_32F);
    ASSERT_EQ(descriptors.rows, (int)keypoints.size());
}

TEST(Features2d_ALIKED, detect_only)
{
    std::string modelPath = cvtest::findDataFile("dnn/onnx/models/aliked-n16rot-top1k-640.onnx", false);
    Ptr<ALIKED> aliked = ALIKED::create(modelPath);

    Mat image = imread(cvtest::findDataFile("features2d/tsukuba.png"));
    ASSERT_FALSE(image.empty());

    std::vector<KeyPoint> keypoints;
    aliked->detect(image, keypoints);
    ASSERT_GT((int)keypoints.size(), 0);
}

TEST(Features2d_ALIKED, descriptor_properties)
{
    std::string modelPath = cvtest::findDataFile("dnn/onnx/models/aliked-n16rot-top1k-640.onnx", false);
    Ptr<ALIKED> aliked = ALIKED::create(modelPath);

    ASSERT_EQ(aliked->descriptorSize(), 128);
    ASSERT_EQ(aliked->descriptorType(), CV_32F);
    ASSERT_EQ(aliked->defaultNorm(), NORM_L2);
}

TEST(Features2d_LightGlueMatcher, create_from_model)
{
    std::string modelPath = cvtest::findDataFile("dnn/onnx/models/aliked_lightglue.onnx", false);
    Ptr<LightGlueMatcher> lg = LightGlueMatcher::create(modelPath);
    ASSERT_FALSE(lg.empty());
}

TEST(Features2d_LightGlueMatcher, setPairInfo_and_match)
{
    std::string alikedPath = cvtest::findDataFile("dnn/onnx/models/aliked-n16rot-top1k-640.onnx", false);
    std::string lgPath = cvtest::findDataFile("dnn/onnx/models/aliked_lightglue.onnx", false);

    Ptr<ALIKED> aliked = ALIKED::create(alikedPath);
    Ptr<LightGlueMatcher> lg = LightGlueMatcher::create(lgPath);

    Mat img1 = imread(cvtest::findDataFile("features2d/tsukuba.png"));
    Mat img2 = imread(cvtest::findDataFile("features2d/tsukuba.png"));
    ASSERT_FALSE(img1.empty());

    std::vector<KeyPoint> kpts1, kpts2;
    Mat descs1, descs2;
    aliked->detectAndCompute(img1, Mat(), kpts1, descs1);
    aliked->detectAndCompute(img2, Mat(), kpts2, descs2);

    // Build keypoint matrices (pixel coordinates)
    Mat kpts1Mat((int)kpts1.size(), 2, CV_32F);
    Mat kpts2Mat((int)kpts2.size(), 2, CV_32F);
    for (size_t i = 0; i < kpts1.size(); i++)
    {
        kpts1Mat.at<float>((int)i, 0) = kpts1[i].pt.x;
        kpts1Mat.at<float>((int)i, 1) = kpts1[i].pt.y;
    }
    for (size_t i = 0; i < kpts2.size(); i++)
    {
        kpts2Mat.at<float>((int)i, 0) = kpts2[i].pt.x;
        kpts2Mat.at<float>((int)i, 1) = kpts2[i].pt.y;
    }

    lg->setPairInfo(kpts1Mat, kpts2Mat, img1.size(), img2.size());

    std::vector<DMatch> matches;
    lg->match(descs1, descs2, matches);

    ASSERT_GT((int)matches.size(), 0);
    for (const auto& m : matches)
    {
        ASSERT_GE(m.queryIdx, 0);
        ASSERT_LT(m.queryIdx, (int)kpts1.size());
        ASSERT_GE(m.trainIdx, 0);
        ASSERT_LT(m.trainIdx, (int)kpts2.size());
    }
}

TEST(Features2d_LightGlueMatcher, match_without_context_throws)
{
    std::string lgPath = cvtest::findDataFile("dnn/onnx/models/aliked_lightglue.onnx", false);
    Ptr<LightGlueMatcher> lg = LightGlueMatcher::create(lgPath);

    Mat desc1 = Mat::zeros(10, 128, CV_32F);
    Mat desc2 = Mat::zeros(10, 128, CV_32F);

    std::vector<DMatch> matches;
    EXPECT_THROW(lg->match(desc1, desc2, matches), cv::Exception);
}

TEST(Features2d_LightGlueMatcher, knnMatch_k1)
{
    std::string alikedPath = cvtest::findDataFile("dnn/onnx/models/aliked-n16rot-top1k-640.onnx", false);
    std::string lgPath = cvtest::findDataFile("dnn/onnx/models/aliked_lightglue.onnx", false);

    Ptr<ALIKED> aliked = ALIKED::create(alikedPath);
    Ptr<LightGlueMatcher> lg = LightGlueMatcher::create(lgPath);

    Mat img = imread(cvtest::findDataFile("features2d/tsukuba.png"));
    std::vector<KeyPoint> kpts1, kpts2;
    Mat descs1, descs2;
    aliked->detectAndCompute(img, Mat(), kpts1, descs1);
    aliked->detectAndCompute(img, Mat(), kpts2, descs2);

    Mat kpts1Mat((int)kpts1.size(), 2, CV_32F);
    Mat kpts2Mat((int)kpts2.size(), 2, CV_32F);
    for (size_t i = 0; i < kpts1.size(); i++)
    {
        kpts1Mat.at<float>((int)i, 0) = kpts1[i].pt.x;
        kpts1Mat.at<float>((int)i, 1) = kpts1[i].pt.y;
    }
    for (size_t i = 0; i < kpts2.size(); i++)
    {
        kpts2Mat.at<float>((int)i, 0) = kpts2[i].pt.x;
        kpts2Mat.at<float>((int)i, 1) = kpts2[i].pt.y;
    }

    lg->setPairInfo(kpts1Mat, kpts2Mat, img.size(), img.size());

    std::vector<std::vector<DMatch>> knnMatches;
    lg->knnMatch(descs1, descs2, knnMatches, 1);
    ASSERT_GT((int)knnMatches.size(), 0);
}

TEST(Features2d_LightGlueMatcher, knnMatch_k2_throws)
{
    std::string lgPath = cvtest::findDataFile("dnn/onnx/models/aliked_lightglue.onnx", false);
    Ptr<LightGlueMatcher> lg = LightGlueMatcher::create(lgPath);

    Mat desc1 = Mat::zeros(10, 128, CV_32F);
    Mat desc2 = Mat::zeros(10, 128, CV_32F);
    lg->setPairInfo(Mat::zeros(10, 2, CV_32F), Mat::zeros(10, 2, CV_32F));

    std::vector<std::vector<DMatch>> knnMatches;
    EXPECT_THROW(lg->knnMatch(desc1, desc2, knnMatches, 2), cv::Exception);
}

#else  // !HAVE_OPENCV_DNN

TEST(Features2d_ALIKED, not_available)
{
    EXPECT_THROW(ALIKED::create("dummy.onnx"), cv::Exception);
}

TEST(Features2d_LightGlueMatcher, not_available)
{
    EXPECT_THROW(LightGlueMatcher::create("dummy.onnx"), cv::Exception);
}

#endif  // HAVE_OPENCV_DNN

}}  // namespace opencv_test
