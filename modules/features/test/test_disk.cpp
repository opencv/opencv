// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "npy_blob.hpp"

#ifdef HAVE_OPENCV_DNN

#include "opencv2/dnn.hpp"
#include "opencv2/core/utils/configuration.private.hpp"

namespace opencv_test { namespace {

static void testDiskRegression(const Size& imageSize, const std::string& tag)
{
    applyTestTag(CV_TEST_TAG_MEMORY_2GB);

    Mat refKpts = blobFromNPY(cvtest::findDataFile("features/disk/box_in_scene_" + tag + "_kpts.npy"));
    Mat refDesc = blobFromNPY(cvtest::findDataFile("features/disk/box_in_scene_" + tag + "_desc.npy"));
    if (refKpts.type() != CV_32F)
        refKpts.convertTo(refKpts, CV_32F);
    ASSERT_EQ(refKpts.cols, 3);
    const int n = refKpts.rows;

    const std::string modelPath = cvtest::findDataFile("dnn/disk.onnx", false);

    Ptr<DISK> detector;
    ASSERT_NO_THROW(detector = DISK::create(modelPath, n, 0.0f, imageSize));
    ASSERT_TRUE(detector);
    EXPECT_FALSE(detector->empty());
    EXPECT_EQ(detector->descriptorSize(), 128);
    EXPECT_EQ(detector->descriptorType(), CV_32F);
    EXPECT_EQ(detector->defaultNorm(),    NORM_L2);

    Mat img = imread(cvtest::findDataFile("shared/box_in_scene.png"));
    ASSERT_FALSE(img.empty());

    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    detector->detectAndCompute(img, noArray(), keypoints, descriptors);

    ASSERT_EQ(static_cast<int>(keypoints.size()), n) << "keypoint count mismatch (" << tag << ")";
    ASSERT_EQ(descriptors.rows, n);
    ASSERT_EQ(descriptors.cols, refDesc.cols);
    ASSERT_EQ(descriptors.type(), CV_32F);

    Mat pos(n, 2, CV_32F), resp(n, 1, CV_32F);
    for (int i = 0; i < n; ++i)
    {
        pos.at<float>(i, 0) = keypoints[i].pt.x;
        pos.at<float>(i, 1) = keypoints[i].pt.y;
        resp.at<float>(i, 0) = keypoints[i].response;
    }

    EXPECT_LE(cvtest::norm(pos, refKpts.colRange(0, 2), NORM_INF), 1e-3)
        << "keypoint positions differ (" << tag << ")";
    EXPECT_LE(cvtest::norm(resp, refKpts.col(2), NORM_INF), 0.1)
        << "keypoint responses differ (" << tag << ")";
    EXPECT_LE(cvtest::norm(descriptors, refDesc, NORM_INF), 1e-2)
        << "descriptors differ (" << tag << ")";
}

TEST(Features2d_DISK, regression_default)
{
    testDiskRegression(Size(), "default");
}

TEST(Features2d_DISK, regression_512x384)
{
    testDiskRegression(Size(512, 384), "512x384");
}

TEST(Features2d_DISK, MaxKeypointsAndThreshold)
{
    applyTestTag(CV_TEST_TAG_MEMORY_2GB);

    const std::string modelPath = cvtest::findDataFile("dnn/disk.onnx", false);

    Ptr<DISK> detector = DISK::create(modelPath);
    ASSERT_TRUE(detector);

    Mat img = imread(cvtest::findDataFile("shared/lena.png"));
    ASSERT_FALSE(img.empty());

    std::vector<KeyPoint> baseKpts;
    Mat baseDesc;
    detector->detectAndCompute(img, noArray(), baseKpts, baseDesc);
    ASSERT_GT(baseKpts.size(), 50u);

    const int kCap = 50;
    detector->setMaxKeypoints(kCap);
    EXPECT_EQ(detector->getMaxKeypoints(), kCap);

    std::vector<KeyPoint> capKpts;
    Mat capDesc;
    detector->detectAndCompute(img, noArray(), capKpts, capDesc);
    EXPECT_EQ(capKpts.size(), static_cast<size_t>(kCap));
    EXPECT_EQ(capDesc.rows, kCap);

    float minKept = std::numeric_limits<float>::max();
    for (const KeyPoint& kp : capKpts)
        minKept = std::min(minKept, kp.response);

    detector->setMaxKeypoints(-1);
    detector->setScoreThreshold(minKept);
    std::vector<KeyPoint> thrKpts;
    detector->detectAndCompute(img, noArray(), thrKpts, noArray());
    for (const KeyPoint& kp : thrKpts)
        EXPECT_GT(kp.response, minKept);
}

TEST(Features2d_DISK, MaskSupport)
{
    applyTestTag(CV_TEST_TAG_MEMORY_2GB);

    const std::string modelPath = cvtest::findDataFile("dnn/disk.onnx", false);

    Ptr<DISK> detector = DISK::create(modelPath);
    Mat img = imread(cvtest::findDataFile("shared/lena.png"));
    ASSERT_FALSE(img.empty());

    Mat mask = Mat::zeros(img.size(), CV_8UC1);
    const Rect roi(img.cols / 4, img.rows / 4, img.cols / 2, img.rows / 2);
    mask(roi).setTo(255);

    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    detector->detectAndCompute(img, mask, keypoints, descriptors);

    ASSERT_FALSE(keypoints.empty());
    ASSERT_EQ(descriptors.rows, static_cast<int>(keypoints.size()));

    for (const KeyPoint& kp : keypoints)
    {
        EXPECT_TRUE(roi.contains(Point(cvFloor(kp.pt.x), cvFloor(kp.pt.y))))
            << "Keypoint " << kp.pt << " escaped the mask ROI " << roi;
    }
}

TEST(Features2d_DISK, InvalidImageSize)
{
    const std::string modelPath = cvtest::findDataFile("dnn/disk.onnx", false);

    EXPECT_THROW(DISK::create(modelPath, -1, 0.0f, Size(1000, 1024)), cv::Exception);
    EXPECT_THROW(DISK::create(modelPath, -1, 0.0f, Size(1024, 1000)), cv::Exception);
    EXPECT_THROW(DISK::create(modelPath, -1, 0.0f, Size(-16, 1024)),  cv::Exception);

    Ptr<DISK> detector;
    ASSERT_NO_THROW(detector = DISK::create(modelPath, -1, 0.0f, Size()));
    ASSERT_TRUE(detector);

    EXPECT_THROW(detector->setImageSize(Size(15, 1024)), cv::Exception);
    EXPECT_NO_THROW(detector->setImageSize(Size(512, 512)));
    EXPECT_EQ(detector->getImageSize(), Size(512, 512));
}

}} // namespace

#endif // HAVE_OPENCV_DNN
