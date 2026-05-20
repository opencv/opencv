// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"

#ifdef HAVE_OPENCV_DNN

namespace opencv_test { namespace {

static std::string findDiskModelOrSkip()
{
    try
    {
        return "/media/abhishek/hugedrive1/opencv_abhishek/test_models/disk.onnx";
        // return cvtest::findDataFile("dnn/disk_standalone.onnx", false);
    }
    catch (const cv::Exception&)
    {
        throw SkipTestException("DISK ONNX model not found in test data");
    }
}

TEST(Features2d_DISK, Regression)
{
    const std::string modelPath = findDiskModelOrSkip();

    Ptr<DISK> detector;
    ASSERT_NO_THROW(detector = DISK::create(modelPath));
    ASSERT_TRUE(detector);
    EXPECT_FALSE(detector->empty());
    EXPECT_EQ(detector->descriptorSize(), 128);
    EXPECT_EQ(detector->descriptorType(), CV_32F);
    EXPECT_EQ(detector->defaultNorm(),    NORM_L2);
    EXPECT_EQ(detector->getMaxKeypoints(), -1);
    EXPECT_FLOAT_EQ(detector->getScoreThreshold(), 0.0f);
    EXPECT_EQ(detector->getImageSize(), Size());

    const std::string imgPath = cvtest::findDataFile("shared/lena.png");
    Mat img = imread(imgPath);
    ASSERT_FALSE(img.empty()) << "Could not load test image: " << imgPath;

    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    detector->detectAndCompute(img, noArray(), keypoints, descriptors);

    EXPECT_GT(keypoints.size(), 100u);
    ASSERT_EQ(descriptors.rows, static_cast<int>(keypoints.size()));
    EXPECT_EQ(descriptors.cols, 128);
    EXPECT_EQ(descriptors.type(), CV_32F);

    for (const KeyPoint& kp : keypoints)
    {
        EXPECT_GE(kp.pt.x, 0.f);
        EXPECT_GE(kp.pt.y, 0.f);
        EXPECT_LT(kp.pt.x, static_cast<float>(img.cols));
        EXPECT_LT(kp.pt.y, static_cast<float>(img.rows));
        EXPECT_GT(kp.response, 0.f);
    }
}

TEST(Features2d_DISK, MaxKeypointsAndThreshold)
{
    const std::string modelPath = findDiskModelOrSkip();

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
    const std::string modelPath = findDiskModelOrSkip();

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
    const std::string modelPath = findDiskModelOrSkip();

    // Sizes that are not positive multiples of 16 must be rejected.
    EXPECT_THROW(DISK::create(modelPath, -1, 0.0f, Size(1000, 1024)), cv::Exception);
    EXPECT_THROW(DISK::create(modelPath, -1, 0.0f, Size(1024, 1000)), cv::Exception);
    EXPECT_THROW(DISK::create(modelPath, -1, 0.0f, Size(-16, 1024)),  cv::Exception);

    // Size() means "use default", so it must succeed.
    Ptr<DISK> detector;
    ASSERT_NO_THROW(detector = DISK::create(modelPath, -1, 0.0f, Size()));
    ASSERT_TRUE(detector);

    EXPECT_THROW(detector->setImageSize(Size(15, 1024)), cv::Exception);
    EXPECT_NO_THROW(detector->setImageSize(Size(512, 512)));
    EXPECT_EQ(detector->getImageSize(), Size(512, 512));
}

}} // namespace

#endif // HAVE_OPENCV_DNN
