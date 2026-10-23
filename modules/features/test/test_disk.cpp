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

// DISK keeps its top-N keypoints in response order, picked with an unstable
// partial_sort, so neither the reference nor the detector output has a
// reproducible row order. Compare both sides in one canonical order instead:
// keypoint position, x then y. Responses make a poor key here, because engines
// can disagree on them by more than the gaps between neighbouring responses,
// while distinct keypoints sit on distinct pixels.
static std::vector<int> positionOrder(const std::vector<Point2f>& points)
{
    std::vector<int> order(points.size());
    for (size_t i = 0; i < order.size(); ++i)
        order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(), [&points](int a, int b)
    {
        if (points[a].x != points[b].x)
            return points[a].x < points[b].x;
        return points[a].y < points[b].y;
    });
    return order;
}

static Mat permuteRows(const Mat& rows, const std::vector<int>& order)
{
    CV_Assert(rows.rows == static_cast<int>(order.size()));

    Mat permuted(rows.size(), rows.type());
    for (size_t i = 0; i < order.size(); ++i)
        rows.row(order[i]).copyTo(permuted.row(static_cast<int>(i)));
    return permuted;
}

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
    ASSERT_EQ(refDesc.rows, n);

    // Put both sides into the canonical order before comparing them row by row
    std::vector<Point2f> refPoints(n), points(n);
    for (int i = 0; i < n; ++i)
    {
        refPoints[i] = Point2f(refKpts.at<float>(i, 0), refKpts.at<float>(i, 1));
        points[i] = keypoints[i].pt;
    }

    const std::vector<int> refOrder = positionOrder(refPoints);
    refKpts = permuteRows(refKpts, refOrder);
    refDesc = permuteRows(refDesc, refOrder);

    const std::vector<int> order = positionOrder(points);
    descriptors = permuteRows(descriptors, order);
    std::vector<KeyPoint> sortedKeypoints(n);
    for (int i = 0; i < n; ++i)
        sortedKeypoints[i] = keypoints[order[i]];
    keypoints.swap(sortedKeypoints);

    for (int i = 0; i < n; ++i)
    {
        float refX = refKpts.at<float>(i, 0);
        float refY = refKpts.at<float>(i, 1);
        float refScore = refKpts.at<float>(i, 2);
        EXPECT_NEAR(keypoints[i].pt.x, refX, 1e-4) << "Keypoint " << i << " x mismatch";
        EXPECT_NEAR(keypoints[i].pt.y, refY, 1e-4) << "Keypoint " << i << " y mismatch";
        EXPECT_NEAR(keypoints[i].response, refScore, 1e-4) << "Keypoint " << i << " score mismatch";;
    }

    // Compare descriptors row by row
    for (int i = 0; i < refDesc.rows; i++)
    {
        Mat diff = descriptors.row(i) - refDesc.row(i);
        double maxDiff = cv::norm(diff, cv::NORM_INF);
        EXPECT_LT(maxDiff, 1e-5) << "Descriptor " << i << " mismatch (max diff=" << maxDiff << ")";
    }
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
