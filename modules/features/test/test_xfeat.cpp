// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "npy_blob.hpp"

#ifdef HAVE_OPENCV_DNN

#include "opencv2/core/utils/configuration.private.hpp"
#include "opencv2/dnn.hpp"

namespace opencv_test { namespace {

static int countNearbyKeypoints(const std::vector<KeyPoint>& keypoints, const Mat& refKpts, float maxDistance)
{
    const float maxDistSq = maxDistance * maxDistance;
    int matched = 0;
    for (const KeyPoint& kp : keypoints)
    {
        float bestDistSq = maxDistSq;
        for (int i = 0; i < refKpts.rows; ++i)
        {
            const float dx = kp.pt.x - refKpts.at<float>(i, 0);
            const float dy = kp.pt.y - refKpts.at<float>(i, 1);
            const float distSq = dx * dx + dy * dy;
            if (distSq < bestDistSq)
                bestDistSq = distSq;
        }
        if (bestDistSq < maxDistSq)
            ++matched;
    }
    return matched;
}

static int countDescriptorMatches(const std::vector<KeyPoint>& keypoints, const Mat& descriptors,
                                  const Mat& refKpts, const Mat& refDesc,
                                  float maxDistance, float maxL2Distance)
{
    const float maxDistSq = maxDistance * maxDistance;
    int matched = 0;
    for (int i = 0; i < descriptors.rows; ++i)
    {
        const KeyPoint& kp = keypoints[i];
        float bestDistSq = maxDistSq;
        int bestIdx = -1;
        for (int j = 0; j < refKpts.rows; ++j)
        {
            const float dx = kp.pt.x - refKpts.at<float>(j, 0);
            const float dy = kp.pt.y - refKpts.at<float>(j, 1);
            const float distSq = dx * dx + dy * dy;
            if (distSq < bestDistSq)
            {
                bestDistSq = distSq;
                bestIdx = j;
            }
        }

        if (bestIdx < 0)
            continue;

        const double l2 = cvtest::norm(descriptors.row(i), refDesc.row(bestIdx), NORM_L2);
        if (l2 <= maxL2Distance)
            ++matched;
    }
    return matched;
}

static void testXFeatRegression(const std::string& imageName, const std::string& tag)
{
    Mat refKpts = blobFromNPY(cvtest::findDataFile("dnn/xfeat_" + tag + "_640_kpts.npy"));
    Mat refDesc = blobFromNPY(cvtest::findDataFile("dnn/xfeat_" + tag + "_640_desc.npy"));
    if (refKpts.type() != CV_32F)
        refKpts.convertTo(refKpts, CV_32F);
    ASSERT_EQ(refKpts.cols, 3);
    const int n = refKpts.rows;
    ASSERT_GT(n, 0);
    ASSERT_EQ(refDesc.rows, n);

    Ptr<XFeat> detector;
    ASSERT_NO_THROW(detector = XFeat::create(cvtest::findDataFile("dnn/onnx/models/xfeat.onnx"), n, 0.5f, Size(640, 640)));
    ASSERT_TRUE(detector);
    EXPECT_FALSE(detector->empty());
    EXPECT_EQ(detector->descriptorSize(), 64);
    EXPECT_EQ(detector->descriptorType(), CV_32F);
    EXPECT_EQ(detector->defaultNorm(), NORM_L2);

    Mat img = imread(cvtest::findDataFile("shared/" + imageName));
    ASSERT_FALSE(img.empty());

    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    detector->detectAndCompute(img, noArray(), keypoints, descriptors);

    ASSERT_EQ(descriptors.rows, static_cast<int>(keypoints.size()));
    ASSERT_EQ(descriptors.cols, refDesc.cols);
    ASSERT_EQ(descriptors.type(), CV_32F);

    const int matched = countNearbyKeypoints(keypoints, refKpts, 1.0f);
    const double matchedRatio = static_cast<double>(matched) / keypoints.size();
    EXPECT_GE(matchedRatio, 0.95)
        << "only " << matched << " of " << keypoints.size()
        << " keypoints matched reference within 1 px (" << tag << ")";

    const int descMatched = countDescriptorMatches(keypoints, descriptors, refKpts, refDesc, 1.0f, 0.25f);
    const double descMatchedRatio = static_cast<double>(descMatched) / descriptors.rows;
    EXPECT_GE(descMatchedRatio, 0.95)
        << "only " << descMatched << " of " << descriptors.rows
        << " descriptors matched reference (L2 <= 0.25 after 1 px keypoint association, " << tag << ")";
}

TEST(Features2d_XFeat, regression_box)
{
    testXFeatRegression("box.png", "box");
}

TEST(Features2d_XFeat, regression_box_in_scene)
{
    testXFeatRegression("box_in_scene.png", "box_in_scene");
}

TEST(Features2d_XFeat, Basic)
{
    Ptr<XFeat> detector = XFeat::create(cvtest::findDataFile("dnn/onnx/models/xfeat.onnx"), 200, 0.5f, Size(640, 640));
    ASSERT_TRUE(detector);
    EXPECT_FALSE(detector->empty());
    EXPECT_EQ(detector->descriptorSize(), 64);
    EXPECT_EQ(detector->descriptorType(), CV_32F);
    EXPECT_EQ(detector->defaultNorm(), NORM_L2);

    Mat img = imread(cvtest::findDataFile("shared/box.png"));
    ASSERT_FALSE(img.empty());

    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    detector->detectAndCompute(img, noArray(), keypoints, descriptors);

    ASSERT_FALSE(keypoints.empty());
    EXPECT_LE(keypoints.size(), 200u);
    ASSERT_EQ(descriptors.rows, static_cast<int>(keypoints.size()));
    EXPECT_EQ(descriptors.cols, 64);
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

TEST(Features2d_XFeat, ParametersAndMask)
{
    Ptr<XFeat> detector = XFeat::create(cvtest::findDataFile("dnn/onnx/models/xfeat.onnx"));
    ASSERT_TRUE(detector);

    detector->setMaxKeypoints(50);
    detector->setScoreThreshold(0.25f);
    detector->setInputSize(Size(640, 640));
    EXPECT_EQ(detector->getMaxKeypoints(), 50);
    EXPECT_EQ(detector->getScoreThreshold(), 0.25f);
    EXPECT_EQ(detector->getInputSize(), Size(640, 640));

    Mat img = imread(cvtest::findDataFile("shared/box_in_scene.png"));
    ASSERT_FALSE(img.empty());

    Mat mask = Mat::zeros(img.size(), CV_8UC1);
    const Rect roi(img.cols / 4, img.rows / 4, img.cols / 2, img.rows / 2);
    mask(roi).setTo(255);

    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    detector->detectAndCompute(img, mask, keypoints, descriptors);

    EXPECT_LE(keypoints.size(), 50u);
    ASSERT_EQ(descriptors.rows, static_cast<int>(keypoints.size()));

    for (const KeyPoint& kp : keypoints){
        EXPECT_TRUE(roi.contains(Point(cvFloor(kp.pt.x), cvFloor(kp.pt.y))));
    }

    Mat boolMask = Mat::zeros(img.size(), CV_BoolC1);
    boolMask(roi).setTo(Scalar(1));
    EXPECT_NO_THROW(detector->detectAndCompute(img, boolMask, keypoints, descriptors));
}

TEST(Features2d_XFeat, InvalidInputSize)
{
    EXPECT_THROW(XFeat::create(cvtest::findDataFile("dnn/onnx/models/xfeat.onnx"), -1, 0.5f, Size(0, 640)), cv::Exception);
    Ptr<XFeat> detector = XFeat::create(cvtest::findDataFile("dnn/onnx/models/xfeat.onnx"));
    ASSERT_TRUE(detector);
    EXPECT_THROW(detector->setInputSize(Size(0, 320)), cv::Exception);
    EXPECT_NO_THROW(detector->setInputSize(Size(320, 320)));
}

}} // namespace

#endif // HAVE_OPENCV_DNN
