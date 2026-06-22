// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "npy_blob.hpp"

#ifdef HAVE_OPENCV_DNN

#include "opencv2/dnn.hpp"
#include "opencv2/core/utils/configuration.private.hpp"

namespace opencv_test { namespace {

static void skipIfClassicDnnEngine()
{
    const auto engine = static_cast<cv::dnn::EngineType>(
        cv::utils::getConfigurationParameterSizeT("OPENCV_FORCE_DNN_ENGINE", cv::dnn::ENGINE_AUTO));
    if (engine == cv::dnn::ENGINE_CLASSIC)
        throw SkipTestException("ALIKED/LightGlue reference outputs are generated with the new DNN engine");
}

TEST(Features2d_ALIKED, Regression)
{
    skipIfClassicDnnEngine();
    const std::string modelPath = cvtest::findDataFile("dnn/onnx/models/aliked-n16rot-top1k-640.onnx", false);

    Ptr<ALIKED> aliked = ALIKED::create(modelPath);
    ASSERT_FALSE(aliked.empty());
    ASSERT_FALSE(aliked->empty());
    EXPECT_EQ(aliked->descriptorSize(), 128);
    EXPECT_EQ(aliked->descriptorType(), CV_32F);
    EXPECT_EQ(aliked->defaultNorm(), NORM_L2);

    const std::string imgPath = cvtest::findDataFile("shared/box.png");
    Mat img = imread(imgPath);
    ASSERT_FALSE(img.empty()) << "Could not load test image: " << imgPath;

    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    aliked->detectAndCompute(img, noArray(), keypoints, descriptors);

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

    // Load ORT reference outputs (generated with same OpenCV preprocessing)
    Mat refKpts   = blobFromNPY(cvtest::findDataFile("dnn/aliked_keypoints_box.npy"));
    Mat refDescs  = blobFromNPY(cvtest::findDataFile("dnn/aliked_descriptors_box.npy"));

    // Keypoint count must match exactly
    ASSERT_EQ(static_cast<int>(keypoints.size()), refKpts.rows)
        << "Keypoint count mismatch: got " << keypoints.size()
        << ", expected " << refKpts.rows;

    // Compare each keypoint (normalized coords -> pixel coords)
    const float origW = static_cast<float>(img.cols);
    const float origH = static_cast<float>(img.rows);
    for (int i = 0; i < refKpts.rows; i++)
    {
        float refX = (refKpts.at<float>(i, 0) + 1.0f) * 0.5f * origW;
        float refY = (refKpts.at<float>(i, 1) + 1.0f) * 0.5f * origH;
        EXPECT_NEAR(keypoints[i].pt.x, refX, 1e-4) << "Keypoint " << i << " x mismatch";
        EXPECT_NEAR(keypoints[i].pt.y, refY, 1e-4) << "Keypoint " << i << " y mismatch";
    }

    // Compare descriptors row by row
    for (int i = 0; i < refDescs.rows; i++)
    {
        Mat diff = descriptors.row(i) - refDescs.row(i);
        double maxDiff = cv::norm(diff, cv::NORM_INF);
        EXPECT_LT(maxDiff, 1e-5) << "Descriptor " << i << " mismatch (max diff=" << maxDiff << ")";
    }
}

TEST(Features2d_LightGlue_ALIKED, Regression)
{
    skipIfClassicDnnEngine();
    const std::string alikedPath = cvtest::findDataFile("dnn/onnx/models/aliked-n16rot-top1k-640.onnx", false);
    const std::string lgPath = cvtest::findDataFile("dnn/onnx/models/aliked_lightglue.onnx", false);

    Ptr<ALIKED> aliked = ALIKED::create(alikedPath);
    Ptr<LightGlueMatcher> lg = LightGlueMatcher::create(lgPath);
    ASSERT_FALSE(aliked.empty());
    ASSERT_FALSE(lg.empty());

    Mat img1 = imread(cvtest::findDataFile("shared/box.png"));
    Mat img2 = imread(cvtest::findDataFile("shared/box_in_scene.png"));
    ASSERT_FALSE(img1.empty());
    ASSERT_FALSE(img2.empty());

    // Detect features on both images
    std::vector<KeyPoint> kpts1, kpts2;
    Mat descs1, descs2;
    aliked->detectAndCompute(img1, noArray(), kpts1, descs1);
    aliked->detectAndCompute(img2, noArray(), kpts2, descs2);

    ASSERT_GT(static_cast<int>(kpts1.size()), 0);
    ASSERT_GT(static_cast<int>(kpts2.size()), 0);

    // Build keypoint matrices (pixel coordinates)
    Mat kpts1Mat(static_cast<int>(kpts1.size()), 2, CV_32F);
    Mat kpts2Mat(static_cast<int>(kpts2.size()), 2, CV_32F);
    for (size_t i = 0; i < kpts1.size(); i++)
    {
        kpts1Mat.at<float>(static_cast<int>(i), 0) = kpts1[i].pt.x;
        kpts1Mat.at<float>(static_cast<int>(i), 1) = kpts1[i].pt.y;
    }
    for (size_t i = 0; i < kpts2.size(); i++)
    {
        kpts2Mat.at<float>(static_cast<int>(i), 0) = kpts2[i].pt.x;
        kpts2Mat.at<float>(static_cast<int>(i), 1) = kpts2[i].pt.y;
    }

    lg->setPairInfo(kpts1Mat, kpts2Mat, img1.size(), img2.size());

    std::vector<DMatch> matches;
    lg->match(descs1, descs2, matches);

    ASSERT_GT(static_cast<int>(matches.size()), 0);
    for (const auto& m : matches)
    {
        EXPECT_GE(m.queryIdx, 0);
        EXPECT_LT(m.queryIdx, static_cast<int>(kpts1.size()));
        EXPECT_GE(m.trainIdx, 0);
        EXPECT_LT(m.trainIdx, static_cast<int>(kpts2.size()));
    }

    // Load ORT reference outputs
    Mat refMatches = blobFromNPY(cvtest::findDataFile("dnn/lightglue_matches.npy"));

    // Match count must match exactly (same OpenCV preprocessing in both)
    ASSERT_EQ(static_cast<int>(matches.size()), refMatches.rows)
        << "Match count mismatch: got " << matches.size()
        << ", expected " << refMatches.rows;

    // Compare each match (index pairs should be identical)
    for (int i = 0; i < refMatches.rows; i++)
    {
        int refQIdx = static_cast<int>(refMatches.at<int64_t>(i, 0));
        int refTIdx = static_cast<int>(refMatches.at<int64_t>(i, 1));
        EXPECT_EQ(matches[i].queryIdx, refQIdx) << "Match " << i << " queryIdx mismatch";
        EXPECT_EQ(matches[i].trainIdx, refTIdx) << "Match " << i << " trainIdx mismatch";
    }
}

#else  // !HAVE_OPENCV_DNN

TEST(Features2d_ALIKED, not_available)
{
    EXPECT_THROW(ALIKED::create("dummy.onnx"), cv::Exception);
}

TEST(Features2d_LightGlueMatcher_ALIKED, not_available)
{
    EXPECT_THROW(LightGlueMatcher::create("dummy.onnx"), cv::Exception);
}

#endif  // HAVE_OPENCV_DNN

}}  // namespace opencv_test
