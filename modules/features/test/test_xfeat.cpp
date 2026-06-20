// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#ifdef HAVE_OPENCV_DNN

#include "opencv2/core/utils/configuration.private.hpp"
#include "opencv2/dnn.hpp"

namespace opencv_test { namespace {

static void skipIfClassicDnnEngine()
{
    const auto engine = static_cast<cv::dnn::EngineType>(
        cv::utils::getConfigurationParameterSizeT("OPENCV_FORCE_DNN_ENGINE", cv::dnn::ENGINE_AUTO));
    if (engine == cv::dnn::ENGINE_CLASSIC)
        throw SkipTestException("XFeat ONNX model is not supported by the classic DNN engine");
}

static std::string xfeatModelPath()
{
    return cvtest::findDataFile("dnn/xfeat.onnx", false);
}

TEST(Features2d_XFeat, Basic)
{
    skipIfClassicDnnEngine();

    Ptr<XFeat> detector = XFeat::create(xfeatModelPath(), 200, 0.5f, 640);
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
    skipIfClassicDnnEngine();

    Ptr<XFeat> detector = XFeat::create(xfeatModelPath());
    ASSERT_TRUE(detector);

    detector->setMaxKeypoints(50);
    detector->setScoreThreshold(0.25f);
    detector->setInputSize(640);
    EXPECT_EQ(detector->getMaxKeypoints(), 50);
    EXPECT_EQ(detector->getScoreThreshold(), 0.25f);
    EXPECT_EQ(detector->getInputSize(), 640);

    Mat img = imread(cvtest::findDataFile("shared/lena.png"));
    ASSERT_FALSE(img.empty());

    Mat mask = Mat::zeros(img.size(), CV_8UC1);
    const Rect roi(img.cols / 4, img.rows / 4, img.cols / 2, img.rows / 2);
    mask(roi).setTo(255);

    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    detector->detectAndCompute(img, mask, keypoints, descriptors);

    EXPECT_LE(keypoints.size(), 50u);
    ASSERT_EQ(descriptors.rows, static_cast<int>(keypoints.size()));

    for (const KeyPoint& kp : keypoints)
        EXPECT_TRUE(roi.contains(Point(cvFloor(kp.pt.x), cvFloor(kp.pt.y))));
}

TEST(Features2d_XFeat, InvalidInputSize)
{
    skipIfClassicDnnEngine();

    EXPECT_THROW(XFeat::create(xfeatModelPath(), -1, 0.5f, 0), cv::Exception);
    Ptr<XFeat> detector = XFeat::create(xfeatModelPath());
    ASSERT_TRUE(detector);
    EXPECT_THROW(detector->setInputSize(0), cv::Exception);
    EXPECT_NO_THROW(detector->setInputSize(320));
}

}} // namespace

#endif // HAVE_OPENCV_DNN
