// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#ifdef HAVE_OPENCV_DNN
#include "opencv2/dnn.hpp"
#include "opencv2/core/utils/configuration.private.hpp"
#include <fstream>
#include <iterator>
#endif

namespace opencv_test { namespace {

typedef testing::TestWithParam<int> Features2d_LightGlue_Keypoints;

TEST_P(Features2d_LightGlue_Keypoints, Coordinates)
{
    const Mat keypoints = (Mat_<float>(3, 2) << 0, 0, 4, 2, 7, 3);
    const Mat expected = GetParam() == LG_ALIKED
        ? (Mat_<float>(3, 2) << -1, -1, 0, 0, 0.75f, 0.5f)
        : (Mat_<float>(3, 2) << 0, 0, 4.0f / 7, 2.0f / 3, 1, 1);
    Mat normalized;
    normalizeLightGlueKeypoints(keypoints, normalized, Size(8, 4), GetParam());
    EXPECT_LE(cvtest::norm(normalized, expected, NORM_INF), 1e-6);
    EXPECT_EQ(keypoints.at<float>(2, 0), 7.0f);
    EXPECT_EQ(keypoints.at<float>(2, 1), 3.0f);
}

TEST_P(Features2d_LightGlue_Keypoints, InPlaceNonContinuous)
{
    Mat storage(3, 4, CV_32F, Scalar(42));
    Mat keypoints = storage.colRange(1, 3);
    const Mat pixels = (Mat_<float>(3, 2) << 0, 0, 4, 2, 7, 3);
    pixels.copyTo(keypoints);
    ASSERT_FALSE(keypoints.isContinuous());

    Mat expected;
    normalizeLightGlueKeypoints(pixels, expected, Size(8, 4), GetParam());
    normalizeLightGlueKeypoints(keypoints, keypoints, Size(8, 4), GetParam());
    EXPECT_LE(cvtest::norm(keypoints, expected, NORM_INF), 1e-6);
    EXPECT_EQ(countNonZero(storage.col(0) != 42), 0);
    EXPECT_EQ(countNonZero(storage.col(3) != 42), 0);
}

TEST_P(Features2d_LightGlue_Keypoints, InvalidInput)
{
    const Mat keypoints = Mat::zeros(3, 2, CV_32F);
    Mat normalized;
    EXPECT_THROW(normalizeLightGlueKeypoints(keypoints, normalized, Size(), GetParam()), cv::Exception);
    EXPECT_THROW(normalizeLightGlueKeypoints(keypoints, normalized, Size(8, 0), GetParam()), cv::Exception);
    EXPECT_THROW(normalizeLightGlueKeypoints(keypoints, normalized, Size(0, 4), GetParam()), cv::Exception);
    EXPECT_THROW(normalizeLightGlueKeypoints(Mat::zeros(3, 2, CV_64F), normalized,
                                             Size(8, 4), GetParam()), cv::Exception);
    EXPECT_THROW(normalizeLightGlueKeypoints(Mat::zeros(3, 3, CV_32F), normalized,
                                             Size(8, 4), GetParam()), cv::Exception);
    if (GetParam() == LG_DISK)
    {
        EXPECT_THROW(normalizeLightGlueKeypoints(keypoints, normalized, Size(1, 4), LG_DISK), cv::Exception);
        EXPECT_THROW(normalizeLightGlueKeypoints(keypoints, normalized, Size(8, 1), LG_DISK), cv::Exception);
    }
    else
    {
        EXPECT_NO_THROW(normalizeLightGlueKeypoints(keypoints, normalized, Size(1, 1), LG_ALIKED));
    }
}

INSTANTIATE_TEST_CASE_P(Models, Features2d_LightGlue_Keypoints, testing::Values(LG_ALIKED, LG_DISK));

TEST(Features2d_LightGlue_Keypoints, InvalidModelType)
{
    Mat normalized;
    EXPECT_THROW(normalizeLightGlueKeypoints(Mat::zeros(3, 2, CV_32F), normalized,
                                             Size(8, 4), -1), cv::Exception);
    EXPECT_THROW(normalizeLightGlueKeypoints(Mat::zeros(3, 2, CV_32F), normalized,
                                             Size(8, 4), 2), cv::Exception);
}

#ifdef HAVE_OPENCV_DNN

static void expectSameMatches(const std::vector<DMatch>& expected, const std::vector<DMatch>& actual)
{
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_EQ(expected[i].queryIdx, actual[i].queryIdx) << i;
        EXPECT_EQ(expected[i].trainIdx, actual[i].trainIdx) << i;
        EXPECT_NEAR(expected[i].distance, actual[i].distance, 1e-6) << i;
    }
}

typedef testing::TestWithParam<int> Features2d_LightGlue_PairInfo;

TEST_P(Features2d_LightGlue_PairInfo, PixelAndCachedCoordinates)
{
    applyTestTag(CV_TEST_TAG_MEMORY_2GB);
    const int type = GetParam();
    Ptr<Feature2D> detector;
    if (type == LG_ALIKED)
        detector = ALIKED::create(cvtest::findDataFile("dnn/onnx/models/aliked-n16rot-top1k-640.onnx", false));
    else
    {
        Ptr<DISK> disk = DISK::create(cvtest::findDataFile("dnn/disk.onnx", false));
        disk->setMaxKeypoints(1024);
        detector = disk;
    }
    Mat images[] = {
        imread(cvtest::findDataFile("shared/box.png")),
        imread(cvtest::findDataFile("shared/box_in_scene.png"))
    };
    Mat descriptors[2], pixels[2], normalized[2];
    std::vector<KeyPoint> keypoints[2];
    for (int i = 0; i < 2; ++i)
    {
        ASSERT_FALSE(images[i].empty());
        detector->detectAndCompute(images[i], noArray(), keypoints[i], descriptors[i]);
        ASSERT_FALSE(keypoints[i].empty());
        pixels[i].create((int)keypoints[i].size(), 2, CV_32F);
        for (size_t j = 0; j < keypoints[i].size(); ++j)
        {
            pixels[i].at<float>((int)j, 0) = keypoints[i][j].pt.x;
            pixels[i].at<float>((int)j, 1) = keypoints[i][j].pt.y;
        }
        normalizeLightGlueKeypoints(pixels[i], normalized[i], images[i].size(), type);
    }

    const std::string modelPath = cvtest::findDataFile(type == LG_ALIKED
        ? "dnn/onnx/models/aliked_lightglue.onnx" : "dnn/onnx/models/disk_lightglue.onnx", false);
    Ptr<LightGlueMatcher> matcher = LightGlueMatcher::create(modelPath, 0.0f, 0, 0, type);
    matcher->setPairInfo(pixels[0], pixels[1], images[0].size(), images[1].size());
    std::vector<DMatch> expected, actual;
    matcher->match(descriptors[0], descriptors[1], expected);
    ASSERT_FALSE(expected.empty());

    // Exercise the virtual entry point used by generic DescriptorMatcher consumers.
    Ptr<DescriptorMatcher> base = matcher;
    base->setImagePairInfo(keypoints[0], keypoints[1], images[0].size(), images[1].size());
    base->match(descriptors[0], descriptors[1], actual);
    expectSameMatches(expected, actual);

    matcher->setPairInfo(normalized[0], normalized[1]);
    matcher->match(descriptors[0], descriptors[1], actual);
    expectSameMatches(expected, actual);

    // Each image can independently use cached coordinates.
    matcher->setPairInfo(pixels[0], normalized[1], images[0].size(), Size());
    matcher->match(descriptors[0], descriptors[1], actual);
    expectSameMatches(expected, actual);

    // Also check the in-memory factory when supported. The ORT importer only loads files.
    if (cv::utils::getConfigurationParameterSizeT("OPENCV_FORCE_DNN_ENGINE", dnn::ENGINE_AUTO)
        != dnn::ENGINE_ORT)
    {
        base.release();
        matcher.release();
        std::ifstream modelFile(modelPath, std::ios::binary);
        ASSERT_TRUE(modelFile.is_open());
        const std::vector<uchar> modelData((std::istreambuf_iterator<char>(modelFile)),
                                         std::istreambuf_iterator<char>());
        matcher = LightGlueMatcher::create(modelData, 0.0f, 0, 0, type);
        matcher->setPairInfo(normalized[0], normalized[1]);
        matcher->match(descriptors[0], descriptors[1], actual);
        expectSameMatches(expected, actual);
    }
}

INSTANTIATE_TEST_CASE_P(Models, Features2d_LightGlue_PairInfo, testing::Values(LG_ALIKED, LG_DISK));

TEST(Features2d_LightGlue_PairInfo, InvalidModelType)
{
    for (int type : {-1, 2})
    {
        try
        {
            LightGlueMatcher::create("unused.onnx", 0.0f, 0, 0, type);
            FAIL() << "Invalid model type accepted";
        }
        catch (const cv::Exception& e)
        {
            EXPECT_EQ(e.code, Error::StsBadArg);
        }
        try
        {
            LightGlueMatcher::create(std::vector<uchar>(), 0.0f, 0, 0, type);
            FAIL() << "Invalid model type accepted";
        }
        catch (const cv::Exception& e)
        {
            EXPECT_EQ(e.code, Error::StsBadArg);
        }
    }
}

#endif  // HAVE_OPENCV_DNN

}}  // namespace opencv_test
