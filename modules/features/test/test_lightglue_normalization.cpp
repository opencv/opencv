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

#ifdef HAVE_OPENCV_DNN

static Mat normalizeKeypointsForTest(const Mat& keypoints, Size imageSize, int type)
{
    Mat normalized(keypoints.size(), keypoints.type());
    const float sx = type == LG_ALIKED ? 2.0f / imageSize.width : 1.0f / (imageSize.width - 1);
    const float sy = type == LG_ALIKED ? 2.0f / imageSize.height : 1.0f / (imageSize.height - 1);
    const float offset = type == LG_ALIKED ? -1.0f : 0.0f;
    for (int i = 0; i < keypoints.rows; ++i)
    {
        normalized.at<float>(i, 0) = keypoints.at<float>(i, 0) * sx + offset;
        normalized.at<float>(i, 1) = keypoints.at<float>(i, 1) * sy + offset;
    }
    return normalized;
}

// cvg/LightGlue normalize_keypoints(): (kpts - size / 2) / (max(W, H) / 2)
static Mat referenceNormalizeKeypointsForTest(const Mat& keypoints, Size imageSize)
{
    Mat normalized(keypoints.size(), keypoints.type());
    const float half = 0.5f * (float)std::max(imageSize.width, imageSize.height);
    for (int i = 0; i < keypoints.rows; ++i)
    {
        normalized.at<float>(i, 0) = (keypoints.at<float>(i, 0) - 0.5f * imageSize.width) / half;
        normalized.at<float>(i, 1) = (keypoints.at<float>(i, 1) - 0.5f * imageSize.height) / half;
    }
    return normalized;
}

// Detects keypoints and descriptors on the box / box_in_scene pair with the detector matching `type`.
static void detectLightGluePair(int type, Mat images[2], std::vector<KeyPoint> keypoints[2],
                                Mat descriptors[2], Mat pixels[2])
{
    Ptr<Feature2D> detector;
    if (type == LG_ALIKED)
        detector = ALIKED::create(cvtest::findDataFile("dnn/onnx/models/aliked-n16rot-top1k-640.onnx", false));
    else
    {
        Ptr<DISK> disk = DISK::create(cvtest::findDataFile("dnn/disk.onnx", false));
        disk->setMaxKeypoints(1024);
        detector = disk;
    }
    images[0] = imread(cvtest::findDataFile("shared/box.png"));
    images[1] = imread(cvtest::findDataFile("shared/box_in_scene.png"));
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
    }
}

static std::string lightGlueModelPath(int type)
{
    return cvtest::findDataFile(type == LG_ALIKED
        ? "dnn/onnx/models/aliked_lightglue.onnx" : "dnn/onnx/models/disk_lightglue.onnx", false);
}

static void expectSameMatches(const std::vector<DMatch>& expected, const std::vector<DMatch>& actual,
                              double maxDistanceDiff = 1e-6)
{
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_EQ(expected[i].queryIdx, actual[i].queryIdx) << i;
        EXPECT_EQ(expected[i].trainIdx, actual[i].trainIdx) << i;
        EXPECT_NEAR(expected[i].distance, actual[i].distance, maxDistanceDiff) << i;
    }
}

typedef testing::TestWithParam<int> Features2d_LightGlue_PairInfo;

TEST_P(Features2d_LightGlue_PairInfo, PixelAndCachedCoordinates)
{
    applyTestTag(CV_TEST_TAG_MEMORY_2GB);
    const int type = GetParam();
    Mat images[2], descriptors[2], pixels[2], normalized[2];
    std::vector<KeyPoint> keypoints[2];
    ASSERT_NO_FATAL_FAILURE(detectLightGluePair(type, images, keypoints, descriptors, pixels));
    for (int i = 0; i < 2; ++i)
        normalized[i] = normalizeKeypointsForTest(pixels[i], images[i].size(), type);

    const std::string modelPath = lightGlueModelPath(type);
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

// LG_NORM_MAX_SIDE must reproduce the LightGlue reference mapping: pixel keypoints with image sizes
// give the same matches as keypoints normalized by the reference formula and passed without sizes.
TEST_P(Features2d_LightGlue_PairInfo, MaxSideNormalization)
{
    applyTestTag(CV_TEST_TAG_MEMORY_2GB);
    const int type = GetParam();
    Mat images[2], descriptors[2], pixels[2], reference[2], perAxis[2];
    std::vector<KeyPoint> keypoints[2];
    ASSERT_NO_FATAL_FAILURE(detectLightGluePair(type, images, keypoints, descriptors, pixels));
    for (int i = 0; i < 2; ++i)
    {
        // Non-square images, where the reference and per-axis mappings differ.
        ASSERT_NE(images[i].cols, images[i].rows);
        reference[i] = referenceNormalizeKeypointsForTest(pixels[i], images[i].size());
        perAxis[i] = normalizeKeypointsForTest(pixels[i], images[i].size(), type);
    }

    Ptr<LightGlueMatcher> matcher = LightGlueMatcher::create(lightGlueModelPath(type), 0.0f, 0, 0, type);
    EXPECT_EQ(matcher->getKeypointNormalization(), (int)LG_NORM_MODEL);
    matcher->setKeypointNormalization(LG_NORM_MAX_SIDE);
    EXPECT_EQ(matcher->getKeypointNormalization(), (int)LG_NORM_MAX_SIDE);

    // The reference formula is evaluated literally here, so the inputs differ from the matcher's
    // in the last float bits; allow the LightGlue score tolerance used by the reference data.
    const double scoreTolerance = 1e-4;
    std::vector<DMatch> expected, actual;
    matcher->setPairInfo(reference[0], reference[1]);
    matcher->match(descriptors[0], descriptors[1], expected);
    ASSERT_FALSE(expected.empty());

    matcher->setPairInfo(pixels[0], pixels[1], images[0].size(), images[1].size());
    matcher->match(descriptors[0], descriptors[1], actual);
    expectSameMatches(expected, actual, scoreTolerance);

    // clone() keeps the setting.
    Ptr<LightGlueMatcher> cloned = matcher->clone(true).dynamicCast<LightGlueMatcher>();
    ASSERT_TRUE(cloned);
    EXPECT_EQ(cloned->getKeypointNormalization(), (int)LG_NORM_MAX_SIDE);
    cloned->setImagePairInfo(keypoints[0], keypoints[1], images[0].size(), images[1].size());
    cloned->match(descriptors[0], descriptors[1], actual);
    expectSameMatches(expected, actual, scoreTolerance);

    for (int invalid : {-1, 2})
    {
        // Rethrow so EXPECT_THROW sees the exception after its error code is checked.
        EXPECT_THROW({
            try
            {
                matcher->setKeypointNormalization(invalid);
            }
            catch (const cv::Exception& e)
            {
                EXPECT_EQ(e.code, cv::Error::StsBadArg) << "normalization " << invalid;
                throw;
            }
        }, cv::Exception) << "normalization " << invalid;
    }
    EXPECT_EQ(matcher->getKeypointNormalization(), (int)LG_NORM_MAX_SIDE);

    // Switching back restores the per-axis mapping of the model type.
    matcher->setKeypointNormalization(LG_NORM_MODEL);
    matcher->setPairInfo(perAxis[0], perAxis[1]);
    matcher->match(descriptors[0], descriptors[1], expected);
    matcher->setPairInfo(pixels[0], pixels[1], images[0].size(), images[1].size());
    matcher->match(descriptors[0], descriptors[1], actual);
    expectSameMatches(expected, actual);
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
