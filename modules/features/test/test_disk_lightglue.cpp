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
        throw SkipTestException("DISK/LightGlue reference outputs are generated with the new DNN engine");
}

TEST(Features2d_LightGlue_DISK, Regression)
{
    skipIfClassicDnnEngine();
    const std::string diskPath = cvtest::findDataFile("dnn/disk.onnx", false);
    const std::string lgPath = cvtest::findDataFile("dnn/onnx/models/disk_lightglue.onnx", false);

    Ptr<DISK> disk = DISK::create(diskPath);
    Ptr<LightGlueMatcher> lg = LightGlueMatcher::create(lgPath, 0.0f, 0, 0, LG_DISK);
    ASSERT_FALSE(disk.empty());
    ASSERT_FALSE(lg.empty());

    // Limit keypoints to prevent OOM (LightGlue Transformer self-attention is O(N²))
    disk->setMaxKeypoints(1024);

    Mat img1 = imread(cvtest::findDataFile("shared/box.png"));
    Mat img2 = imread(cvtest::findDataFile("shared/box_in_scene.png"));
    ASSERT_FALSE(img1.empty());
    ASSERT_FALSE(img2.empty());

    // Detect features on both images
    std::vector<KeyPoint> kpts1, kpts2;
    Mat descs1, descs2;
    disk->detectAndCompute(img1, noArray(), kpts1, descs1);
    disk->detectAndCompute(img2, noArray(), kpts2, descs2);

    ASSERT_GT(static_cast<int>(kpts1.size()), 0);
    ASSERT_GT(static_cast<int>(kpts2.size()), 0);
    ASSERT_EQ(kpts1.size(), 1024u);  // maxKeypoints applied
    ASSERT_EQ(kpts2.size(), 1024u);

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
    Mat refMatches = blobFromNPY(cvtest::findDataFile("features/disk/disk_lightglue_matches.npy"));

    // Match count tolerance: ±2 to allow for ONNX Runtime version / floating-point differences
    ASSERT_LE(std::abs(static_cast<int>(matches.size()) - refMatches.rows), 2)
        << "Match count mismatch: got " << matches.size()
        << ", expected " << refMatches.rows << " (±2 tolerance)";

    // Compare match sets instead of ordered sequences. DNN engine differences
    // (e.g. new graph engine vs ORT) may produce different match ordering while
    // the actual matched pairs are equivalent. Allow up to 3 mismatches.
    int refNotFound = 0;
    for (int i = 0; i < refMatches.rows; i++)
    {
        int refQIdx = static_cast<int>(refMatches.at<int64_t>(i, 0));
        int refTIdx = static_cast<int>(refMatches.at<int64_t>(i, 1));

        bool found = false;
        for (const auto& m : matches)
        {
            if (m.queryIdx == refQIdx && m.trainIdx == refTIdx)
            {
                found = true;
                break;
            }
        }
        if (!found)
            refNotFound++;
    }

    int actNotFound = 0;
    for (const auto& m : matches)
    {
        bool found = false;
        for (int i = 0; i < refMatches.rows; i++)
        {
            int refQIdx = static_cast<int>(refMatches.at<int64_t>(i, 0));
            int refTIdx = static_cast<int>(refMatches.at<int64_t>(i, 1));
            if (m.queryIdx == refQIdx && m.trainIdx == refTIdx)
            {
                found = true;
                break;
            }
        }
        if (!found)
            actNotFound++;
    }

    EXPECT_LE(refNotFound, 3) << refNotFound << "/" << refMatches.rows
                              << " reference matches not found in actual output";
    EXPECT_LE(actNotFound, 3) << actNotFound << "/" << matches.size()
                              << " actual matches not found in reference";

    if (refNotFound > 0 || actNotFound > 0)
        std::cout << "[INFO] DISK LightGlue: " << refNotFound << " ref missing, "
                  << actNotFound << " actual extra (within tolerance)" << std::endl;
}

#else  // !HAVE_OPENCV_DNN

TEST(Features2d_LightGlueMatcher_DISK, not_available)
{
    EXPECT_THROW(LightGlueMatcher::create("dummy.onnx", 0.0f, 0, 0, LG_DISK), cv::Exception);
}

#endif  // HAVE_OPENCV_DNN

}}  // namespace opencv_test
