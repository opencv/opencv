// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "npy_blob.hpp"

#ifdef HAVE_OPENCV_DNN
#include "opencv2/dnn.hpp"
#include "opencv2/core/utils/logger.hpp"
#endif

namespace opencv_test { namespace {

#ifdef HAVE_OPENCV_DNN

TEST(Features2d_LightGlue_DISK, Regression)
{
    applyTestTag(CV_TEST_TAG_MEMORY_2GB);

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
        CV_LOG_INFO(NULL, "DISK LightGlue: " << refNotFound << " ref missing, "
                    << actNotFound << " actual extra (within tolerance)");
}

TEST(Features2d_LightGlue_DISK, DynamicShapes_Regression_30502)
{
    applyTestTag(CV_TEST_TAG_MEMORY_2GB);

    const std::string lgPath = cvtest::findDataFile("dnn/onnx/models/disk_lightglue.onnx", false);
    Ptr<LightGlueMatcher> lg = LightGlueMatcher::create(lgPath, 0.0f, 0, 0, LG_DISK);
    ASSERT_FALSE(lg.empty());

    const int descDim = 128;
    RNG rng(42);

    auto generate_descriptors = [&](int count, int dim) {
        Mat desc(count, dim, CV_32F);
        rng.fill(desc, RNG::UNIFORM, -1.0f, 1.0f);
        for (int i = 0; i < count; ++i) {
            Mat row = desc.row(i);
            normalize(row, row, 1.0, 0.0, NORM_L2);
        }
        return desc;
    };

    auto generate_keypoints = [&](int count, int seed_offset) {
        Mat kpts(count, 2, CV_32F);
        for (int i = 0; i < count; ++i) {
            kpts.at<float>(i, 0) = static_cast<float>((i * 5 + seed_offset) % 640);
            kpts.at<float>(i, 1) = static_cast<float>((i * 3 + seed_offset) % 480);
        }
        return kpts;
    };

    // Step 1: Initial small shape (8 -> 8)
    Mat kpts1_8 = generate_keypoints(8, 0);
    Mat kpts2_8 = generate_keypoints(8, 10);
    Mat desc1_8 = generate_descriptors(8, descDim);
    Mat desc2_8 = generate_descriptors(8, descDim);

    lg->setPairInfo(kpts1_8, kpts2_8, Size(640, 480), Size(640, 480));
    std::vector<DMatch> initial_matches_8_8;
    EXPECT_NO_THROW(lg->match(desc1_8, desc2_8, initial_matches_8_8));
    EXPECT_GT(initial_matches_8_8.size(), 0u);
    for (const auto& m : initial_matches_8_8) {
        EXPECT_GE(m.queryIdx, 0);
        EXPECT_LT(m.queryIdx, 8);
        EXPECT_GE(m.trainIdx, 0);
        EXPECT_LT(m.trainIdx, 8);
    }

    // Step 2: Expanded asymmetric shape (50 -> 200)
    Mat kpts1_50 = generate_keypoints(50, 0);
    Mat kpts2_200 = generate_keypoints(200, 10);
    Mat desc1_50 = generate_descriptors(50, descDim);
    Mat desc2_200 = generate_descriptors(200, descDim);
    for (int i = 0; i < 5; ++i) {
        desc1_50.row(i).copyTo(desc2_200.row(i));
    }

    lg->setPairInfo(kpts1_50, kpts2_200, Size(640, 480), Size(640, 480));
    std::vector<DMatch> matches_50_200;
    EXPECT_NO_THROW(lg->match(desc1_50, desc2_200, matches_50_200));
    EXPECT_GT(matches_50_200.size(), 0u);
    for (const auto& m : matches_50_200) {
        EXPECT_GE(m.queryIdx, 0);
        EXPECT_LT(m.queryIdx, 50);
        EXPECT_GE(m.trainIdx, 0);
        EXPECT_LT(m.trainIdx, 200);
    }

    // Step 3: Large-to-small asymmetric shape (200 -> 50)
    Mat kpts1_200 = generate_keypoints(200, 0);
    Mat kpts2_50 = generate_keypoints(50, 10);
    Mat desc1_200 = generate_descriptors(200, descDim);
    Mat desc2_50 = generate_descriptors(50, descDim);
    for (int i = 0; i < 5; ++i) {
        desc2_50.row(i).copyTo(desc1_200.row(i));
    }

    lg->setPairInfo(kpts1_200, kpts2_50, Size(640, 480), Size(640, 480));
    std::vector<DMatch> matches_200_50;
    EXPECT_NO_THROW(lg->match(desc1_200, desc2_50, matches_200_50));
    EXPECT_GT(matches_200_50.size(), 0u);
    for (const auto& m : matches_200_50) {
        EXPECT_GE(m.queryIdx, 0);
        EXPECT_LT(m.queryIdx, 200);
        EXPECT_GE(m.trainIdx, 0);
        EXPECT_LT(m.trainIdx, 50);
    }

    // Step 4: Back to initial small shape (8 -> 8)
    lg->setPairInfo(kpts1_8, kpts2_8, Size(640, 480), Size(640, 480));
    std::vector<DMatch> final_matches_8_8;
    EXPECT_NO_THROW(lg->match(desc1_8, desc2_8, final_matches_8_8));

    ASSERT_EQ(final_matches_8_8.size(), initial_matches_8_8.size());
    for (size_t i = 0; i < initial_matches_8_8.size(); ++i) {
        EXPECT_EQ(final_matches_8_8[i].queryIdx, initial_matches_8_8[i].queryIdx);
        EXPECT_EQ(final_matches_8_8[i].trainIdx, initial_matches_8_8[i].trainIdx);
        EXPECT_NEAR(final_matches_8_8[i].distance, initial_matches_8_8[i].distance, 1e-5f);
    }
}

#else  // !HAVE_OPENCV_DNN

TEST(Features2d_LightGlueMatcher_DISK, not_available)
{
    EXPECT_THROW(LightGlueMatcher::create("dummy.onnx", 0.0f, 0, 0, LG_DISK), cv::Exception);
}

#endif  // HAVE_OPENCV_DNN

}}  // namespace opencv_test
