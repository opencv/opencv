// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "../test_precomp.hpp"
#include "cvconfig.h"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {
TEST(Features2d_AKAZE, ocl_accuracy)
{
    Mat testImg(640, 480, CV_8U);
    theRNG().fill(testImg, RNG::UNIFORM, Scalar(0), Scalar(255), true);

    // CPU version - use MLDB_UPRIGHT to match GPU implementation
    Ptr<Feature2D> akaze_cpu = AKAZE::create(AKAZE::DESCRIPTOR_MLDB_UPRIGHT, 0, 3, 0.001f, 1, 1, KAZE::DIFF_PM_G2);
    vector<KeyPoint> kp_cpu;
    Mat desc_cpu;
    akaze_cpu->detectAndCompute(testImg, noArray(), kp_cpu, desc_cpu);

    // OpenCL version - use MLDB_UPRIGHT to match GPU implementation
    UMat testImg_umat;
    testImg.copyTo(testImg_umat);
    Ptr<Feature2D> akaze_ocl = AKAZE::create(AKAZE::DESCRIPTOR_MLDB_UPRIGHT, 0, 3, 0.001f, 1, 1, KAZE::DIFF_PM_G2);
    vector<KeyPoint> kp_ocl;
    UMat desc_ocl_umat;
    akaze_ocl->detectAndCompute(testImg_umat, noArray(), kp_ocl, desc_ocl_umat);
    Mat desc_ocl = desc_ocl_umat.getMat(ACCESS_READ);

    // Check that both detected keypoints
    ASSERT_FALSE(kp_cpu.empty()) << "CPU should detect keypoints";
    ASSERT_FALSE(kp_ocl.empty()) << "OpenCL should detect keypoints";

    // Allow small keypoint count difference due to border handling
    float kp_ratio = (float)kp_ocl.size() / kp_cpu.size();
    EXPECT_GT(kp_ratio, 0.95f) << "OpenCL keypoint count should be within 5% of CPU, got " << kp_ratio;
    EXPECT_LT(kp_ratio, 1.05f) << "OpenCL keypoint count should be within 5% of CPU, got " << kp_ratio;

    // Check descriptor dimensions match
    ASSERT_EQ(desc_cpu.cols, desc_ocl.cols) << "Descriptor size should match";
    ASSERT_EQ(desc_cpu.type(), desc_ocl.type()) << "Descriptor type should match";

    // Match keypoints by position (within 1 pixel tolerance)
    int matched_kpts = 0;
    int total_compared = 0;
    int matching_bytes = 0;

    for (size_t i = 0; i < kp_cpu.size(); i++) {
        // Find matching keypoint in OpenCL results
        for (size_t j = 0; j < kp_ocl.size(); j++) {
            float dx = fabs(kp_cpu[i].pt.x - kp_ocl[j].pt.x);
            float dy = fabs(kp_cpu[i].pt.y - kp_ocl[j].pt.y);

            if (dx < 1.0f && dy < 1.0f) {
                // Found matching keypoint, compare descriptors
                matched_kpts++;
                for (int k = 0; k < desc_cpu.cols; k++) {
                    if (desc_cpu.at<uchar>((int)i, k) == desc_ocl.at<uchar>((int)j, k)) {
                        matching_bytes++;
                    }
                }
                total_compared += desc_cpu.cols;
                break;
            }
        }
    }

    float match_rate = total_compared > 0 ? (float)matching_bytes / total_compared : 0.0f;
    EXPECT_GT(matched_kpts, (int)(kp_cpu.size() * 0.9f)) << "Should match at least 90% of keypoints by position";
    EXPECT_GT(match_rate, 0.95f) << "Descriptor match rate should be > 95%, got " << match_rate;
}

static void requireDiscreteOpenCL()
{
    if (!cv::ocl::useOpenCL() || cv::ocl::Device::getDefault().hostUnifiedMemory())
        throw SkipTestException("Requires OpenCL device without host-unified memory");
}

struct UseOpenCLGuard
{
    bool prev;
    explicit UseOpenCLGuard(bool enable) : prev(cv::ocl::useOpenCL()) { cv::ocl::setUseOpenCL(enable); }
    ~UseOpenCLGuard() { cv::ocl::setUseOpenCL(prev); }
};

static Mat makeSyntheticFeatureImage(RNG& rng)
{
    Mat img(280, 340, CV_8UC1);
    rng.fill(img, RNG::UNIFORM, 15, 95);
    for (int r = 0; r < 14; ++r)
        for (int c = 0; c < 17; ++c)
            if ((r * 3 + c) % 2 == 0)
                img(Rect(c * 20 + 2, r * 20 + 2, 14, 14)).setTo(Scalar(225));
    return img;
}

TEST(Features2d_AKAZE, ocl_rejects_invalid_class_id)
{
#if defined(_M_IX86) || defined(__i386__)
    throw SkipTestException("AKAZE OpenCL path is disabled on 32-bit x86");
#else
    requireDiscreteOpenCL();
    Mat img = makeSyntheticFeatureImage(theRNG());
    UMat uimg;
    img.copyTo(uimg);

    Ptr<AKAZE> akaze = AKAZE::create(AKAZE::DESCRIPTOR_MLDB_UPRIGHT, 0, 3, 0.001f, 2, 2);
    vector<KeyPoint> detected;
    akaze->detect(img, detected);
    ASSERT_GE(detected.size(), 2u);

    const int invalidLevels[2] = { -1, 2 * 2 }; // 2 octaves * 2 sublevels = 4 levels (0..3)
    for (int badLevel : invalidLevels)
    {
        vector<KeyPoint> kps = { detected[0], detected[1] };
        kps[1].class_id = badLevel;

        UMat udesc(1, 61, CV_8UC1, Scalar::all(0xAB));
        EXPECT_THROW(akaze->compute(uimg, kps, udesc), cv::Exception) << "badLevel=" << badLevel;

        // Pre-filled output buffer must remain untouched when validation fails before allocation
        Mat check = udesc.getMat(ACCESS_READ);
        ASSERT_EQ(1, check.rows);
        ASSERT_EQ(61, check.cols);
        EXPECT_EQ(0xAB, check.at<uchar>(0, 0)) << "badLevel=" << badLevel;
    }
#endif
}

TEST(Features2d_AKAZE, ocl_matches_cpu_at_tie_sizes_and_half_pixels)
{
#if defined(_M_IX86) || defined(__i386__)
    throw SkipTestException("AKAZE OpenCL path is disabled on 32-bit x86");
#else
    requireDiscreteOpenCL();
    Mat img = makeSyntheticFeatureImage(theRNG());
    UMat uimg;
    img.copyTo(uimg);

    Ptr<AKAZE> akaze = AKAZE::create(AKAZE::DESCRIPTOR_MLDB_UPRIGHT);

    const float offsets[3] = { 0.0f, 0.25f, 0.5f };
    const float sizes[7] = { 4.0f, 5.0f, 6.0f, 7.0f, 9.0f, 11.0f, 13.0f };
    for (float off : offsets)
    {
        for (float sz : sizes)
        {
            vector<KeyPoint> kpsCpu, kpsOcl;
            // Include a near-border point (10.5 -> sample -0.5) and interior grid points in octave 0
            kpsCpu.push_back(KeyPoint(10.0f + off, 10.0f + off, sz, 0.0f, 1.0f, 0, 2));
            for (int y = 60; y <= 200; y += 35)
                for (int x = 60; x <= 270; x += 35)
                    kpsCpu.push_back(KeyPoint((float)x + off, (float)y + off, sz, 0.0f, 1.0f, 0, 2));
            kpsOcl = kpsCpu;

            Mat descCpu;
            UMat descOclU;
            {
                UseOpenCLGuard guard(false);
                akaze->compute(img, kpsCpu, descCpu);
            }
            {
                UseOpenCLGuard guard(true);
                akaze->compute(uimg, kpsOcl, descOclU);
            }
            Mat descOcl = descOclU.getMat(ACCESS_READ);
            ASSERT_EQ(descCpu.size(), descOcl.size());
            EXPECT_EQ(0.0, cvtest::norm(descCpu, descOcl, NORM_HAMMING))
                << "off=" << off << " sz=" << sz;
        }
    }

    // Octave 1 (octave_ratio = 2, class_id = 4): size 10 gives scale_f = 2.5 -> even scale 2
    vector<KeyPoint> kpsLo, kpsTie, kpsHi;
    for (int y = 70; y <= 190; y += 40)
        for (int x = 70; x <= 250; x += 45)
        {
            kpsLo.push_back(KeyPoint((float)x, (float)y, 9.6f, 0.0f, 1.0f, 1, 4));
            kpsTie.push_back(KeyPoint((float)x, (float)y, 10.0f, 0.0f, 1.0f, 1, 4));
            kpsHi.push_back(KeyPoint((float)x, (float)y, 10.4f, 0.0f, 1.0f, 1, 4));
        }
    UMat uLo, uTie, uHi;
    {
        UseOpenCLGuard guard(true);
        akaze->compute(uimg, kpsLo, uLo);
        akaze->compute(uimg, kpsTie, uTie);
        akaze->compute(uimg, kpsHi, uHi);
    }
    EXPECT_EQ(0.0, cvtest::norm(uTie.getMat(ACCESS_READ), uLo.getMat(ACCESS_READ), NORM_HAMMING));
    EXPECT_GT(cvtest::norm(uTie.getMat(ACCESS_READ), uHi.getMat(ACCESS_READ), NORM_HAMMING), 50.0);
#endif
}

TEST(Features2d_KAZE, ocl_accuracy)
{
    Mat testImg(640, 480, CV_8U);
    theRNG().fill(testImg, RNG::UNIFORM, Scalar(0), Scalar(255), true);

    // CPU version - use upright=true to match GPU implementation
    Ptr<KAZE> kaze_cpu = KAZE::create(false, true, 0.001f, 4, 4, KAZE::DIFF_PM_G2);
    vector<KeyPoint> kp_cpu;
    Mat desc_cpu;
    kaze_cpu->detectAndCompute(testImg, noArray(), kp_cpu, desc_cpu);

    // OpenCL version - use upright=true to match GPU implementation
    UMat testImg_umat;
    testImg.copyTo(testImg_umat);
    Ptr<KAZE> kaze_ocl = KAZE::create(false, true, 0.001f, 4, 4, KAZE::DIFF_PM_G2);
    vector<KeyPoint> kp_ocl;
    UMat desc_ocl_umat;
    kaze_ocl->detectAndCompute(testImg_umat, noArray(), kp_ocl, desc_ocl_umat);
    Mat desc_ocl = desc_ocl_umat.getMat(ACCESS_READ);

    // Check that both detected keypoints
    ASSERT_FALSE(kp_cpu.empty()) << "CPU should detect keypoints";
    ASSERT_FALSE(kp_ocl.empty()) << "OpenCL should detect keypoints";

    // Allow small keypoint count difference due to border handling
    float kp_ratio = (float)kp_ocl.size() / kp_cpu.size();
    EXPECT_GT(kp_ratio, 0.95f) << "OpenCL keypoint count should be within 5% of CPU, got " << kp_ratio;
    EXPECT_LT(kp_ratio, 1.05f) << "OpenCL keypoint count should be within 5% of CPU, got " << kp_ratio;

    // Check descriptor dimensions match
    ASSERT_EQ(desc_cpu.cols, desc_ocl.cols) << "Descriptor size should match";
    ASSERT_EQ(desc_cpu.type(), desc_ocl.type()) << "Descriptor type should match";

    // Match keypoints by position (within 1 pixel tolerance)
    int matched_kpts = 0;
    int total_compared = 0;
    double total_diff = 0;

    for (size_t i = 0; i < kp_cpu.size(); i++) {
        // Find matching keypoint in OpenCL results
        for (size_t j = 0; j < kp_ocl.size(); j++) {
            float dx = fabs(kp_cpu[i].pt.x - kp_ocl[j].pt.x);
            float dy = fabs(kp_cpu[i].pt.y - kp_ocl[j].pt.y);

            if (dx < 1.0f && dy < 1.0f) {
                // Found matching keypoint, compare descriptors
                matched_kpts++;
                for (int k = 0; k < desc_cpu.cols; k++) {
                    double diff = fabs(desc_cpu.at<float>((int)i, k) - desc_ocl.at<float>((int)j, k));
                    total_diff += diff;
                    total_compared++;
                }
                break;
            }
        }
    }

    float avg_diff = total_compared > 0 ? (float)(total_diff / total_compared) : 0.0f;
    EXPECT_GT(matched_kpts, (int)(kp_cpu.size() * 0.9f)) << "Should match at least 90% of keypoints by position";
    EXPECT_LT(avg_diff, 0.01f) << "Average descriptor difference should be < 0.01, got " << avg_diff;
}

TEST(Features2d_KAZE, ocl_rounds_tie_scale_to_even)
{
    requireDiscreteOpenCL();
    Mat img = makeSyntheticFeatureImage(theRNG());
    UMat uimg;
    img.copyTo(uimg);

    auto computeAtSize = [&](Ptr<KAZE>& kaze, float sz) -> Mat {
        vector<KeyPoint> kps;
        for (int y = 70; y <= 210; y += 35)
            for (int x = 70; x <= 280; x += 35)
                kps.push_back(KeyPoint((float)x, (float)y, sz, 0.0f, 1.0f, 0, 2));
        UMat udesc;
        UseOpenCLGuard guard(true);
        kaze->compute(uimg, kps, udesc);
        return udesc.getMat(ACCESS_READ).clone();
    };

    for (bool extended : { false, true })
    {
        Ptr<KAZE> kaze = KAZE::create(extended, true, 0.001f, 4, 4, KAZE::DIFF_PM_G2);

        // Even-half ties (5/2=2.5 -> 2, 9/2=4.5 -> 4) must match s - 0.2 and differ from s + 0.2
        for (float sz : { 5.0f, 9.0f })
        {
            Mat dLo = computeAtSize(kaze, sz - 0.2f);
            Mat dAt = computeAtSize(kaze, sz);
            Mat dHi = computeAtSize(kaze, sz + 0.2f);
            EXPECT_LE(cvtest::norm(dAt, dLo, NORM_INF), 1e-6)
                << "extended=" << extended << " sz=" << sz;
            EXPECT_GT(cvtest::norm(dAt, dHi, NORM_INF), 1e-2)
                << "extended=" << extended << " sz=" << sz;
        }

        // Odd-half ties (7/2=3.5 -> 4, 11/2=5.5 -> 6) must match s + 0.2 and differ from s - 0.2
        for (float sz : { 7.0f, 11.0f })
        {
            Mat dLo = computeAtSize(kaze, sz - 0.2f);
            Mat dAt = computeAtSize(kaze, sz);
            Mat dHi = computeAtSize(kaze, sz + 0.2f);
            EXPECT_LE(cvtest::norm(dAt, dHi, NORM_INF), 1e-6)
                << "extended=" << extended << " sz=" << sz;
            EXPECT_GT(cvtest::norm(dAt, dLo, NORM_INF), 1e-2)
                << "extended=" << extended << " sz=" << sz;
        }
    }
}

}} // namespace

#endif
