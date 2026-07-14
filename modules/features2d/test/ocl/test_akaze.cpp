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

}} // namespace

#endif
