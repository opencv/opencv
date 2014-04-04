// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

PARAM_TEST_CASE(UpdateMotionHistory, bool)
{
    double timestamp, duration;
    bool use_roi;

    TEST_DECLARE_INPUT_PARAMETER(silhouette);
    TEST_DECLARE_OUTPUT_PARAMETER(mhi);

    virtual void SetUp()
    {
        use_roi = GET_PARAM(0);
    }

    virtual void generateTestData()
    {
        Size roiSize = randomSize(1, MAX_VALUE);
        Border silhouetteBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(silhouette, silhouette_roi, roiSize, silhouetteBorder, CV_8UC1, -11, 11);

        Border mhiBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(mhi, mhi_roi, roiSize, mhiBorder, CV_32FC1, 0, 1);

        timestamp = randomDouble(0, 1);
        duration = randomDouble(0, 1);
        if (timestamp < duration)
            std::swap(timestamp, duration);

        UMAT_UPLOAD_INPUT_PARAMETER(silhouette);
        UMAT_UPLOAD_OUTPUT_PARAMETER(mhi);
    }
};

OCL_TEST_P(UpdateMotionHistory, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::updateMotionHistory(silhouette_roi, mhi_roi, timestamp, duration));
        OCL_ON(cv::updateMotionHistory(usilhouette_roi, umhi_roi, timestamp, duration));

        OCL_EXPECT_MATS_NEAR(mhi, 0);
    }
}

//////////////////////////////////////// Instantiation /////////////////////////////////////////

OCL_INSTANTIATE_TEST_CASE_P(Video, UpdateMotionHistory, Values(false, true));

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
