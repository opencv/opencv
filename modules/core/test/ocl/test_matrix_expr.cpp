// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test {
namespace ocl {

//////////////////////////////// UMat Expressions /////////////////////////////////////////////////

PARAM_TEST_CASE(UMatExpr, MatDepth, Channels)
{
    int type;
    Size size;

    virtual void SetUp()
    {
        type = CV_MAKE_TYPE(GET_PARAM(0), GET_PARAM(1));
    }

    void generateTestData()
    {
        size = randomSize(1, MAX_VALUE);
    }
};

//////////////////////////////// UMat::eye /////////////////////////////////////////////////

OCL_TEST_P(UMatExpr, Eye)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        Mat m = Mat::eye(size, type);
        UMat um = UMat::eye(size, type);

        EXPECT_MAT_NEAR(m, um, 0);
    }
}

//////////////////////////////// UMat::zeros /////////////////////////////////////////////////

OCL_TEST_P(UMatExpr, Zeros)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        Mat m = Mat::zeros(size, type);
        UMat um = UMat::zeros(size, type);

        EXPECT_MAT_NEAR(m, um, 0);
    }
}

//////////////////////////////// UMat::ones /////////////////////////////////////////////////

OCL_TEST_P(UMatExpr, Ones)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        Mat m = Mat::ones(size, type);
        UMat um = UMat::ones(size, type);

        EXPECT_MAT_NEAR(m, um, 0);
    }
}

//////////////////////////////// with usageFlags /////////////////////////////////////////////////

OCL_TEST_P(UMatExpr, WithUsageFlags)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        UMat u0 = UMat::zeros(size, type, cv::USAGE_ALLOCATE_HOST_MEMORY);
        UMat u1 = UMat::ones(size, type, cv::USAGE_ALLOCATE_HOST_MEMORY);
        UMat u8 = UMat::eye(size, type, cv::USAGE_ALLOCATE_HOST_MEMORY);

        EXPECT_EQ(cv::USAGE_ALLOCATE_HOST_MEMORY, u0.usageFlags);
        EXPECT_EQ(cv::USAGE_ALLOCATE_HOST_MEMORY, u1.usageFlags);
        EXPECT_EQ(cv::USAGE_ALLOCATE_HOST_MEMORY, u8.usageFlags);
    }
}

//////////////////////////////// Instantiation /////////////////////////////////////////////////

OCL_INSTANTIATE_TEST_CASE_P(MatrixOperation, UMatExpr, Combine(OCL_ALL_DEPTHS_16F, OCL_ALL_CHANNELS));

} } // namespace opencv_test::ocl

#endif
