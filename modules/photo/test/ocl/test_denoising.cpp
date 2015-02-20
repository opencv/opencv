// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

PARAM_TEST_CASE(FastNlMeansDenoisingTestBase, Channels, bool)
{
    int cn, templateWindowSize, searchWindowSize;
    float h;
    bool use_roi;

    TEST_DECLARE_INPUT_PARAMETER(src);
    TEST_DECLARE_OUTPUT_PARAMETER(dst);

    virtual void SetUp()
    {
        cn = GET_PARAM(0);
        use_roi = GET_PARAM(1);

        templateWindowSize = 7;
        searchWindowSize = 21;
        h = 3.0f;
    }

    virtual void generateTestData()
    {
        Mat image;
        if (cn == 1)
        {
            image = readImage("denoising/lena_noised_gaussian_sigma=10.png", IMREAD_GRAYSCALE);
            ASSERT_FALSE(image.empty());
        }

        const int type = CV_8UC(cn);

        Size roiSize = cn == 1 ? image.size() : randomSize(1, MAX_VALUE);
        Border srcBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, type, 0, 255);
        if (cn == 1)
            image.copyTo(src_roi);

        Border dstBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(dst, dst_roi, roiSize, dstBorder, type, 0, 255);

        UMAT_UPLOAD_INPUT_PARAMETER(src);
        UMAT_UPLOAD_OUTPUT_PARAMETER(dst);
    }
};

typedef FastNlMeansDenoisingTestBase FastNlMeansDenoising;

OCL_TEST_P(FastNlMeansDenoising, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::fastNlMeansDenoising(src_roi, dst_roi, h, templateWindowSize, searchWindowSize));
        OCL_ON(cv::fastNlMeansDenoising(usrc_roi, udst_roi, h, templateWindowSize, searchWindowSize));

        OCL_EXPECT_MATS_NEAR(dst, 1);
    }
}

typedef FastNlMeansDenoisingTestBase FastNlMeansDenoisingColored;

OCL_TEST_P(FastNlMeansDenoisingColored, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::fastNlMeansDenoisingColored(src_roi, dst_roi, h, h, templateWindowSize, searchWindowSize));
        OCL_ON(cv::fastNlMeansDenoisingColored(usrc_roi, udst_roi, h, h, templateWindowSize, searchWindowSize));

        OCL_EXPECT_MATS_NEAR(dst, 1);
    }
}

OCL_INSTANTIATE_TEST_CASE_P(Photo, FastNlMeansDenoising, Combine(Values(1, 2), Bool()));
OCL_INSTANTIATE_TEST_CASE_P(Photo, FastNlMeansDenoisingColored, Combine(Values(3, 4), Bool()));

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
