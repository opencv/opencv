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

PARAM_TEST_CASE(FastNlMeansDenoisingTestBase, Channels, int, bool, bool)
{
    int cn, normType, templateWindowSize, searchWindowSize;
    std::vector<float> h;
    bool use_roi, use_image;

    TEST_DECLARE_INPUT_PARAMETER(src);
    TEST_DECLARE_OUTPUT_PARAMETER(dst);

    virtual void SetUp()
    {
        cn = GET_PARAM(0);
        normType = GET_PARAM(1);
        use_roi = GET_PARAM(2);
        use_image = GET_PARAM(3);

        templateWindowSize = 7;
        searchWindowSize = 21;

        h.resize(cn);
        for (int i=0; i<cn; i++)
            h[i] = 3.0f + 0.5f*i;
    }

    virtual void generateTestData()
    {
        const int type = CV_8UC(cn);
        Mat image;

        if (use_image) {
            image = readImage("denoising/lena_noised_gaussian_sigma=10.png",
                                  cn == 1 ? IMREAD_GRAYSCALE : IMREAD_COLOR);
            ASSERT_FALSE(image.empty());
        }

        Size roiSize = use_image ? image.size() : randomSize(1, MAX_VALUE);
        Border srcBorder = randomBorder(0, use_roi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSize, srcBorder, type, 0, 255);
        if (use_image) {
            ASSERT_TRUE(cn > 0 && cn <= 4);
            if (cn == 2) {
                int from_to[] = { 0,0, 1,1 };
                src_roi.create(roiSize, type);
                mixChannels(&image, 1, &src_roi, 1, from_to, 2);
            }
            else if (cn == 4) {
                int from_to[] = { 0,0, 1,1, 2,2, 1,3};
                src_roi.create(roiSize, type);
                mixChannels(&image, 1, &src_roi, 1, from_to, 4);
            }
            else image.copyTo(src_roi);
        }

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

        OCL_OFF(cv::fastNlMeansDenoising(src_roi, dst_roi, std::vector<float>(1, h[0]), templateWindowSize, searchWindowSize, normType));
        OCL_ON(cv::fastNlMeansDenoising(usrc_roi, udst_roi, std::vector<float>(1, h[0]), templateWindowSize, searchWindowSize, normType));

        OCL_EXPECT_MATS_NEAR(dst, 1);
    }
}

typedef FastNlMeansDenoisingTestBase FastNlMeansDenoising_hsep;

OCL_TEST_P(FastNlMeansDenoising_hsep, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::fastNlMeansDenoising(src_roi, dst_roi, h, templateWindowSize, searchWindowSize, normType));
        OCL_ON(cv::fastNlMeansDenoising(usrc_roi, udst_roi, h, templateWindowSize, searchWindowSize, normType));

        OCL_EXPECT_MATS_NEAR(dst, 1);
    }
}

typedef FastNlMeansDenoisingTestBase FastNlMeansDenoisingColored;

OCL_TEST_P(FastNlMeansDenoisingColored, Mat)
{
    for (int j = 0; j < test_loop_times; j++)
    {
        generateTestData();

        OCL_OFF(cv::fastNlMeansDenoisingColored(src_roi, dst_roi, h[0], h[0], templateWindowSize, searchWindowSize));
        OCL_ON(cv::fastNlMeansDenoisingColored(usrc_roi, udst_roi, h[0], h[0], templateWindowSize, searchWindowSize));

        OCL_EXPECT_MATS_NEAR(dst, 1);
    }
}

OCL_INSTANTIATE_TEST_CASE_P(Photo, FastNlMeansDenoising,
                            Combine(Values(1, 2, 3, 4), Values((int)NORM_L2, (int)NORM_L1),
                                    Bool(), Values(true)));
OCL_INSTANTIATE_TEST_CASE_P(Photo, FastNlMeansDenoising_hsep,
                            Combine(Values(1, 2, 3, 4), Values((int)NORM_L2, (int)NORM_L1),
                                    Bool(), Values(true)));
OCL_INSTANTIATE_TEST_CASE_P(Photo, FastNlMeansDenoisingColored,
                            Combine(Values(3, 4), Values((int)NORM_L2), Bool(), Values(false)));

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
