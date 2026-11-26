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

    void generateTestData()
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

OCL_TEST(Photo_FastNlMeansDenoising, SubMatrixIsolation)
{
    const Size roi_size(29, 35);
    const int pad_left = 11;
    const int pad_top = 14;
    const int pad_right = 19;
    const int pad_bottom = 17;
    const int template_window_size = 5;
    const int search_window_size = 11;
    const Rect roi_rect(pad_left, pad_top, roi_size.width, roi_size.height);
    const Size parent_size(roi_size.width + pad_left + pad_right,
                           roi_size.height + pad_top + pad_bottom);

    struct OclCase
    {
        int type;
        int norm_type;
        float h_val;
    };
    const OclCase cases[] = {
        { CV_8UC1, NORM_L2, 50.0f },
        { CV_8UC2, NORM_L2, 50.0f },
        { CV_8UC3, NORM_L2, 50.0f },
        { CV_8UC4, NORM_L2, 50.0f },
        { CV_16UC1, NORM_L1, 4000.0f }
    };

    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c)
    {
        const int type = cases[c].type;
        const bool is_16u = (CV_MAT_DEPTH(type) == CV_16U);
        const double min_val = is_16u ? 10000.0 : 40.0;
        const double max_val = is_16u ? 50000.0 : 200.0;
        const double high_border_val = is_16u ? 65000.0 : 255.0;

        Mat isolated(roi_size, type);
        theRNG().fill(isolated, RNG::UNIFORM, Scalar::all(min_val), Scalar::all(max_val));

        Mat parent_low(parent_size, type, Scalar::all(0));
        Mat parent_high(parent_size, type, Scalar::all(high_border_val));
        isolated.copyTo(parent_low(roi_rect));
        isolated.copyTo(parent_high(roi_rect));

        UMat u_isolated, u_parent_low, u_parent_high;
        isolated.copyTo(u_isolated);
        parent_low.copyTo(u_parent_low);
        parent_high.copyTo(u_parent_high);

        const std::vector<float> h(1, cases[c].h_val);
        UMat u_dst_isolated, u_dst_low, u_dst_high;
        OCL_ON(cv::fastNlMeansDenoising(u_isolated, u_dst_isolated, h, template_window_size, search_window_size, cases[c].norm_type));
        OCL_ON(cv::fastNlMeansDenoising(u_parent_low(roi_rect), u_dst_low, h, template_window_size, search_window_size, cases[c].norm_type));
        OCL_ON(cv::fastNlMeansDenoising(u_parent_high(roi_rect), u_dst_high, h, template_window_size, search_window_size, cases[c].norm_type));

        Mat dst_isolated = u_dst_isolated.getMat(ACCESS_READ);
        Mat dst_low = u_dst_low.getMat(ACCESS_READ);
        Mat dst_high = u_dst_high.getMat(ACCESS_READ);

        ASSERT_EQ(isolated.size(), dst_isolated.size());
        ASSERT_EQ(isolated.type(), dst_isolated.type());
        ASSERT_GT(cvtest::norm(dst_isolated, isolated, NORM_L1), 0.0);
        EXPECT_EQ(0.0, cvtest::norm(dst_low, dst_isolated, NORM_INF));
        EXPECT_EQ(0.0, cvtest::norm(dst_high, dst_isolated, NORM_INF));
    }
}

} } // namespace opencv_test::ocl

#endif // HAVE_OPENCL
