/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

//#define DUMP_RESULTS

#ifdef DUMP_RESULTS
#  define DUMP(image, path) imwrite(path, image)
#else
#  define DUMP(image, path)
#endif


TEST(Photo_DenoisingGrayscale, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "denoising/";
    string original_path = folder + "lena_noised_gaussian_sigma=10.png";
    string expected_path = folder + "lena_noised_denoised_grayscale_tw=7_sw=21_h=10.png";

    Mat original = imread(original_path, IMREAD_GRAYSCALE);
    Mat expected = imread(expected_path, IMREAD_GRAYSCALE);

    ASSERT_FALSE(original.empty()) << "Could not load input image " << original_path;
    ASSERT_FALSE(expected.empty()) << "Could not load reference image " << expected_path;

    Mat result;
    fastNlMeansDenoising(original, result, 10);

    DUMP(result, expected_path + ".res.png");

    ASSERT_EQ(0, cvtest::norm(result, expected, NORM_L2));
}

TEST(Photo_DenoisingColored, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "denoising/";
    string original_path = folder + "lena_noised_gaussian_sigma=10.png";
    string expected_path = folder + "lena_noised_denoised_lab12_tw=7_sw=21_h=10_h2=10.png";

    Mat original = imread(original_path, IMREAD_COLOR);
    Mat expected = imread(expected_path, IMREAD_COLOR);

    ASSERT_FALSE(original.empty()) << "Could not load input image " << original_path;
    ASSERT_FALSE(expected.empty()) << "Could not load reference image " << expected_path;

    Mat result;
    fastNlMeansDenoisingColored(original, result, 10, 10);

    DUMP(result, expected_path + ".res.png");

    ASSERT_EQ(0, cvtest::norm(result, expected, NORM_L2));
}

TEST(Photo_DenoisingGrayscaleMulti, regression)
{
    const int imgs_count = 3;
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "denoising/";

    string expected_path = folder + "lena_noised_denoised_multi_tw=7_sw=21_h=15.png";
    Mat expected = imread(expected_path, IMREAD_GRAYSCALE);
    ASSERT_FALSE(expected.empty()) << "Could not load reference image " << expected_path;

    vector<Mat> original(imgs_count);
    for (int i = 0; i < imgs_count; i++)
    {
        string original_path = format("%slena_noised_gaussian_sigma=20_multi_%d.png", folder.c_str(), i);
        original[i] = imread(original_path, IMREAD_GRAYSCALE);
        ASSERT_FALSE(original[i].empty()) << "Could not load input image " << original_path;
    }

    Mat result;
    fastNlMeansDenoisingMulti(original, result, imgs_count / 2, imgs_count, 15);

    DUMP(result, expected_path + ".res.png");

    ASSERT_EQ(0, cvtest::norm(result, expected, NORM_L2));
}

TEST(Photo_DenoisingColoredMulti, regression)
{
    const int imgs_count = 3;
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "denoising/";

    string expected_path = folder + "lena_noised_denoised_multi_lab12_tw=7_sw=21_h=10_h2=15.png";
    Mat expected = imread(expected_path, IMREAD_COLOR);
    ASSERT_FALSE(expected.empty()) << "Could not load reference image " << expected_path;

    vector<Mat> original(imgs_count);
    for (int i = 0; i < imgs_count; i++)
    {
        string original_path = format("%slena_noised_gaussian_sigma=20_multi_%d.png", folder.c_str(), i);
        original[i] = imread(original_path, IMREAD_COLOR);
        ASSERT_FALSE(original[i].empty()) << "Could not load input image " << original_path;
    }

    Mat result;
    fastNlMeansDenoisingColoredMulti(original, result, imgs_count / 2, imgs_count, 10, 15);

    DUMP(result, expected_path + ".res.png");

    ASSERT_EQ(0, cvtest::norm(result, expected, NORM_L2));
}

TEST(Photo_White, issue_2646)
{
    cv::Mat img(50, 50, CV_8UC1, cv::Scalar::all(255));
    cv::Mat filtered;
    cv::fastNlMeansDenoising(img, filtered);

    int nonWhitePixelsCount = (int)img.total() - cv::countNonZero(filtered == img);

    ASSERT_EQ(0, nonWhitePixelsCount);
}

TEST(Photo_Denoising, speed)
{
    string imgname = string(cvtest::TS::ptr()->get_data_path()) + "shared/5MP.png";
    Mat src = imread(imgname, IMREAD_GRAYSCALE), dst;

    double t = (double)getTickCount();
    fastNlMeansDenoising(src, dst, 5, 7, 21);
    t = (double)getTickCount() - t;
    printf("execution time: %gms\n", t*1000./getTickFrequency());
}

// Related issue : https://github.com/opencv/opencv/issues/26582
TEST(Photo_DenoisingGrayscaleMulti16bitL1, regression)
{
    const int imgs_count = 3;
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "denoising/";

    vector<Mat> original_8u(imgs_count);
    vector<Mat> original_16u(imgs_count);
    for (int i = 0; i < imgs_count; i++)
    {
        string original_path = format("%slena_noised_gaussian_sigma=20_multi_%d.png", folder.c_str(), i);
        original_8u[i] = imread(original_path, IMREAD_GRAYSCALE);
        ASSERT_FALSE(original_8u[i].empty()) << "Could not load input image " << original_path;
        original_8u[i].convertTo(original_16u[i], CV_16U);
    }

    Mat result_8u, result_16u;
    std::vector<float> h = {15};
    fastNlMeansDenoisingMulti(original_8u, result_8u, /*imgToDenoiseIndex*/ imgs_count / 2, /*temporalWindowSize*/ imgs_count, h, 7, 21, NORM_L1);
    fastNlMeansDenoisingMulti(original_16u, result_16u, /*imgToDenoiseIndex*/ imgs_count / 2, /*temporalWindowSize*/ imgs_count, h, 7, 21, NORM_L1);
    DUMP(result_8u, "8u.res.png");
    DUMP(result_16u, "16u.res.png");

    cv::Mat expected;
    result_8u.convertTo(expected, CV_16U);

    EXPECT_MAT_NEAR(result_16u, expected, 1);
}

class Photo_DenoisingSubmatrixROI : public testing::TestWithParam<tuple<int, bool> >
{
protected:
    static const int template_window_size = 5;
    static const int search_window_size = 11;

    void SetUp() CV_OVERRIDE
    {
        type = get<0>(GetParam());
        use_umat = get<1>(GetParam());
        const bool is_16u = (CV_MAT_DEPTH(type) == CV_16U);
        norm_type = is_16u ? NORM_L1 : NORM_L2;
        h_val = is_16u ? 4000.0f : 50.0f;
        min_val = is_16u ? 10000.0 : 40.0;
        max_val = is_16u ? 50000.0 : 200.0;
        high_border_val = is_16u ? 65000.0 : 255.0;
        roi_size = Size(29, 35);
        roi_rect = Rect(11, 14, roi_size.width, roi_size.height);
        parent_size = Size(roi_size.width + 11 + 19, roi_size.height + 14 + 17);
    }

    void makeFrame(Mat& isolated, Mat& parent_low, Mat& parent_high) const
    {
        isolated.create(roi_size, type);
        theRNG().fill(isolated, RNG::UNIFORM, Scalar::all(min_val), Scalar::all(max_val));

        parent_low.create(parent_size, type);
        parent_low.setTo(Scalar::all(0));
        parent_high.create(parent_size, type);
        parent_high.setTo(Scalar::all(high_border_val));
        isolated.copyTo(parent_low(roi_rect));
        isolated.copyTo(parent_high(roi_rect));
    }

    void verifyIsolation(const Mat& isolated, const Mat& dst_isolated,
                         const Mat& dst_low, const Mat& dst_high) const
    {
        ASSERT_EQ(isolated.size(), dst_isolated.size());
        ASSERT_EQ(isolated.type(), dst_isolated.type());
        ASSERT_GT(cvtest::norm(dst_isolated, isolated, NORM_L1), 0.0);
        EXPECT_EQ(0.0, cvtest::norm(dst_low, dst_isolated, NORM_INF));
        EXPECT_EQ(0.0, cvtest::norm(dst_high, dst_isolated, NORM_INF));
    }

    int type;
    bool use_umat;
    int norm_type;
    float h_val;
    double min_val;
    double max_val;
    double high_border_val;
    Size roi_size;
    Rect roi_rect;
    Size parent_size;
};

TEST_P(Photo_DenoisingSubmatrixROI, single)
{
    Mat isolated, parent_low, parent_high;
    makeFrame(isolated, parent_low, parent_high);

    const std::vector<float> h(1, h_val);
    Mat dst_isolated, dst_low, dst_high;

    if (use_umat)
    {
        UMat u_isolated, u_parent_low, u_parent_high;
        isolated.copyTo(u_isolated);
        parent_low.copyTo(u_parent_low);
        parent_high.copyTo(u_parent_high);

        UMat u_dst_isolated, u_dst_low, u_dst_high;
        fastNlMeansDenoising(u_isolated, u_dst_isolated, h, template_window_size, search_window_size, norm_type);
        fastNlMeansDenoising(u_parent_low(roi_rect), u_dst_low, h, template_window_size, search_window_size, norm_type);
        fastNlMeansDenoising(u_parent_high(roi_rect), u_dst_high, h, template_window_size, search_window_size, norm_type);

        u_dst_isolated.copyTo(dst_isolated);
        u_dst_low.copyTo(dst_low);
        u_dst_high.copyTo(dst_high);
    }
    else
    {
        fastNlMeansDenoising(isolated, dst_isolated, h, template_window_size, search_window_size, norm_type);
        fastNlMeansDenoising(parent_low(roi_rect), dst_low, h, template_window_size, search_window_size, norm_type);
        fastNlMeansDenoising(parent_high(roi_rect), dst_high, h, template_window_size, search_window_size, norm_type);
    }

    verifyIsolation(isolated, dst_isolated, dst_low, dst_high);
}

TEST_P(Photo_DenoisingSubmatrixROI, multi)
{
    const int temporal_window_size = 3;
    const size_t num_frames = static_cast<size_t>(temporal_window_size);
    std::vector<Mat> isolated_frames(num_frames);
    std::vector<Mat> parent_low_frames(num_frames);
    std::vector<Mat> parent_high_frames(num_frames);
    std::vector<Mat> low_rois(num_frames);
    std::vector<Mat> high_rois(num_frames);

    for (size_t i = 0; i < num_frames; ++i)
    {
        makeFrame(isolated_frames[i], parent_low_frames[i], parent_high_frames[i]);
        low_rois[i] = parent_low_frames[i](roi_rect);
        high_rois[i] = parent_high_frames[i](roi_rect);
    }

    const std::vector<float> h(1, h_val);
    Mat dst_isolated, dst_low, dst_high;

    if (use_umat)
    {
        std::vector<UMat> u_isolated(num_frames), u_low_rois(num_frames), u_high_rois(num_frames);
        std::vector<UMat> u_parent_low(num_frames), u_parent_high(num_frames);
        for (size_t i = 0; i < num_frames; ++i)
        {
            isolated_frames[i].copyTo(u_isolated[i]);
            parent_low_frames[i].copyTo(u_parent_low[i]);
            parent_high_frames[i].copyTo(u_parent_high[i]);
            u_low_rois[i] = u_parent_low[i](roi_rect);
            u_high_rois[i] = u_parent_high[i](roi_rect);
        }

        UMat u_dst_isolated, u_dst_low, u_dst_high;
        fastNlMeansDenoisingMulti(u_isolated, u_dst_isolated, 1, temporal_window_size, h,
                                  template_window_size, search_window_size, norm_type);
        fastNlMeansDenoisingMulti(u_low_rois, u_dst_low, 1, temporal_window_size, h,
                                  template_window_size, search_window_size, norm_type);
        fastNlMeansDenoisingMulti(u_high_rois, u_dst_high, 1, temporal_window_size, h,
                                  template_window_size, search_window_size, norm_type);

        u_dst_isolated.copyTo(dst_isolated);
        u_dst_low.copyTo(dst_low);
        u_dst_high.copyTo(dst_high);
    }
    else
    {
        fastNlMeansDenoisingMulti(isolated_frames, dst_isolated, 1, temporal_window_size, h,
                                  template_window_size, search_window_size, norm_type);
        fastNlMeansDenoisingMulti(low_rois, dst_low, 1, temporal_window_size, h,
                                  template_window_size, search_window_size, norm_type);
        fastNlMeansDenoisingMulti(high_rois, dst_high, 1, temporal_window_size, h,
                                  template_window_size, search_window_size, norm_type);
    }

    verifyIsolation(isolated_frames[1], dst_isolated, dst_low, dst_high);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Photo_DenoisingSubmatrixROI, testing::Values(
    make_tuple(CV_8UC1, false),
    make_tuple(CV_8UC3, false),
    make_tuple(CV_16UC1, false),
    make_tuple(CV_8UC1, true),
    make_tuple(CV_8UC2, true),
    make_tuple(CV_8UC3, true),
    make_tuple(CV_8UC4, true),
    make_tuple(CV_16UC1, true)
));

}} // namespace
