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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

void loadImage(string path, Mat &img)
{
    img = imread(path, -1);
    ASSERT_FALSE(img.empty()) << "Could not load input image " << path;
}

void checkEqual(Mat img0, Mat img1, double threshold, const string& name)
{
    // minMaxLoc() skips NaN on some builds and returns it on others, and a NaN
    // maximum passes "max > threshold", so reject non-finite values explicitly.
    ASSERT_TRUE(cv::checkRange(img0)) << "non-finite values in the first image, method=" << name;
    ASSERT_TRUE(cv::checkRange(img1)) << "non-finite values in the second image, method=" << name;
    double max = 1.0;
    minMaxLoc(abs(img0 - img1), NULL, &max);
    ASSERT_LE(max, threshold) << "method=" << name;
}

static vector<float> DEFAULT_VECTOR;
void loadExposureSeq(String path, vector<Mat>& images, vector<float>& times = DEFAULT_VECTOR)
{
    std::ifstream list_file((path + "list.txt").c_str());
    ASSERT_TRUE(list_file.is_open());
    string name;
    float val;
    while(list_file >> name >> val) {
        Mat img = imread(path + name);
        ASSERT_FALSE(img.empty()) << "Could not load input image " << path + name;
        images.push_back(img);
        times.push_back(1 / val);
    }
    list_file.close();
}

void loadResponseCSV(String path, Mat& response)
{
    response = Mat(256, 1, CV_32FC3);
    std::ifstream resp_file(path.c_str());
    for(int i = 0; i < 256; i++) {
        for(int c = 0; c < 3; c++) {
            resp_file >> response.at<Vec3f>(i)[c];
            resp_file.ignore(1);
        }
    }
    resp_file.close();
}

TEST(Photo_Tonemap, regression)
{
    string test_path = string(cvtest::TS::ptr()->get_data_path()) + "hdr/tonemap/";

    Mat img, expected, result;
    loadImage(test_path + "image.hdr", img);
    float gamma = 2.2f;

    Ptr<Tonemap> linear = createTonemap(gamma);
    linear->process(img, result);
    loadImage(test_path + "linear.png", expected);
    result.convertTo(result, CV_8UC3, 255);
    checkEqual(result, expected, 3, "Simple");

    Ptr<TonemapDrago> drago = createTonemapDrago(gamma);
    drago->process(img, result);
    loadImage(test_path + "drago.png", expected);
    result.convertTo(result, CV_8UC3, 255);
    checkEqual(result, expected, 3, "Drago");

    Ptr<TonemapReinhard> reinhard = createTonemapReinhard(gamma);
    reinhard->process(img, result);
    loadImage(test_path + "reinhard.png", expected);
    result.convertTo(result, CV_8UC3, 255);
    checkEqual(result, expected, 3, "Reinhard");

    Ptr<TonemapMantiuk> mantiuk = createTonemapMantiuk(gamma);
    mantiuk->process(img, result);
    loadImage(test_path + "mantiuk.png", expected);
    result.convertTo(result, CV_8UC3, 255);
    checkEqual(result, expected, 3, "Mantiuk");
}

TEST(Photo_Tonemap, reinhard_preserves_intensity)
{
    string test_path = string(cvtest::TS::ptr()->get_data_path()) + "hdr/tonemap/";

    Mat img, first, second;
    loadImage(test_path + "image.hdr", img);
    Mat original = img.clone();

    Ptr<TonemapReinhard> tonemap = createTonemapReinhard(1.0f, 0.0f, 1.0f, 0.0f);
    EXPECT_FLOAT_EQ(0.0f, tonemap->getIntensity());

    tonemap->process(img, first);
    EXPECT_FLOAT_EQ(0.0f, tonemap->getIntensity());

    tonemap->process(img, second);
    EXPECT_FLOAT_EQ(0.0f, tonemap->getIntensity());

    checkEqual(first, second, 0, "Reinhard repeated process");
    checkEqual(img, original, 0, "Reinhard input preservation");
}

TEST(Photo_AlignMTB, regression)
{
    const int TESTS_COUNT = 100;
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "shared/";

    string file_name = folder + "lena.png";
    Mat img;
    loadImage(file_name, img);
    cvtColor(img, img, COLOR_RGB2GRAY);

    int max_bits = 5;
    int max_shift = 32;
    srand(static_cast<unsigned>(time(0)));
    int errors = 0;

    Ptr<AlignMTB> align = createAlignMTB(max_bits);
    RNG rng = theRNG();

    for(int i = 0; i < TESTS_COUNT; i++) {
        Point shift(rng.uniform(0, max_shift), rng.uniform(0, max_shift));
        Mat res;
        align->shiftMat(img, res, shift);
        Point calc = align->calculateShift(img, res);
        errors += (calc != -shift);
    }
    ASSERT_TRUE(errors < 5) << errors << " errors";
}

TEST(Photo_MergeMertens, regression)
{
    string test_path = string(cvtest::TS::ptr()->get_data_path()) + "hdr/";

    vector<Mat> images;
    loadExposureSeq((test_path + "exposures/").c_str() , images);

    Ptr<MergeMertens> merge = createMergeMertens();

    Mat result, expected;
    loadImage(test_path + "merge/mertens.png", expected);
    merge->process(images, result);
    result.convertTo(result, CV_8UC3, 255);
    checkEqual(expected, result, 3, "Mertens");

    Mat uniform(100, 100, CV_8UC3);
    uniform = Scalar(0, 255, 0);

    images.clear();
    images.push_back(uniform);

    merge->process(images, result);
    result.convertTo(result, CV_8UC3, 255);
    checkEqual(uniform, result, 1e-2f, "Mertens");
}

TEST(Photo_MergeDebevec, regression)
{
    string test_path = string(cvtest::TS::ptr()->get_data_path()) + "hdr/";

    vector<Mat> images;
    vector<float> times;
    Mat response;
    loadExposureSeq(test_path + "exposures/", images, times);
    loadResponseCSV(test_path + "exposures/response.csv", response);

    Ptr<MergeDebevec> merge = createMergeDebevec();

    Mat result, expected;
    loadImage(test_path + "merge/debevec.hdr", expected);
    merge->process(images, result, times, response);
    Ptr<Tonemap> map = createTonemap();
    map->process(result, result);
    map->process(expected, expected);
    checkEqual(expected, result, 1e-2f, "Debevec");
}

TEST(Photo_MergeRobertson, regression)
{
    string test_path = string(cvtest::TS::ptr()->get_data_path()) + "hdr/";

    vector<Mat> images;
    vector<float> times;
    loadExposureSeq(test_path + "exposures/", images, times);
    Ptr<MergeRobertson> merge = createMergeRobertson();
    Mat result, expected;
    loadImage(test_path + "merge/robertson.hdr", expected);
    merge->process(images, result, times);

    const float eps = 6.f;
    checkEqual(expected, result, eps, "MergeRobertson");
}

TEST(Photo_MergeDebevec, regression_depth_consistency)
{
    string test_path = string(cvtest::TS::ptr()->get_data_path()) + "hdr/";

    vector<Mat> images8;
    vector<float> times;
    loadExposureSeq(test_path + "exposures/", images8, times);

    vector<Mat> images16(images8.size()), images32(images8.size());
    for (size_t i = 0; i < images8.size(); ++i)
    {
        images8[i].convertTo(images16[i], CV_16UC3, 257.0);
        images8[i].convertTo(images32[i], CV_32FC3, 1.0 / 255.0);
    }

    Ptr<MergeDebevec> merge = createMergeDebevec();
    Ptr<Tonemap> map = createTonemap();

    Mat hdr8, hdr16, hdr32;
    merge->process(images8,  hdr8,  times);
    merge->process(images16, hdr16, times);
    merge->process(images32, hdr32, times);

    map->process(hdr8,  hdr8);
    map->process(hdr16, hdr16);
    map->process(hdr32, hdr32);

    checkEqual(hdr8, hdr16, 2e-2f, "Debevec realdata 16U vs 8U");
    checkEqual(hdr8, hdr32, 2e-2f, "Debevec realdata 32F vs 8U");
}

static void toGraySeq(const vector<Mat>& bgr, vector<Mat>& gray)
{
    gray.resize(bgr.size());
    for (size_t i = 0; i < bgr.size(); ++i)
        cvtColor(bgr[i], gray[i], COLOR_BGR2GRAY);
}

static Mat linearResponseC1(int length)
{
    Mat response(length, 1, CV_32FC1);
    for (int i = 0; i < length; i++)
        response.at<float>(i) = static_cast<float>(i);
    response.at<float>(0) = response.at<float>(1);
    return response;
}

TEST(Photo_MergeDebevec, regression_1channel)
{
    string test_path = string(cvtest::TS::ptr()->get_data_path()) + "hdr/";

    vector<Mat> images_bgr;
    vector<float> times;
    loadExposureSeq(test_path + "exposures/", images_bgr, times);

    vector<Mat> images;
    toGraySeq(images_bgr, images);

    Ptr<MergeDebevec> merge = createMergeDebevec();

    Mat result_default;
    merge->process(images, result_default, times);
    ASSERT_EQ(result_default.type(), CV_32FC1);
    ASSERT_EQ(result_default.size(), images[0].size());
    ASSERT_TRUE(cv::checkRange(result_default)) << "Debevec 1ch default produced non-finite values";

    Mat result_linear;
    merge->process(images, result_linear, times, linearResponseC1(256));
    checkEqual(result_default, result_linear, 1e-5f, "Debevec 1ch default vs linear CRF");

    Mat response3;
    loadResponseCSV(test_path + "exposures/response.csv", response3);
    Mat response1(256, 1, CV_32FC1);
    for (int i = 0; i < 256; i++)
        response1.at<float>(i) = response3.at<Vec3f>(i)[0];
    Mat result_custom;
    merge->process(images, result_custom, times, response1);
    ASSERT_EQ(result_custom.type(), CV_32FC1);
    ASSERT_TRUE(cv::checkRange(result_custom)) << "Debevec 1ch custom CRF produced non-finite values";
}

TEST(Photo_MergeDebevec, regression_1channel_depth_consistency)
{
    string test_path = string(cvtest::TS::ptr()->get_data_path()) + "hdr/";

    vector<Mat> images_bgr;
    vector<float> times;
    loadExposureSeq(test_path + "exposures/", images_bgr, times);

    vector<Mat> images8;
    toGraySeq(images_bgr, images8);

    vector<Mat> images16(images8.size()), images32(images8.size());
    for (size_t i = 0; i < images8.size(); ++i)
    {
        images8[i].convertTo(images16[i], CV_16UC1, 257.0);
        images8[i].convertTo(images32[i], CV_32FC1, 1.0 / 255.0);
    }

    Ptr<MergeDebevec> debevec = createMergeDebevec();
    Ptr<Tonemap> map = createTonemap();

    Mat hdr8, hdr16, hdr32;
    debevec->process(images8, hdr8, times);
    debevec->process(images16, hdr16, times);
    debevec->process(images32, hdr32, times);
    ASSERT_TRUE(cv::checkRange(hdr8) && cv::checkRange(hdr16) && cv::checkRange(hdr32));

    auto to3 = [](const Mat& c1) {
        Mat c3;
        std::vector<Mat> ch{c1, c1, c1};
        cv::merge(ch, c3);
        return c3;
    };
    Mat t8, t16, t32;
    map->process(to3(hdr8), t8);
    map->process(to3(hdr16), t16);
    map->process(to3(hdr32), t32);

    checkEqual(t8, t16, 2e-2f, "Debevec 1ch realdata 16U vs 8U");
    checkEqual(t8, t32, 2e-2f, "Debevec 1ch realdata 32F vs 8U");

    Mat hdr16_linear;
    debevec->process(images16, hdr16_linear, times, linearResponseC1(65536));
    checkEqual(hdr16, hdr16_linear, 1e-5f, "Debevec 1ch 16U default vs linear CRF");
}

TEST(Photo_MergeRobertson, regression_depth_consistency)
{
    string test_path = string(cvtest::TS::ptr()->get_data_path()) + "hdr/";

    vector<Mat> images8;
    vector<float> times;
    loadExposureSeq(test_path + "exposures/", images8, times);

    vector<Mat> images16(images8.size()), images32(images8.size());
    for (size_t i = 0; i < images8.size(); ++i)
    {
        images8[i].convertTo(images16[i], CV_16UC3, 257.0);
        images8[i].convertTo(images32[i], CV_32FC3, 1.0 / 255.0);
    }

    Ptr<MergeRobertson> merge = createMergeRobertson();
    Ptr<Tonemap> map = createTonemap();

    Mat hdr8, hdr16, hdr32;
    merge->process(images8,  hdr8,  times);
    merge->process(images16, hdr16, times);
    merge->process(images32, hdr32, times);

    map->process(hdr8,  hdr8);
    map->process(hdr16, hdr16);
    map->process(hdr32, hdr32);

    checkEqual(hdr8, hdr16, 3e-2f, "Robertson realdata 16U vs 8U");
    checkEqual(hdr8, hdr32, 3e-2f, "Robertson realdata 32F vs 8U");
}

TEST(Photo_CalibrateDebevec, regression)
{
    string test_path = string(cvtest::TS::ptr()->get_data_path()) + "hdr/";

    vector<Mat> images;
    vector<float> times;
    Mat response, expected;
    loadExposureSeq(test_path + "exposures/", images, times);
    loadResponseCSV(test_path + "calibrate/debevec.csv", expected);
    Ptr<CalibrateDebevec> calibrate = createCalibrateDebevec();
    calibrate->process(images, response, times);
    Mat diff = abs(response - expected);
    diff = diff.mul(1.0f / response);
    double max;
    minMaxLoc(diff, NULL, &max);
    ASSERT_LT(max, 0.15);
}

TEST(Photo_CalibrateDebevec, middle_value_is_fixed)
{
    // the response is normalized by g(LDR_SIZE / 2) = 0, i.e. response(128) == 1
    string test_path = string(cvtest::TS::ptr()->get_data_path()) + "hdr/";

    vector<Mat> images;
    vector<float> times;
    Mat response;
    loadExposureSeq(test_path + "exposures/", images, times);
    Ptr<CalibrateDebevec> calibrate = createCalibrateDebevec();
    calibrate->process(images, response, times);
    Vec3f middle = response.at<Vec3f>(128);
    for(int c = 0; c < 3; c++) {
        EXPECT_NEAR(1.0f, middle[c], 1e-3f) << "channel=" << c;
    }
}

TEST(Photo_CalibrateRobertson, regression)
{
    string test_path = string(cvtest::TS::ptr()->get_data_path()) + "hdr/";

    vector<Mat> images;
    vector<float> times;
    Mat response, expected;
    loadExposureSeq(test_path + "exposures/", images, times);
    loadResponseCSV(test_path + "calibrate/robertson.csv", expected);

    Ptr<CalibrateRobertson> calibrate = createCalibrateRobertson();
    calibrate->process(images, response, times);
    checkEqual(expected, response, 1e-1f, "CalibrateRobertson");
}

TEST(Photo_CalibrateRobertson, regression_1channel)
{
    string test_path = string(cvtest::TS::ptr()->get_data_path()) + "hdr/";

    vector<Mat> images_bgr;
    vector<float> times;
    loadExposureSeq(test_path + "exposures/", images_bgr, times);

    vector<Mat> images, images_gray3;
    toGraySeq(images_bgr, images);
    for (const Mat& gray : images)
    {
        Mat gray3;
        cvtColor(gray, gray3, COLOR_GRAY2BGR);
        images_gray3.push_back(gray3);
    }

    Ptr<CalibrateRobertson> calibrate = createCalibrateRobertson();
    Mat response, response3;
    calibrate->process(images, response, times);
    ASSERT_EQ(CV_32FC1, response.type());
    ASSERT_EQ(Size(1, 256), response.size());
    ASSERT_TRUE(cv::checkRange(response));

    // Channels are calibrated independently, so a one-channel image must give the
    // same curve as each channel of the same image repeated three times.
    calibrate->process(images_gray3, response3, times);
    vector<Mat> channels;
    split(response3, channels);
    checkEqual(channels[0], response, 1e-5f, "CalibrateRobertson 1ch vs 3ch");
}

// One exposure of two grey levels at unit exposure time: the curve is exact at the
// observed levels, and every level that occurs in no image gets a zero response.
TEST(Photo_CalibrateRobertson, exact_values_and_unseen_levels)
{
    Mat image(4, 4, CV_8UC1, Scalar(64));
    image.rowRange(2, 4).setTo(128);
    vector<Mat> images(1, image);
    vector<float> times(1, 1.0f);

    Mat response;
    createCalibrateRobertson()->process(images, response, times);

    ASSERT_EQ(CV_32FC1, response.type());
    EXPECT_EQ(0.5f, response.at<float>(64));
    EXPECT_EQ(1.0f, response.at<float>(128));
    for (int i = 0; i < 256; i++)
    {
        if (i != 64 && i != 128)
            EXPECT_EQ(0.0f, response.at<float>(i)) << "level=" << i;
    }
}

// Without the middle level the curve cannot be scaled to 1 there; it is scaled by
// the nearest observed level, which keeps its estimate, and stays finite.
TEST(Photo_CalibrateRobertson, missing_middle_level)
{
    vector<Mat> images(1, Mat(4, 4, CV_8UC3, Scalar(64, 64, 64)));
    vector<float> times(1, 1.0f);

    Mat response;
    createCalibrateRobertson()->process(images, response, times);

    ASSERT_TRUE(cv::checkRange(response));
    EXPECT_EQ(Vec3f::all(0.5f), response.at<Vec3f>(64));
    EXPECT_EQ(Vec3f::all(0.0f), response.at<Vec3f>(128));
}

// The zero response for unseen values holds without any iteration too.
TEST(Photo_CalibrateRobertson, unseen_levels_zero_without_iterations)
{
    vector<Mat> images(1, Mat(4, 4, CV_8UC1, Scalar(64)));
    vector<float> times(1, 1.0f);

    Mat response;
    createCalibrateRobertson(0, 0.01f)->process(images, response, times);

    EXPECT_EQ(0.5f, response.at<float>(64));
    for (int i = 0; i < 256; i++)
    {
        if (i != 64)
            EXPECT_EQ(0.0f, response.at<float>(i)) << "level=" << i;
    }
}

// An all-black image has no value to scale by; the curve stays finite and every
// value that occurs in no image still gets a zero response.
TEST(Photo_CalibrateRobertson, all_black_image)
{
    vector<Mat> images(1, Mat(4, 4, CV_8UC1, Scalar(0)));
    vector<float> times(1, 1.0f);

    Mat response;
    createCalibrateRobertson()->process(images, response, times);

    ASSERT_TRUE(cv::checkRange(response));
    for (int i = 1; i < 256; i++)
        EXPECT_EQ(0.0f, response.at<float>(i)) << "level=" << i;
}

// Channels are calibrated independently apart from the shared stopping test, so
// with a fixed number of iterations a two-channel image gives the curves of its
// two channels calibrated on their own.
TEST(Photo_CalibrateRobertson, two_channels_match_single_channels)
{
    string test_path = string(cvtest::TS::ptr()->get_data_path()) + "hdr/";
    vector<Mat> images_bgr;
    vector<float> times;
    loadExposureSeq(test_path + "exposures/", images_bgr, times);

    vector<Mat> images2, images_b, images_r;
    for (const Mat& bgr : images_bgr)
    {
        vector<Mat> channels;
        split(bgr, channels);
        Mat two;
        merge(vector<Mat>{channels[0], channels[2]}, two);
        images2.push_back(two);
        images_b.push_back(channels[0]);
        images_r.push_back(channels[2]);
    }

    Ptr<CalibrateRobertson> calibrate = createCalibrateRobertson(5, 0.0f);
    Mat response2, response_b, response_r;
    calibrate->process(images2, response2, times);
    calibrate->process(images_b, response_b, times);
    calibrate->process(images_r, response_r, times);

    ASSERT_EQ(CV_32FC2, response2.type());
    vector<Mat> curves;
    split(response2, curves);
    checkEqual(curves[0], response_b, 1e-5f, "CalibrateRobertson 2ch, channel 0");
    checkEqual(curves[1], response_r, 1e-5f, "CalibrateRobertson 2ch, channel 1");
}

// An image region is read row by row, not as if it were continuous memory.
TEST(Photo_CalibrateRobertson, region_matches_copy)
{
    string test_path = string(cvtest::TS::ptr()->get_data_path()) + "hdr/";
    vector<Mat> images;
    vector<float> times;
    loadExposureSeq(test_path + "exposures/", images, times);

    vector<Mat> regions, copies;
    const Rect roi(10, 20, images[0].cols / 2, images[0].rows / 2);
    for (const Mat& image : images)
    {
        regions.push_back(image(roi));
        copies.push_back(image(roi).clone());
    }

    Ptr<CalibrateRobertson> calibrate = createCalibrateRobertson();
    Mat response_region, response_copy;
    calibrate->process(regions, response_region, times);
    calibrate->process(copies, response_copy, times);

    checkEqual(response_copy, response_region, 0, "CalibrateRobertson region vs copy");
}

TEST(Photo_MergeRobertson, regression_1channel)
{
    string test_path = string(cvtest::TS::ptr()->get_data_path()) + "hdr/";

    vector<Mat> images_bgr;
    vector<float> times;
    loadExposureSeq(test_path + "exposures/", images_bgr, times);

    vector<Mat> images, images_gray3;
    toGraySeq(images_bgr, images);
    for (const Mat& gray : images)
    {
        Mat gray3;
        cvtColor(gray, gray3, COLOR_GRAY2BGR);
        images_gray3.push_back(gray3);
    }

    Ptr<MergeRobertson> merge = createMergeRobertson();
    Mat result, result3;
    merge->process(images, result, times);
    ASSERT_EQ(CV_32FC1, result.type());
    ASSERT_EQ(images[0].size(), result.size());

    merge->process(images_gray3, result3, times);
    vector<Mat> channels;
    split(result3, channels);
    checkEqual(channels[0], result, 1e-5f, "MergeRobertson 1ch vs 3ch");
}

TEST(Photo_CalibrateRobertson, bug_18180)
{
    vector<Mat> images;
    vector<cv::String> fn;
    string test_path = cvtest::TS::ptr()->get_data_path() + "hdr/exposures/bug_18180/";
    for(int i = 1; i <= 4; ++i)
        images.push_back(imread(test_path + std::to_string(i) + ".jpg"));
    vector<float> times {15.0f, 2.5f, 0.25f, 0.33f};
    Mat response, expected;
    Ptr<CalibrateRobertson> calibrate = createCalibrateRobertson(2, 0.01f);
    calibrate->process(images, response, times);
    Mat response_no_nans = response.clone();
    patchNaNs(response_no_nans);
    // since there should be no NaNs, original response vs. response with NaNs patched should be identical
    EXPECT_EQ(0.0, cv::norm(response, response_no_nans, NORM_L2));
}

TEST(Photo_CalibrateDebevec, bug_24966)
{
    string test_path = string(cvtest::TS::ptr()->get_data_path()) + "hdr/";
    vector<Mat> all_images;
    vector<float> all_times;
    loadExposureSeq(test_path + "exposures/", all_images, all_times);
    vector<Mat> images;
    vector<float> times;
    for (int idx : {1, 2, 3, 4, 5}) {
        images.push_back(all_images[idx]);
        times.push_back(all_times[idx]);
    }
    Ptr<CalibrateDebevec> calibrate = createCalibrateDebevec();
    Mat response, response_ext;
    calibrate->process(images, response, times);

    // #24966: pixels at 0 and 255 get (almost) zero weight, so a fully saturated and a fully black
    // exposure only add zero-weight equations and must not change the response.
    images.push_back(Mat(images[0].size(), images[0].type(), Scalar::all(255)));
    times.push_back(times.front() * 2);
    images.push_back(Mat(images[0].size(), images[0].type(), Scalar::all(0)));
    times.push_back(times[4] / 2);
    calibrate->process(images, response_ext, times);

    double max_diff = cvtest::norm(response.rowRange(15, 240), response_ext.rowRange(15, 240), NORM_INF);
    EXPECT_LT(max_diff, 0.5);
}
}} // namespace
