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
    double max = 1.0;
    minMaxLoc(abs(img0 - img1), NULL, &max);
    ASSERT_FALSE(max > threshold) << "max=" << max << " threshold=" << threshold << " method=" << name;
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
#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
    ASSERT_LT(max, 0.25);
#else
    ASSERT_LT(max, 0.15);
#endif
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
    // Use a balanced subset of exposures
    vector<int> selected_indices = {1,2,3,4,5};
    vector<Mat> images;
    vector<float> times;
    for (int idx : selected_indices) {
        images.push_back(all_images[idx]);
        times.push_back(all_times[idx]);
    }
    // Run CRF estimation for different sample points
    vector<int> sample_points = {200,300,400};
    vector<Mat> responses;
    for (int samples : sample_points) {
        Ptr<CalibrateDebevec> calibrate = createCalibrateDebevec(samples);
        Mat response;
        calibrate->process(images, response, times);
        Mat roi = response.rowRange(15, 240); //Checking CRF only in the middle of the image
        responses.push_back(roi);
    }

    // Compare consecutive pairs of CRFs
    for (size_t i = 0; i < responses.size()-1; ++i) {
        Mat diff = abs(responses[i] - responses[i+1]);
        double max_diff;
        minMaxLoc(diff, nullptr, &max_diff);
        cout << "max_diff = " << max_diff << endl;
        #if defined(__aarch64__) && defined(__APPLE__)
            ASSERT_LT(max_diff, 10) << "CRF instability detected between samples="
                << sample_points[i] << " and " << sample_points[i+1]
                << " (max diff = " << max_diff << ")";
        #else
            ASSERT_LT(max_diff, 5) << "CRF instability detected between samples="
                << sample_points[i] << " and " << sample_points[i+1]
                << " (max diff = " << max_diff << ")";
        #endif
    }
}

TEST(Photo_TonemapMantiuk, synthetic_hdr_determinism)
{
    RNG& rng = theRNG();
    Mat hdr(96, 128, CV_32FC3);
    for (int y = 0; y < hdr.rows; ++y)
    {
        Vec3f* row = hdr.ptr<Vec3f>(y);
        for (int x = 0; x < hdr.cols; ++x)
        {
            const float base = std::exp((x * 0.03f) + (y * 0.02f) - 2.0f);
            row[x] = Vec3f(base * rng.uniform(0.8f, 1.2f),
                           base * rng.uniform(0.7f, 1.3f),
                           base * rng.uniform(0.9f, 1.1f));
        }
    }

    Ptr<TonemapMantiuk> tm = createTonemapMantiuk(2.2f, 0.7f, 1.0f);
    Mat ldr1, ldr2;
    tm->process(hdr, ldr1);
    tm->process(hdr, ldr2);

    ASSERT_EQ(ldr1.size(), hdr.size());
    ASSERT_EQ(ldr1.type(), CV_32FC3);
    ASSERT_TRUE(checkRange(ldr1, true, nullptr, 0.0, 1.0 + 1e-5));
    EXPECT_EQ(0.0, cvtest::norm(ldr1, ldr2, NORM_INF));
}

TEST(Photo_Tonemap, drago_reinhard_synthetic_reference_parity)
{
    RNG rng(20261113ULL);
    Mat hdr(64, 80, CV_32FC3);
    for (int y = 0; y < hdr.rows; ++y)
    {
        Vec3f* row = hdr.ptr<Vec3f>(y);
        for (int x = 0; x < hdr.cols; ++x)
        {
            const float base = std::exp((x * 0.04f) + (y * 0.03f) - 2.0f);
            row[x] = Vec3f(base * rng.uniform(0.6f, 1.4f),
                           base * rng.uniform(0.5f, 1.5f),
                           base * rng.uniform(0.7f, 1.3f));
        }
    }

    // Verify linear tonemap matches explicit scalar min/max/gamma formula
    {
        Ptr<Tonemap> tm = createTonemap(2.2f);
        Mat out;
        tm->process(hdr, out);
        double mn = 0, mx = 0;
        minMaxLoc(hdr, &mn, &mx);
        Mat ref = (hdr - static_cast<float>(mn)) / static_cast<float>(mx - mn);
        pow(ref, 1.0f / 2.2f, ref);
        EXPECT_LT(cvtest::norm(out, ref, NORM_INF), 3e-5);
    }

    // Verify Drago matches reference channel-split math (saturation == 1.0f and != 1.0f)
    for (float sat : {1.0f, 0.7f})
    {
        const float gamma = 2.2f, bias = 0.85f;
        Ptr<TonemapDrago> tm = createTonemapDrago(gamma, sat, bias);
        Mat out1, out2;
        tm->process(hdr, out1);
        tm->process(hdr, out2);
        EXPECT_EQ(0.0, cvtest::norm(out1, out2, NORM_INF));
        ASSERT_TRUE(checkRange(out1, true, nullptr, 0.0, 1.0 + 1e-5));

        double mn = 0, mx = 0;
        minMaxLoc(hdr, &mn, &mx);
        Mat imgRef = (hdr - static_cast<float>(mn)) / static_cast<float>(mx - mn);
        Mat gray;
        cvtColor(imgRef, gray, COLOR_RGB2GRAY);
        Mat logG;
        max(gray, Scalar::all(1e-4), logG);
        log(logG, logG);
        float meanL = expf(static_cast<float>(sum(logG)[0]) / logG.total());
        gray /= meanL;
        double dmax = 0;
        minMaxLoc(gray, nullptr, &dmax);
        Mat numMap, denMap;
        log(gray + 1.0f, numMap);
        pow(gray / static_cast<float>(dmax), logf(bias) / logf(0.5f), denMap);
        log(2.0f + 8.0f * denMap, denMap);
        Mat newLum = numMap / denMap;
        std::vector<Mat> ch(3);
        split(imgRef, ch);
        for (int c = 0; c < 3; ++c)
        {
            ch[c] = ch[c].mul(1.0f / gray);
            pow(ch[c], sat, ch[c]);
            ch[c] = ch[c].mul(newLum);
        }
        merge(ch, imgRef);
        minMaxLoc(imgRef, &mn, &mx);
        imgRef = (imgRef - static_cast<float>(mn)) / static_cast<float>(mx - mn);
        pow(imgRef, 1.0f / gamma, imgRef);
        EXPECT_LT(cvtest::norm(out1, imgRef, NORM_INF), 1e-5);
    }

    // Verify Reinhard matches reference channel-split math
    {
        const float gamma = 2.2f, intensity = 0.25f, light_adapt = 0.8f, color_adapt = 0.4f;
        Ptr<TonemapReinhard> tm = createTonemapReinhard(gamma, intensity, light_adapt, color_adapt);
        Mat out1, out2;
        tm->process(hdr, out1);
        tm->process(hdr, out2);
        EXPECT_EQ(0.0, cvtest::norm(out1, out2, NORM_INF));
        ASSERT_TRUE(checkRange(out1, true, nullptr, 0.0, 1.0 + 1e-5));

        double mn = 0, mx = 0;
        minMaxLoc(hdr, &mn, &mx);
        Mat imgRef = (hdr - static_cast<float>(mn)) / static_cast<float>(mx - mn);
        Mat gray;
        cvtColor(imgRef, gray, COLOR_RGB2GRAY);
        Mat logG;
        max(gray, Scalar::all(1e-4), logG);
        log(logG, logG);
        float logMean = static_cast<float>(sum(logG)[0] / logG.total());
        double dmin = 0, dmax = 0;
        minMaxLoc(logG, &dmin, &dmax);
        float key = (static_cast<float>(dmax) - logMean) / static_cast<float>(dmax - dmin);
        float mapKey = 0.3f + 0.7f * std::pow(key, 1.4f);
        float intScale = exp(-intensity);
        Scalar cMean = mean(imgRef);
        float gMean = static_cast<float>(mean(gray)[0]);
        std::vector<Mat> ch(3);
        split(imgRef, ch);
        for (int c = 0; c < 3; ++c)
        {
            float global = color_adapt * static_cast<float>(cMean[c]) + (1.0f - color_adapt) * gMean;
            Mat adapt = color_adapt * ch[c] + (1.0f - color_adapt) * gray;
            adapt = light_adapt * adapt + (1.0f - light_adapt) * global;
            pow(intScale * adapt, mapKey, adapt);
            ch[c] = ch[c].mul(1.0f / (adapt + ch[c]));
        }
        merge(ch, imgRef);
        minMaxLoc(imgRef, &mn, &mx);
        imgRef = (imgRef - static_cast<float>(mn)) / static_cast<float>(mx - mn);
        pow(imgRef, 1.0f / gamma, imgRef);
        EXPECT_LT(cvtest::norm(out1, imgRef, NORM_INF), 1e-5);
    }
}

TEST(Photo_Tonemap, zero_luminance_and_flat_finite)
{
    Mat hdr(32, 48, CV_32FC3);
    RNG rng(20261113ULL);
    rng.fill(hdr, RNG::UNIFORM, Scalar(0.1f, 0.1f, 0.1f), Scalar(4.0f, 4.0f, 4.0f));
    hdr.at<Vec3f>(0, 0) = Vec3f(0.0f, 0.0f, 0.0f);
    hdr.at<Vec3f>(3, 5) = Vec3f(0.0f, 0.0f, 0.0f);

    for (float sat : {1.0f, 0.7f})
    {
        Ptr<TonemapDrago> drago = createTonemapDrago(2.2f, sat, 0.85f);
        Mat out;
        drago->process(hdr, out);
        EXPECT_TRUE(checkRange(out, true, nullptr, 0.0, 1.0 + 1e-5));
        EXPECT_EQ(Vec3f(0.0f, 0.0f, 0.0f), out.at<Vec3f>(0, 0));
    }

    Mat flatHdr(16, 16, CV_32FC3, Scalar(0.5f, 0.5f, 0.5f));
    Ptr<TonemapReinhard> reinhard = createTonemapReinhard(2.2f, 0.25f, 0.8f, 0.4f);
    Mat outBlack, outFlat;
    reinhard->process(hdr, outBlack);
    reinhard->process(flatHdr, outFlat);
    EXPECT_TRUE(checkRange(outBlack, true, nullptr, 0.0, 1.0 + 1e-5));
    EXPECT_TRUE(checkRange(outFlat, true, nullptr, 0.0, 1.0 + 1e-5));
}

TEST(Photo_MergeMertens, synthetic_parity)
{
    vector<Mat> imgs3(3);
    for (int i = 0; i < 3; ++i)
    {
        imgs3[i] = Mat(97, 131, CV_8UC3);
        RNG rng(100 + i);
        rng.fill(imgs3[i], RNG::UNIFORM, Scalar::all(20 * i), Scalar::all(120 + 60 * i));
        circle(imgs3[i], Point(40 + 15 * i, 50), 20, Scalar(30 + 80 * i, 200 - 40 * i, 120), FILLED);
    }

    Ptr<MergeMertens> mm1 = createMergeMertens(1.0f, 1.0f, 1.0f);
    Ptr<MergeMertens> mm2 = createMergeMertens(0.5f, 1.2f, 0.8f);
    Mat out3a, out3b, out3w;
    mm1->process(imgs3, out3a);
    mm1->process(imgs3, out3b);
    ASSERT_EQ(CV_32FC3, out3a.type());
    ASSERT_EQ(imgs3[0].size(), out3a.size());
    ASSERT_TRUE(checkRange(out3a, true));
    EXPECT_EQ(0.0, cvtest::norm(out3a, out3b, NORM_INF));

    // Constant single exposure must reproduce normalized input closely regardless of weights
    vector<Mat> single3(1, imgs3[1]);
    Mat singleExpected, singleOut;
    imgs3[1].convertTo(singleExpected, CV_32F, 1.0f / 255.0f);
    mm2->process(single3, singleOut);
    EXPECT_LE(cvtest::norm(singleOut, singleExpected, NORM_INF), 1e-5);

    // Grayscale (1-channel) path
    vector<Mat> imgs1(3);
    for (int i = 0; i < 3; ++i)
        cvtColor(imgs3[i], imgs1[i], COLOR_BGR2GRAY);
    Mat out1a, out1b;
    mm1->process(imgs1, out1a);
    mm2->process(imgs1, out1b);
    ASSERT_EQ(CV_32FC1, out1a.type());
    ASSERT_EQ(imgs1[0].size(), out1a.size());
    ASSERT_TRUE(checkRange(out1a, true));
    ASSERT_TRUE(checkRange(out1b, true));
    mm2->process(imgs3, out3w);
    ASSERT_TRUE(checkRange(out3w, true));

    // Exact mathematical reference for MergeMertens weights + Laplacian pyramid fusion
    auto refMertens = [](const vector<Mat>& seq, float wcon, float wsat, float wexp) -> Mat {
        Size sz = seq[0].size();
        int n = (int)seq.size();
        vector<Mat> w(n), fimgs(n);
        Mat wsum = Mat::zeros(sz, CV_32F);
        for (int i = 0; i < n; ++i) {
            seq[i].convertTo(fimgs[i], CV_32F, 1.0f / 255.0f);
            Mat gray, contrast, sat = Mat::zeros(sz, CV_32F), we = Mat::ones(sz, CV_32F);
            cvtColor(fimgs[i], gray, COLOR_RGB2GRAY);
            Laplacian(gray, contrast, CV_32F);
            contrast = abs(contrast);
            vector<Mat> ch;
            split(fimgs[i], ch);
            Mat mean = (ch[0] + ch[1] + ch[2]) / 3.0f;
            for (int c = 0; c < 3; ++c) {
                Mat dev = ch[c] - mean;
                sat += dev.mul(dev);
                Mat expo;
                exp(-((ch[c] - 0.5f).mul(ch[c] - 0.5f)) / 0.08f, expo);
                we = we.mul(expo);
            }
            sqrt(sat, sat);
            pow(contrast, wcon, contrast);
            pow(sat, wsat, sat);
            pow(we, wexp, we);
            w[i] = contrast.mul(sat).mul(we) + 1e-12f;
            wsum += w[i];
        }
        int maxlvl = (int)(logf((float)min(sz.width, sz.height)) / logf(2.0f));
        vector<Mat> res(maxlvl + 1);
        for (int i = 0; i < n; ++i) {
            w[i] /= wsum;
            vector<Mat> ip, wp;
            buildPyramid(fimgs[i], ip, maxlvl);
            buildPyramid(w[i], wp, maxlvl);
            for (int l = 0; l < maxlvl; ++l) {
                Mat up; pyrUp(ip[l + 1], up, ip[l].size()); ip[l] -= up;
            }
            for (int l = 0; l <= maxlvl; ++l) {
                vector<Mat> ch; split(ip[l], ch);
                for (int c = 0; c < 3; ++c) ch[c] = ch[c].mul(wp[l]);
                merge(ch, ip[l]);
                if (res[l].empty()) res[l] = ip[l]; else res[l] += ip[l];
            }
        }
        for (int l = maxlvl; l > 0; --l) {
            Mat up; pyrUp(res[l], up, res[l - 1].size()); res[l - 1] += up;
        }
        return res[0];
    };
    checkEqual(out3a, refMertens(imgs3, 1.0f, 1.0f, 1.0f), 1e-5, "MergeMertens default ref");
    checkEqual(out3w, refMertens(imgs3, 0.5f, 1.2f, 0.8f), 1e-5, "MergeMertens weighted ref");
}
}} // namespace
