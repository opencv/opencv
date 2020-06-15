// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

namespace opencv_test { namespace {

#define OUTPUT_SAVING 0
#if OUTPUT_SAVING
#define SAVE(x) std::vector<int> params;\
                params.push_back(16);\
                params.push_back(0);\
                imwrite(folder + "output.png", x ,params);
#else
#define SAVE(x)
#endif

static const double numerical_precision = 0.05; // 95% of pixels should have exact values

TEST(Photo_LayerModelBlend_DARKEN, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
    string target_path = "samples/cpp/lena.jpg";
    string reference_path = folder + "DARKEN_RESULT.jpg";
    
    Mat target = imread(target_path, IMREAD_COLOR);
    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
    GaussianBlur(target, blend, Size(33, 33), 0);
    
    Mat result(target.size(), CV_8UC3, Scalar::all(0));
    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);
    
    SAVE(result);
    
    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;
    
    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}
TEST(Photo_LayerModelBlend_MULTIPY, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
    string target_path = "samples/cpp/lena.jpg";
    string reference_path = folder + "MULTIPY_RESULT.jpg";

    Mat target = imread(target_path, IMREAD_COLOR);
    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
    GaussianBlur(target, blend, Size(33, 33), 0);

    Mat result(target.size(), CV_8UC3, Scalar::all(0));
    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}
TEST(Photo_LayerModelBlend_COLOR_BURN, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
    string target_path = "samples/cpp/lena.jpg";
    string reference_path = folder + "COLOR_BURN_RESULT.jpg";

    Mat target = imread(target_path, IMREAD_COLOR);
    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
    GaussianBlur(target, blend, Size(33, 33), 0);

    Mat result(target.size(), CV_8UC3, Scalar::all(0));
    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}
TEST(Photo_LayerModelBlend_LINEAR_BRUN, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
    string target_path = "samples/cpp/lena.jpg";
    string reference_path = folder + "LINEAR_BRUN_RESULT.jpg";

    Mat target = imread(target_path, IMREAD_COLOR);
    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
    GaussianBlur(target, blend, Size(33, 33), 0);

    Mat result(target.size(), CV_8UC3, Scalar::all(0));
    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}
TEST(Photo_LayerModelBlend_LIGHTEN, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
    string target_path = "samples/cpp/lena.jpg";
    string reference_path = folder + "LIGHTEN_RESULT.jpg";

    Mat target = imread(target_path, IMREAD_COLOR);
    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
    GaussianBlur(target, blend, Size(33, 33), 0);

    Mat result(target.size(), CV_8UC3, Scalar::all(0));
    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}
TEST(Photo_LayerModelBlend_SCREEN, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
    string target_path = "samples/cpp/lena.jpg";
    string reference_path = folder + "SCREEN_RESULT.jpg";

    Mat target = imread(target_path, IMREAD_COLOR);
    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
    GaussianBlur(target, blend, Size(33, 33), 0);

    Mat result(target.size(), CV_8UC3, Scalar::all(0));
    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}
TEST(Photo_LayerModelBlend_COLOR_DODGE, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
    string target_path = "samples/cpp/lena.jpg";
    string reference_path = folder + "COLOR_DODGE_RESULT.jpg";

    Mat target = imread(target_path, IMREAD_COLOR);
    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
    GaussianBlur(target, blend, Size(33, 33), 0);

    Mat result(target.size(), CV_8UC3, Scalar::all(0));
    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}
TEST(Photo_LayerModelBlend_LINEAR_DODGE, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
    string target_path = "samples/cpp/lena.jpg";
    string reference_path = folder + "LINEAR_DODGE_RESULT.jpg";

    Mat target = imread(target_path, IMREAD_COLOR);
    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
    GaussianBlur(target, blend, Size(33, 33), 0);

    Mat result(target.size(), CV_8UC3, Scalar::all(0));
    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}
TEST(Photo_LayerModelBlend_OVERLAY, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
    string target_path = "samples/cpp/lena.jpg";
    string reference_path = folder + "OVERLAY_RESULT.jpg";

    Mat target = imread(target_path, IMREAD_COLOR);
    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
    GaussianBlur(target, blend, Size(33, 33), 0);

    Mat result(target.size(), CV_8UC3, Scalar::all(0));
    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}
TEST(Photo_LayerModelBlend_SOFT_LIGHT, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
    string target_path = "samples/cpp/lena.jpg";
    string reference_path = folder + "SOFT_LIGHT_RESULT.jpg";

    Mat target = imread(target_path, IMREAD_COLOR);
    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
    GaussianBlur(target, blend, Size(33, 33), 0);

    Mat result(target.size(), CV_8UC3, Scalar::all(0));
    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}
TEST(Photo_LayerModelBlend_HARD_LIGHT, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
    string target_path = "samples/cpp/lena.jpg";
    string reference_path = folder + "HARD_LIGHT_RESULT.jpg";

    Mat target = imread(target_path, IMREAD_COLOR);
    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
    GaussianBlur(target, blend, Size(33, 33), 0);

    Mat result(target.size(), CV_8UC3, Scalar::all(0));
    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}
TEST(Photo_LayerModelBlend_VIVID_LIGHT, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
    string target_path = "samples/cpp/lena.jpg";
    string reference_path = folder + "VIVID_LIGHT_RESULT.jpg";

    Mat target = imread(target_path, IMREAD_COLOR);
    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
    GaussianBlur(target, blend, Size(33, 33), 0);

    Mat result(target.size(), CV_8UC3, Scalar::all(0));
    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}
TEST(Photo_LayerModelBlend_LINEAR_LIGHT, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
    string target_path = "samples/cpp/lena.jpg";
    string reference_path = folder + "LINEAR_LIGHT_RESULT.jpg";

    Mat target = imread(target_path, IMREAD_COLOR);
    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
    GaussianBlur(target, blend, Size(33, 33), 0);

    Mat result(target.size(), CV_8UC3, Scalar::all(0));
    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}
TEST(Photo_LayerModelBlend_PIN_LIGHT, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
    string target_path = "samples/cpp/lena.jpg";
    string reference_path = folder + "PIN_LIGHT_RESULT.jpg";

    Mat target = imread(target_path, IMREAD_COLOR);
    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
    GaussianBlur(target, blend, Size(33, 33), 0);

    Mat result(target.size(), CV_8UC3, Scalar::all(0));
    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}
TEST(Photo_LayerModelBlend_DIFFERENCE, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
    string target_path = "samples/cpp/lena.jpg";
    string reference_path = folder + "DIFFERENCE_RESULT.jpg";

    Mat target = imread(target_path, IMREAD_COLOR);
    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
    GaussianBlur(target, blend, Size(33, 33), 0);

    Mat result(target.size(), CV_8UC3, Scalar::all(0));
    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}
TEST(Photo_LayerModelBlend_EXCLUSION, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
    string target_path = "samples/cpp/lena.jpg";
    string reference_path = folder + "EXCLUSION_RESULT.jpg";

    Mat target = imread(target_path, IMREAD_COLOR);
    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
    GaussianBlur(target, blend, Size(33, 33), 0);

    Mat result(target.size(), CV_8UC3, Scalar::all(0));
    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}
TEST(Photo_LayerModelBlend_DIVIDE, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
    string target_path = "samples/cpp/lena.jpg";
    string reference_path = folder + "DIVIDE_RESULT.jpg";

    Mat target = imread(target_path, IMREAD_COLOR);
    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
    GaussianBlur(target, blend, Size(33, 33), 0);

    Mat result(target.size(), CV_8UC3, Scalar::all(0));
    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);

    SAVE(result);

    Mat reference = imread(reference_path);
    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;

    double errorINF = cvtest::norm(reference, result, NORM_INF);
    EXPECT_LE(errorINF, 1);
    double errorL1 = cvtest::norm(reference, result, NORM_L1);
    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
}
}}// namespace
