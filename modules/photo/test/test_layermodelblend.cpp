// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

using namespace cvtest::ocl;

namespace opencv_test { namespace {

//==================================================================================================

typedef Point3_<uchar> Pixel;

struct GradFiller_8UC3_2D
{
    GradFiller_8UC3_2D(uchar start_, uchar finish_, int sz_) : start(start_), finish(finish_), sz(sz_) {}
    void operator()(Pixel & pix, const int pos[]) const
    {
        const double p = (double)(pos[0] + pos[1]) / (2 * (sz - 1));
        pix.x = pix.y = pix.z = (uchar)cvRound(start * (1. - p) + finish * p);
    }
private:
    uchar start, finish;
    int sz;
};

Mat createSquareDiagGradient(size_t sz, uchar top_left, uchar bottom_right)
{
    Mat result(Size(sz, sz), CV_8UC3);
    result.forEach<Pixel>(GradFiller_8UC3_2D(top_left, bottom_right, sz));
    return result;
}

//==================================================================================================

static const double numerical_precision = 0.05; // 95% of pixels should have exact values


class Photo_Blend : public testing::Test
{
public:
    Photo_Blend() :
        sz(256),
        ssz(sz, sz),
        white(ssz, CV_8UC3, Scalar::all(255)),
        black(ssz, CV_8UC3, Scalar::all(0)),
        grad(createSquareDiagGradient(sz, 0, 255))
    {
    }

    void twoBlends(InputArray a, InputArray b, int flag)
    {
        layerModelBlending(a, b, res_direct, flag);
        layerModelBlending(b, a, res_reverse, flag);
    }

protected:
    const short sz;
    const Size ssz;
    const Mat white;
    const Mat black;
    const Mat grad;
    Mat res_direct, res_reverse;
};

//==================================================================================================

TEST_F(Photo_Blend, basic_Darken)
{
    twoBlends(white, black, BLEND_MODEL_DARKEN);
    EXPECT_MAT_NEAR(res_direct, black, 0);
    EXPECT_MAT_NEAR(res_reverse, black, 0);

    twoBlends(grad, black, BLEND_MODEL_DARKEN);
    EXPECT_MAT_NEAR(res_direct, black, 0);
    EXPECT_MAT_NEAR(res_reverse, black, 0);

    twoBlends(grad, white, BLEND_MODEL_DARKEN);
    EXPECT_MAT_NEAR(res_direct, grad, 0);
    EXPECT_MAT_NEAR(res_reverse, grad, 0);
}

TEST_F(Photo_Blend, basic_Lighten)
{
    twoBlends(white, black, BLEND_MODEL_LIGHTEN);
    EXPECT_MAT_NEAR(res_direct, white, 0);
    EXPECT_MAT_NEAR(res_reverse, white, 0);

    twoBlends(grad, black, BLEND_MODEL_LIGHTEN);
    EXPECT_MAT_NEAR(res_direct, grad, 0);
    EXPECT_MAT_NEAR(res_reverse, grad, 0);

    twoBlends(grad, white, BLEND_MODEL_LIGHTEN);
    EXPECT_MAT_NEAR(res_direct, white, 0);
    EXPECT_MAT_NEAR(res_reverse, white, 0);
}


TEST_F(Photo_Blend, basic_Multiply)
{
    twoBlends(white, black, BLEND_MODEL_MULTIPY);
    EXPECT_MAT_NEAR(res_direct, black, 0);
    EXPECT_MAT_NEAR(res_reverse, black, 0);

    twoBlends(grad, black, BLEND_MODEL_MULTIPY);
    EXPECT_MAT_NEAR(res_direct, black, 0);
    EXPECT_MAT_NEAR(res_reverse, black, 0);

    twoBlends(grad, white, BLEND_MODEL_MULTIPY);
    EXPECT_MAT_NEAR(res_direct, grad, 0);
    EXPECT_MAT_NEAR(res_reverse, grad, 0);
}

TEST_F(Photo_Blend, basic_ColorBurn)
{
    twoBlends(white, black, BLEND_MODEL_COLOR_BURN);
    EXPECT_MAT_NEAR(res_direct, black, 0);
    EXPECT_MAT_NEAR(res_reverse, black, 0);

    twoBlends(grad, black, BLEND_MODEL_COLOR_BURN);
    EXPECT_MAT_NEAR(res_direct, black, 0);
    EXPECT_MAT_NEAR(res_reverse, black, 0);

    twoBlends(grad, white, BLEND_MODEL_COLOR_BURN);
    EXPECT_MAT_NEAR(res_direct, grad, 0);
//    EXPECT_MAT_NEAR(res_reverse, white, 0);
}


TEST_F(Photo_Blend, basic_LinearBurn)
{
    twoBlends(white, black, BLEND_MODEL_LINEAR_BURN);
    EXPECT_MAT_NEAR(res_direct, black, 0);
    EXPECT_MAT_NEAR(res_reverse, black, 0);

    twoBlends(grad, black, BLEND_MODEL_LINEAR_BURN);
    EXPECT_MAT_NEAR(res_direct, black, 0);
    EXPECT_MAT_NEAR(res_reverse, black, 0);


    twoBlends(grad, white, BLEND_MODEL_LINEAR_BURN);
    EXPECT_MAT_NEAR(res_direct, grad, 0);
    EXPECT_MAT_NEAR(res_reverse, grad, 0);
}


TEST_F(Photo_Blend, basic_Screen)
{
    twoBlends(white, black, BLEND_MODEL_SCREEN);
    EXPECT_MAT_NEAR(res_direct, white, 0);
    EXPECT_MAT_NEAR(res_reverse, white, 0);

    twoBlends(grad, black, BLEND_MODEL_SCREEN);
//    EXPECT_MAT_NEAR(res_direct, grad, 0);
//    EXPECT_MAT_NEAR(res_reverse, grad, 0);

    twoBlends(grad, white, BLEND_MODEL_SCREEN);
    EXPECT_MAT_NEAR(res_direct, white, 0);
    EXPECT_MAT_NEAR(res_reverse, white, 0);
}


TEST_F(Photo_Blend, basic_ColorDodge)
{
    twoBlends(white, black, BLEND_MODEL_COLOR_DODGE);
    EXPECT_MAT_NEAR(res_direct, white, 0);
    EXPECT_MAT_NEAR(res_reverse, white, 0);

    twoBlends(grad, black, BLEND_MODEL_COLOR_DODGE);
    EXPECT_MAT_NEAR(res_direct, grad, 0);
//    EXPECT_MAT_NEAR(res_reverse, grad, 0);

    twoBlends(grad, white, BLEND_MODEL_COLOR_DODGE);
    EXPECT_MAT_NEAR(res_direct, white, 0);
    EXPECT_MAT_NEAR(res_reverse, white, 0);
}

TEST_F(Photo_Blend, basic_LinearDodge)
{
    twoBlends(white, black, BLEND_MODEL_LINEAR_DODGE);
    EXPECT_MAT_NEAR(res_direct, white, 0);
    EXPECT_MAT_NEAR(res_reverse, white, 0);

    twoBlends(grad, black, BLEND_MODEL_LINEAR_DODGE);
    EXPECT_MAT_NEAR(res_direct, grad, 0);
    EXPECT_MAT_NEAR(res_reverse, grad, 0);

    twoBlends(grad, white, BLEND_MODEL_LINEAR_DODGE);
    EXPECT_MAT_NEAR(res_direct, white, 0);
    EXPECT_MAT_NEAR(res_reverse, white, 0);
}

//TEST(Photo_LayerModelBlend_LINEAR_DODGE, regression)
//{
//    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
//    string target_path = "samples/cpp/lena.jpg";
//    string reference_path = folder + "LINEAR_DODGE_RESULT.jpg";
//    Mat target = imread(target_path, IMREAD_COLOR);
//    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
//    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
//    GaussianBlur(target, blend, Size(33, 33), 0);
//    Mat result(target.size(), CV_8UC3, Scalar::all(0));
//    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);
//    SAVE(result);
//    Mat reference = imread(reference_path);
//    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;
//    double errorINF = cvtest::norm(reference, result, NORM_INF);
//    EXPECT_LE(errorINF, 1);
//    double errorL1 = cvtest::norm(reference, result, NORM_L1);
//    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
//}
//TEST(Photo_LayerModelBlend_OVERLAY, regression)
//{
//    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
//    string target_path = "samples/cpp/lena.jpg";
//    string reference_path = folder + "OVERLAY_RESULT.jpg";
//    Mat target = imread(target_path, IMREAD_COLOR);
//    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
//    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
//    GaussianBlur(target, blend, Size(33, 33), 0);
//    Mat result(target.size(), CV_8UC3, Scalar::all(0));
//    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);
//    SAVE(result);
//    Mat reference = imread(reference_path);
//    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;
//    double errorINF = cvtest::norm(reference, result, NORM_INF);
//    EXPECT_LE(errorINF, 1);
//    double errorL1 = cvtest::norm(reference, result, NORM_L1);
//    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
//}
//TEST(Photo_LayerModelBlend_SOFT_LIGHT, regression)
//{
//    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
//    string target_path = "samples/cpp/lena.jpg";
//    string reference_path = folder + "SOFT_LIGHT_RESULT.jpg";
//    Mat target = imread(target_path, IMREAD_COLOR);
//    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
//    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
//    GaussianBlur(target, blend, Size(33, 33), 0);
//    Mat result(target.size(), CV_8UC3, Scalar::all(0));
//    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);
//    SAVE(result);
//    Mat reference = imread(reference_path);
//    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;
//    double errorINF = cvtest::norm(reference, result, NORM_INF);
//    EXPECT_LE(errorINF, 1);
//    double errorL1 = cvtest::norm(reference, result, NORM_L1);
//    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
//}
//TEST(Photo_LayerModelBlend_HARD_LIGHT, regression)
//{
//    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
//    string target_path = "samples/cpp/lena.jpg";
//    string reference_path = folder + "HARD_LIGHT_RESULT.jpg";
//    Mat target = imread(target_path, IMREAD_COLOR);
//    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
//    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
//    GaussianBlur(target, blend, Size(33, 33), 0);
//    Mat result(target.size(), CV_8UC3, Scalar::all(0));
//    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);
//    SAVE(result);
//    Mat reference = imread(reference_path);
//    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;
//    double errorINF = cvtest::norm(reference, result, NORM_INF);
//    EXPECT_LE(errorINF, 1);
//    double errorL1 = cvtest::norm(reference, result, NORM_L1);
//    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
//}
//TEST(Photo_LayerModelBlend_VIVID_LIGHT, regression)
//{
//    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
//    string target_path = "samples/cpp/lena.jpg";
//    string reference_path = folder + "VIVID_LIGHT_RESULT.jpg";
//    Mat target = imread(target_path, IMREAD_COLOR);
//    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
//    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
//    GaussianBlur(target, blend, Size(33, 33), 0);
//    Mat result(target.size(), CV_8UC3, Scalar::all(0));
//    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);
//    SAVE(result);
//    Mat reference = imread(reference_path);
//    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;
//    double errorINF = cvtest::norm(reference, result, NORM_INF);
//    EXPECT_LE(errorINF, 1);
//    double errorL1 = cvtest::norm(reference, result, NORM_L1);
//    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
//}
//TEST(Photo_LayerModelBlend_LINEAR_LIGHT, regression)
//{
//    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
//    string target_path = "samples/cpp/lena.jpg";
//    string reference_path = folder + "LINEAR_LIGHT_RESULT.jpg";
//    Mat target = imread(target_path, IMREAD_COLOR);
//    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
//    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
//    GaussianBlur(target, blend, Size(33, 33), 0);
//    Mat result(target.size(), CV_8UC3, Scalar::all(0));
//    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);
//    SAVE(result);
//    Mat reference = imread(reference_path);
//    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;
//    double errorINF = cvtest::norm(reference, result, NORM_INF);
//    EXPECT_LE(errorINF, 1);
//    double errorL1 = cvtest::norm(reference, result, NORM_L1);
//    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
//}
//TEST(Photo_LayerModelBlend_PIN_LIGHT, regression)
//{
//    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
//    string target_path = "samples/cpp/lena.jpg";
//    string reference_path = folder + "PIN_LIGHT_RESULT.jpg";
//    Mat target = imread(target_path, IMREAD_COLOR);
//    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
//    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
//    GaussianBlur(target, blend, Size(33, 33), 0);
//    Mat result(target.size(), CV_8UC3, Scalar::all(0));
//    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);
//    SAVE(result);
//    Mat reference = imread(reference_path);
//    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;
//    double errorINF = cvtest::norm(reference, result, NORM_INF);
//    EXPECT_LE(errorINF, 1);
//    double errorL1 = cvtest::norm(reference, result, NORM_L1);
//    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
//}
//TEST(Photo_LayerModelBlend_DIFFERENCE, regression)
//{
//    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
//    string target_path = "samples/cpp/lena.jpg";
//    string reference_path = folder + "DIFFERENCE_RESULT.jpg";
//    Mat target = imread(target_path, IMREAD_COLOR);
//    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
//    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
//    GaussianBlur(target, blend, Size(33, 33), 0);
//    Mat result(target.size(), CV_8UC3, Scalar::all(0));
//    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);
//    SAVE(result);
//    Mat reference = imread(reference_path);
//    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;
//    double errorINF = cvtest::norm(reference, result, NORM_INF);
//    EXPECT_LE(errorINF, 1);
//    double errorL1 = cvtest::norm(reference, result, NORM_L1);
//    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
//}
//TEST(Photo_LayerModelBlend_EXCLUSION, regression)
//{
//    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
//    string target_path = "samples/cpp/lena.jpg";
//    string reference_path = folder + "EXCLUSION_RESULT.jpg";
//    Mat target = imread(target_path, IMREAD_COLOR);
//    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
//    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
//    GaussianBlur(target, blend, Size(33, 33), 0);
//    Mat result(target.size(), CV_8UC3, Scalar::all(0));
//    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);
//    SAVE(result);
//    Mat reference = imread(reference_path);
//    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;
//    double errorINF = cvtest::norm(reference, result, NORM_INF);
//    EXPECT_LE(errorINF, 1);
//    double errorL1 = cvtest::norm(reference, result, NORM_L1);
//    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
//}
//TEST(Photo_LayerModelBlend_DIVIDE, regression)
//{
//    string folder = string(cvtest::TS::ptr()->get_data_path()) + "LayerModelBlend/";
//    string target_path = "samples/cpp/lena.jpg";
//    string reference_path = folder + "DIVIDE_RESULT.jpg";
//    Mat target = imread(target_path, IMREAD_COLOR);
//    ASSERT_FALSE(target.empty()) << "Could not load target image " << target_path;
//    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
//    GaussianBlur(target, blend, Size(33, 33), 0);
//    Mat result(target.size(), CV_8UC3, Scalar::all(0));
//    layerModelBlending(target, blend, result, BLEND_MODEL_DARKEN);
//    SAVE(result);
//    Mat reference = imread(reference_path);
//    ASSERT_FALSE(reference.empty()) << "Could not load reference image " << reference_path;
//    double errorINF = cvtest::norm(reference, result, NORM_INF);
//    EXPECT_LE(errorINF, 1);
//    double errorL1 = cvtest::norm(reference, result, NORM_L1);
//    EXPECT_LE(errorL1, reference.total() * numerical_precision) << "size=" << reference.size();
//}
}}// namespace
