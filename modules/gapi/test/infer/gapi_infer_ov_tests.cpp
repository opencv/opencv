// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation

//#ifdef OPENCV_GAPI_WITH_OPENVINO

#include "../test_precomp.hpp"

#include <opencv2/gapi/infer/ov.hpp>

#include <openvino/openvino.hpp>

namespace opencv_test
{

// FIXME: taken from DNN module
static void initDLDTDataPath()
{
#ifndef WINRT
    static bool initialized = false;
    if (!initialized)
    {
        const char* omzDataPath = getenv("OPENCV_OPEN_MODEL_ZOO_DATA_PATH");
        if (omzDataPath)
            cvtest::addDataSearchPath(omzDataPath);
        const char* dnnDataPath = getenv("OPENCV_DNN_TEST_DATA_PATH");
        if (dnnDataPath) {
            // Add the dnnDataPath itself - G-API is using some images there directly
            cvtest::addDataSearchPath(dnnDataPath);
            cvtest::addDataSearchPath(dnnDataPath + std::string("/omz_intel_models"));
        }
        initialized = true;
    }
#endif // WINRT
}

#if INF_ENGINE_RELEASE >= 2020010000
static const std::string SUBDIR = "intel/age-gender-recognition-retail-0013/FP32/";
#else
static const std::string SUBDIR = "Retail/object_attributes/age_gender/dldt/";
#endif

TEST(TestAgeGenderOV, InferTensor)
{
    initDLDTDataPath();

    const std::string xml_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    const std::string bin_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    const std::string device   = "CPU";

    //cv::Mat img = cv::imread(Configure);
    ///home/atalaman/workspace/opencv_extra/testdata/cv/face/david2.jpg
    //
    //cv::Mat in_mat(62, 62, CV_8UC3);
    cv::Mat in_mat({1, 3, 62, 62}, CV_32F);
    cv::randu(in_mat, -1, 1);

    cv::Mat ov_age, ov_gender;
    //cv::Mat in_mat(120, 120, CV_8UC3);

    //{
    ov::Core core;
    auto model = core.read_model(xml_path, bin_path);
    auto compiled_model = core.compile_model(model, device);
    auto infer_request  = compiled_model.create_infer_request();
    infer_request.set_input_tensor(
            ov::Tensor(ov::element::f32,
                      {1, 3, 62, 62},
                      in_mat.ptr<void>()));

    infer_request.infer();

    auto age_tensor = infer_request.get_tensor("age_conv3");
    auto shape = age_tensor.get_shape();
    std::vector<int> dims{shape.begin(), shape.end()};
    ov_age = cv::Mat(dims, CV_32F, age_tensor.data());

    auto gender_tensor = infer_request.get_tensor("prob");
    shape = gender_tensor.get_shape();
    std::vector<int> dims2{shape.begin(), shape.end()};
    ov_gender = cv::Mat(dims2, CV_32F, gender_tensor.data());
    //}

    // Configure & run G-API
    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    cv::GMat in;
    cv::GMat age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(age, gender));

    auto pp = cv::gapi::ov::Params<AgeGender> {
        xml_path, bin_path, device
    }.cfgOutputLayers({ "age_conv3", "prob" });

     //.cfgTensorOutputPrecision(CV_32F),
     //.cfgTensorInputLayout("NHWC"),
     //.cfgModelInputLayout("NCHW"),
     //

    //comp.compile(cv::descr_of(img), cv::compile_args(cv::gapi::networks(pp)));

    cv::Mat gapi_age, gapi_gender;
    comp.apply(cv::gin(in_mat), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    std::cout << "G-API Age: " << gapi_age.ptr<float>()[0] << std::endl;
    std::cout << "OV Age: "    << ov_age.ptr<float>()[0] << std::endl;

    std::cout << std::endl;

    std::cout << "G-API Gender: " << gapi_gender.ptr<float>()[0] << " "
                                  << gapi_gender.ptr<float>()[1] << std::endl;

    std::cout << "OV Gender: " << ov_gender.ptr<float>()[0] << " "
                               << ov_gender.ptr<float>()[1] << std::endl;
}

TEST(TestAgeGenderOV, InferImage)
{
    initDLDTDataPath();

    const std::string xml_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    const std::string bin_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    const std::string device   = "CPU";

    cv::Mat in_mat(62, 62, CV_8UC3);
    cv::randu(in_mat, 0, 255);

    cv::Mat ov_age, ov_gender;

    ov::Core core;
    auto model = core.read_model(xml_path, bin_path);

    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor().set_layout(ov::Layout("NHWC"));
    ppp.input().tensor().set_element_type(ov::element::u8);
    ppp.input().model().set_layout(ov::Layout("NCHW"));
    model = ppp.build();

    auto compiled_model = core.compile_model(model, device);
    auto infer_request  = compiled_model.create_infer_request();
    infer_request.set_input_tensor(
            ov::Tensor(ov::element::u8,
                      {1, 62, 62, 3},
                      in_mat.ptr<void>()));

    infer_request.infer();

    auto age_tensor = infer_request.get_tensor("age_conv3");
    auto shape = age_tensor.get_shape();
    std::vector<int> dims{shape.begin(), shape.end()};
    ov_age = cv::Mat(dims, CV_32F, age_tensor.data());

    auto gender_tensor = infer_request.get_tensor("prob");
    shape = gender_tensor.get_shape();
    std::vector<int> dims2{shape.begin(), shape.end()};
    ov_gender = cv::Mat(dims2, CV_32F, gender_tensor.data());

    // Configure & run G-API
    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    cv::GMat in;
    cv::GMat age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(age, gender));

    auto pp = cv::gapi::ov::Params<AgeGender> {
        xml_path, bin_path, device
    }.cfgOutputLayers({ "age_conv3", "prob" });

     //.cfgTensorOutputPrecision(CV_32F),
     //.cfgTensorInputLayout("NHWC"),
     //.cfgModelInputLayout("NCHW"),
     //

    //comp.compile(cv::descr_of(img), cv::compile_args(cv::gapi::networks(pp)));

    cv::Mat gapi_age, gapi_gender;
    comp.apply(cv::gin(in_mat), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    std::cout << "G-API Age: " << gapi_age.ptr<float>()[0] << std::endl;
    std::cout << "OV Age: "    << ov_age.ptr<float>()[0] << std::endl;

    std::cout << std::endl;

    std::cout << "G-API Gender: " << gapi_gender.ptr<float>()[0] << " "
                                  << gapi_gender.ptr<float>()[1] << std::endl;

    std::cout << "OV Gender: " << ov_gender.ptr<float>()[0] << " "
                               << ov_gender.ptr<float>()[1] << std::endl;
}

} // namespace opencv_test

//#endif // OPENCV_GAPI_WITH_OPENVINO
