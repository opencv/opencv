// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#include "../test_precomp.hpp"

#ifdef HAVE_INF_ENGINE

#include <stdexcept>

#include <inference_engine.hpp>

#include <ade/util/iota_range.hpp>

#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/infer/ie/util.hpp>

namespace opencv_test
{
namespace {

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

// FIXME: taken from the DNN module
void normAssert(cv::InputArray ref, cv::InputArray test,
                const char *comment /*= ""*/,
                double l1 = 0.00001, double lInf = 0.0001)
{
    double normL1 = cvtest::norm(ref, test, cv::NORM_L1) / ref.getMat().total();
    EXPECT_LE(normL1, l1) << comment;

    double normInf = cvtest::norm(ref, test, cv::NORM_INF);
    EXPECT_LE(normInf, lInf) << comment;
}

} // anonymous namespace

// TODO: Probably DNN/IE part can be further parametrized with a template
// NOTE: here ".." is used to leave the default "gapi/" search scope
TEST(TestAgeGenderIE, InferBasicTensor)
{
    initDLDTDataPath();

    const std::string path = "Retail/object_attributes/age_gender/dldt/age-gender-recognition-retail-0013";
    const auto topology_path = findDataFile(path + ".xml");
    const auto weights_path  = findDataFile(path + ".bin");

    // Load IE network, initialize input data using that.
    namespace IE = InferenceEngine;
    cv::Mat in_mat;
    cv::Mat gapi_age, gapi_gender;

    IE::Blob::Ptr ie_age, ie_gender;
    {
        IE::CNNNetReader reader;
        reader.ReadNetwork(topology_path);
        reader.ReadWeights(weights_path);
        auto net = reader.getNetwork();

        const auto &iedims = net.getInputsInfo().begin()->second->getDims();
        std::vector<int> cvdims(iedims.rbegin(), iedims.rend());
        in_mat.create(cvdims, CV_32F);
        cv::randu(in_mat, -1, 1);

        auto plugin = IE::PluginDispatcher().getPluginByDevice("CPU");
        auto plugin_net = plugin.LoadNetwork(net, {});
        auto infer_request = plugin_net.CreateInferRequest();

        infer_request.SetBlob("data", cv::gapi::ie::util::to_ie(in_mat));
        infer_request.Infer();
        ie_age    = infer_request.GetBlob("age_conv3");
        ie_gender = infer_request.GetBlob("prob");
    }

    // Configure & run G-API
    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    cv::GMat in;
    cv::GMat age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<AgeGender> {
        topology_path, weights_path, "CPU"
    }.cfgOutputLayers({ "age_conv3", "prob" });
    comp.apply(cv::gin(in_mat), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(cv::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(cv::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
}

TEST(TestAgeGenderIE, InferBasicImage)
{
    initDLDTDataPath();

    const std::string path = "Retail/object_attributes/age_gender/dldt/age-gender-recognition-retail-0013";
    const auto topology_path = findDataFile(path + ".xml");
    const auto weights_path  = findDataFile(path + ".bin");

    cv::Mat in_mat = cv::imread(findDataFile("grace_hopper_227.png"));
    cv::Mat gapi_age, gapi_gender;

    // Load & run IE network
    namespace IE = InferenceEngine;
    IE::Blob::Ptr ie_age, ie_gender;
    {
        IE::CNNNetReader reader;
        reader.ReadNetwork(topology_path);
        reader.ReadWeights(weights_path);
        auto net = reader.getNetwork();
        auto &ii = net.getInputsInfo().at("data");
        ii->setPrecision(IE::Precision::U8);
        ii->setLayout(IE::Layout::NHWC);
        ii->getPreProcess().setResizeAlgorithm(IE::RESIZE_BILINEAR);

        auto plugin = IE::PluginDispatcher().getPluginByDevice("CPU");
        auto plugin_net = plugin.LoadNetwork(net, {});
        auto infer_request = plugin_net.CreateInferRequest();

        infer_request.SetBlob("data", cv::gapi::ie::util::to_ie(in_mat));
        infer_request.Infer();
        ie_age    = infer_request.GetBlob("age_conv3");
        ie_gender = infer_request.GetBlob("prob");
    }

    // Configure & run G-API
    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    cv::GMat in;
    cv::GMat age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<AgeGender> {
        topology_path, weights_path, "CPU"
    }.cfgOutputLayers({ "age_conv3", "prob" });
    comp.apply(cv::gin(in_mat), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(cv::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(cv::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
}

} // namespace opencv_test

#endif //  HAVE_INF_ENGINE
