// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#include "../test_precomp.hpp"

#ifdef HAVE_INF_ENGINE

#include <stdexcept>

////////////////////////////////////////////////////////////////////////////////
// FIXME: Suppress deprecation warnings for OpenVINO 2019R2+
// BEGIN {{{
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#ifdef _MSC_VER
#pragma warning(disable: 4996)  // was declared deprecated
#endif

#if defined(__GNUC__)
#pragma GCC visibility push(default)
#endif

#include <inference_engine.hpp>

#if defined(__GNUC__)
#pragma GCC visibility pop
#endif
// END }}}
////////////////////////////////////////////////////////////////////////////////

#include <ade/util/iota_range.hpp>

#include <opencv2/gapi/infer/ie.hpp>

#include "backends/ie/util.hpp"

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
    const auto topology_path = findDataFile(path + ".xml", false);
    const auto weights_path  = findDataFile(path + ".bin", false);

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

        const auto &iedims = net.getInputsInfo().begin()->second->getTensorDesc().getDims();
              auto  cvdims = cv::gapi::ie::util::to_ocv(iedims);
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
    const auto topology_path = findDataFile(path + ".xml", false);
    const auto weights_path  = findDataFile(path + ".bin", false);

    // FIXME: Ideally it should be an image from disk
    // cv::Mat in_mat = cv::imread(findDataFile("grace_hopper_227.png"));
    cv::Mat in_mat(cv::Size(320, 240), CV_8UC3);
    cv::randu(in_mat, 0, 255);

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

TEST(TestAgeGenderIE, InferROIList)
{
    initDLDTDataPath();

    const std::string path = "Retail/object_attributes/age_gender/dldt/age-gender-recognition-retail-0013";
    const auto topology_path = findDataFile(path + ".xml", false);
    const auto weights_path  = findDataFile(path + ".bin", false);

    // FIXME: Ideally it should be an image from disk
    // cv::Mat in_mat = cv::imread(findDataFile("grace_hopper_227.png"));
    cv::Mat in_mat(cv::Size(640, 480), CV_8UC3);
    cv::randu(in_mat, 0, 255);

    std::vector<cv::Rect> rois = {
        cv::Rect(cv::Point{ 0,   0}, cv::Size{80, 120}),
        cv::Rect(cv::Point{50, 100}, cv::Size{96, 160}),
    };

    std::vector<cv::Mat> gapi_age, gapi_gender;

    // Load & run IE network
    namespace IE = InferenceEngine;
    std::vector<cv::Mat> ie_age, ie_gender;
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
        auto frame_blob = cv::gapi::ie::util::to_ie(in_mat);

        for (auto &&rc : rois) {
            const auto ie_rc = IE::ROI {
                  0u
                , static_cast<std::size_t>(rc.x)
                , static_cast<std::size_t>(rc.y)
                , static_cast<std::size_t>(rc.width)
                , static_cast<std::size_t>(rc.height)
            };
            infer_request.SetBlob("data", IE::make_shared_blob(frame_blob, ie_rc));
            infer_request.Infer();

            using namespace cv::gapi::ie::util;
            ie_age.push_back(to_ocv(infer_request.GetBlob("age_conv3")).clone());
            ie_gender.push_back(to_ocv(infer_request.GetBlob("prob")).clone());
        }
    }

    // Configure & run G-API
    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    cv::GArray<cv::Rect> rr;
    cv::GMat in;
    cv::GArray<cv::GMat> age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(rr, in);
    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<AgeGender> {
        topology_path, weights_path, "CPU"
    }.cfgOutputLayers({ "age_conv3", "prob" });
    comp.apply(cv::gin(in_mat, rois), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    ASSERT_EQ(2u, ie_age.size()   );
    ASSERT_EQ(2u, ie_gender.size());
    ASSERT_EQ(2u, gapi_age.size()   );
    ASSERT_EQ(2u, gapi_gender.size());

    normAssert(ie_age   [0], gapi_age   [0], "0: Test age output");
    normAssert(ie_gender[0], gapi_gender[0], "0: Test gender output");
    normAssert(ie_age   [1], gapi_age   [1], "1: Test age output");
    normAssert(ie_gender[1], gapi_gender[1], "1: Test gender output");
}


} // namespace opencv_test

#endif //  HAVE_INF_ENGINE
