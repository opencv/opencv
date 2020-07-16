// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019-2020 Intel Corporation

#include "../test_precomp.hpp"

#ifdef HAVE_INF_ENGINE

#include <stdexcept>

#include <inference_engine.hpp>

#include <ade/util/iota_range.hpp>

#include <opencv2/gapi/infer/ie.hpp>

#include "backends/ie/util.hpp"
#include "backends/ie/giebackend/giewrapper.hpp"

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

std::vector<std::string> modelPathByName(const std::string &model_name) {
    // Handle OMZ model layout changes among OpenVINO versions here
    static const std::unordered_multimap<std::string, std::string> map = {
#if INF_ENGINE_RELEASE >= 2019040000  // >= 2019.R4
        {"age-gender-recognition-retail-0013",
         "2020.3.0/intel/age-gender-recognition-retail-0013/FP32"},
#endif // INF_ENGINE_RELEASE >= 2019040000
        {"age-gender-recognition-retail-0013",
         "Retail/object_attributes/age_gender/dldt"},
    };
    const auto range = map.equal_range(model_name);
    std::vector<std::string> result;
    for (auto it = range.first; it != range.second; ++it) {
        result.emplace_back(it->second);
    }
    return result;
}

std::tuple<std::string, std::string> findModel(const std::string &model_name) {
    const auto candidates = modelPathByName(model_name);
    CV_Assert(!candidates.empty() && "No model path candidates found at all");

    for (auto &&path : candidates) {
        std::string model_xml, model_bin;
        try {
            model_xml = findDataFile(path + "/" + model_name + ".xml", false);
            model_bin = findDataFile(path + "/" + model_name + ".bin", false);
            // Return the first file which actually works
            return std::make_tuple(model_xml, model_bin);
        } catch (SkipTestException&) {
            // This is quite ugly but it is a way for OpenCV to let us know
            // this file wasn't found.
            continue;
        }
    }

    // Default behavior if reached here.
    throw SkipTestException("Files for " + model_name + " were not found");
}

namespace IE = InferenceEngine;

void setNetParameters(IE::CNNNetwork& net) {
    auto &ii = net.getInputsInfo().at("data");
    ii->setPrecision(IE::Precision::U8);
    ii->getPreProcess().setResizeAlgorithm(IE::RESIZE_BILINEAR);
}
} // anonymous namespace

// TODO: Probably DNN/IE part can be further parametrized with a template
// NOTE: here ".." is used to leave the default "gapi/" search scope
TEST(TestAgeGenderIE, InferBasicTensor)
{
    initDLDTDataPath();

    cv::gapi::ie::detail::ParamDesc params;
    std::tie(params.model_path, params.weights_path) = findModel("age-gender-recognition-retail-0013");
    params.device_id = "CPU";

    // Load IE network, initialize input data using that.
    cv::Mat in_mat;
    cv::Mat gapi_age, gapi_gender;

    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin        = cv::gimpl::ie::wrap::getPlugin(params);
        auto net           = cv::gimpl::ie::wrap::readNetwork(params);
        auto this_network  = cv::gimpl::ie::wrap::loadNetwork(plugin, net, params);
        auto infer_request = this_network.CreateInferRequest();

        const auto &iedims = net.getInputsInfo().begin()->second->getTensorDesc().getDims();
              auto  cvdims = cv::gapi::ie::util::to_ocv(iedims);
        in_mat.create(cvdims, CV_32F);
        cv::randu(in_mat, -1, 1);

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
        params.model_path, params.weights_path, params.device_id
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

    cv::gapi::ie::detail::ParamDesc params;
    std::tie(params.model_path, params.weights_path) = findModel("age-gender-recognition-retail-0013");
    params.device_id = "CPU";

    // FIXME: Ideally it should be an image from disk
    // cv::Mat in_mat = cv::imread(findDataFile("grace_hopper_227.png"));
    cv::Mat in_mat(cv::Size(320, 240), CV_8UC3);
    cv::randu(in_mat, 0, 255);

    cv::Mat gapi_age, gapi_gender;

    // Load & run IE network
    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin        = cv::gimpl::ie::wrap::getPlugin(params);
        auto net           = cv::gimpl::ie::wrap::readNetwork(params);
        setNetParameters(net);
        auto this_network  = cv::gimpl::ie::wrap::loadNetwork(plugin, net, params);
        auto infer_request = this_network.CreateInferRequest();
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
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });
    comp.apply(cv::gin(in_mat), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(cv::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(cv::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
}

struct ROIList: public ::testing::Test {
    cv::gapi::ie::detail::ParamDesc params;

    cv::Mat m_in_mat;
    std::vector<cv::Rect> m_roi_list;

    std::vector<cv::Mat> m_out_ie_ages;
    std::vector<cv::Mat> m_out_ie_genders;

    std::vector<cv::Mat> m_out_gapi_ages;
    std::vector<cv::Mat> m_out_gapi_genders;

    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    ROIList() {
        initDLDTDataPath();
        std::tie(params.model_path, params.weights_path) = findModel("age-gender-recognition-retail-0013");
        params.device_id = "CPU";

        // FIXME: it must be cv::imread(findDataFile("../dnn/grace_hopper_227.png", false));
        m_in_mat = cv::Mat(cv::Size(320, 240), CV_8UC3);
        cv::randu(m_in_mat, 0, 255);

        // both ROIs point to the same face, with a slightly changed geometry
        m_roi_list = {
            cv::Rect(cv::Point{64, 60}, cv::Size{ 96,  96}),
            cv::Rect(cv::Point{50, 32}, cv::Size{128, 160}),
        };

        // Load & run IE network
        {
            auto plugin        = cv::gimpl::ie::wrap::getPlugin(params);
            auto net           = cv::gimpl::ie::wrap::readNetwork(params);
            setNetParameters(net);
            auto this_network  = cv::gimpl::ie::wrap::loadNetwork(plugin, net, params);
            auto infer_request = this_network.CreateInferRequest();
            auto frame_blob = cv::gapi::ie::util::to_ie(m_in_mat);

            for (auto &&rc : m_roi_list) {
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
                m_out_ie_ages.push_back(to_ocv(infer_request.GetBlob("age_conv3")).clone());
                m_out_ie_genders.push_back(to_ocv(infer_request.GetBlob("prob")).clone());
            }
        } // namespace IE = ..
    } // ROIList()

    void validate() {
        // Validate with IE itself (avoid DNN module dependency here)
        ASSERT_EQ(2u, m_out_ie_ages.size());
        ASSERT_EQ(2u, m_out_ie_genders.size());
        ASSERT_EQ(2u, m_out_gapi_ages.size());
        ASSERT_EQ(2u, m_out_gapi_genders.size());

        normAssert(m_out_ie_ages   [0], m_out_gapi_ages   [0], "0: Test age output");
        normAssert(m_out_ie_genders[0], m_out_gapi_genders[0], "0: Test gender output");
        normAssert(m_out_ie_ages   [1], m_out_gapi_ages   [1], "1: Test age output");
        normAssert(m_out_ie_genders[1], m_out_gapi_genders[1], "1: Test gender output");
    }
}; // ROIList

TEST_F(ROIList, TestInfer)
{
    cv::GArray<cv::Rect> rr;
    cv::GMat in;
    cv::GArray<cv::GMat> age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(rr, in);
    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });
    comp.apply(cv::gin(m_in_mat, m_roi_list),
               cv::gout(m_out_gapi_ages, m_out_gapi_genders),
               cv::compile_args(cv::gapi::networks(pp)));
    validate();
}

TEST_F(ROIList, TestInfer2)
{
    cv::GArray<cv::Rect> rr;
    cv::GMat in;
    cv::GArray<cv::GMat> age, gender;
    std::tie(age, gender) = cv::gapi::infer2<AgeGender>(in, rr);
    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });
    comp.apply(cv::gin(m_in_mat, m_roi_list),
               cv::gout(m_out_gapi_ages, m_out_gapi_genders),
               cv::compile_args(cv::gapi::networks(pp)));
    validate();
}

TEST(DISABLED_TestTwoIENNPipeline, InferBasicImage)
{
    initDLDTDataPath();

    cv::gapi::ie::detail::ParamDesc AGparams;
    std::tie(AGparams.model_path, AGparams.weights_path) = findModel("age-gender-recognition-retail-0013");
    AGparams.device_id = "MYRIAD";

    // FIXME: Ideally it should be an image from disk
    // cv::Mat in_mat = cv::imread(findDataFile("grace_hopper_227.png"));
    cv::Mat in_mat(cv::Size(320, 240), CV_8UC3);
    cv::randu(in_mat, 0, 255);

    cv::Mat gapi_age1, gapi_gender1, gapi_age2, gapi_gender2;

    // Load & run IE network
    IE::Blob::Ptr ie_age1, ie_gender1, ie_age2, ie_gender2;
    {
        auto AGplugin1         = cv::gimpl::ie::wrap::getPlugin(AGparams);
        auto AGnet1            = cv::gimpl::ie::wrap::readNetwork(AGparams);
        setNetParameters(AGnet1);
        auto AGplugin_network1 = cv::gimpl::ie::wrap::loadNetwork(AGplugin1, AGnet1, AGparams);
        auto AGinfer_request1  = AGplugin_network1.CreateInferRequest();
        AGinfer_request1.SetBlob("data", cv::gapi::ie::util::to_ie(in_mat));
        AGinfer_request1.Infer();
        ie_age1    = AGinfer_request1.GetBlob("age_conv3");
        ie_gender1 = AGinfer_request1.GetBlob("prob");

        auto AGplugin2         = cv::gimpl::ie::wrap::getPlugin(AGparams);
        auto AGnet2            = cv::gimpl::ie::wrap::readNetwork(AGparams);
        setNetParameters(AGnet2);
        auto AGplugin_network2 = cv::gimpl::ie::wrap::loadNetwork(AGplugin2, AGnet2, AGparams);
        auto AGinfer_request2     = AGplugin_network2.CreateInferRequest();
        AGinfer_request2.SetBlob("data", cv::gapi::ie::util::to_ie(in_mat));
        AGinfer_request2.Infer();
        ie_age2    = AGinfer_request2.GetBlob("age_conv3");
        ie_gender2 = AGinfer_request2.GetBlob("prob");
    }

    // Configure & run G-API
    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender1, <AGInfo(cv::GMat)>,   "test-age-gender1");
    G_API_NET(AgeGender2, <AGInfo(cv::GMat)>,   "test-age-gender2");
    cv::GMat in;
    cv::GMat age1, gender1;
    std::tie(age1, gender1) = cv::gapi::infer<AgeGender1>(in);

    cv::GMat age2, gender2;
    // FIXME: "Multi-node inference is not supported!", workarounded 'till enabling proper tools
    std::tie(age2, gender2) = cv::gapi::infer<AgeGender2>(cv::gapi::copy(in));
    cv::GComputation comp(cv::GIn(in), cv::GOut(age1, gender1, age2, gender2));

    auto age_net1 = cv::gapi::ie::Params<AgeGender1> {
        AGparams.model_path, AGparams.weights_path, AGparams.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });
    auto age_net2 = cv::gapi::ie::Params<AgeGender2> {
        AGparams.model_path, AGparams.weights_path, AGparams.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });

    comp.apply(cv::gin(in_mat), cv::gout(gapi_age1, gapi_gender1, gapi_age2, gapi_gender2),
               cv::compile_args(cv::gapi::networks(age_net1, age_net2)));

    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(cv::gapi::ie::util::to_ocv(ie_age1),    gapi_age1,    "Test age output 1");
    normAssert(cv::gapi::ie::util::to_ocv(ie_gender1), gapi_gender1, "Test gender output 1");
    normAssert(cv::gapi::ie::util::to_ocv(ie_age2),    gapi_age2,    "Test age output 2");
    normAssert(cv::gapi::ie::util::to_ocv(ie_gender2), gapi_gender2, "Test gender output 2");
}

} // namespace opencv_test

#endif //  HAVE_INF_ENGINE
