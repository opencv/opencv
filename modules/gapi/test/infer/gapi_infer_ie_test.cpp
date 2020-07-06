// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019-2020 Intel Corporation

////////////////////////////////////////////////////////////////////////////////
// FIXME: Suppress deprecation warnings for OpenVINO < 2021
#ifdef _MSC_VER
#pragma warning(disable: 4996)  // was declared deprecated
#endif
////////////////////////////////////////////////////////////////////////////////

#include "../test_precomp.hpp"

#ifdef HAVE_INF_ENGINE

#include <stdexcept>

#include <inference_engine.hpp>

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

std::vector<std::string> modelPathByName(const std::string &model_name) {
    // Handle OMZ model layout changes among OpenVINO versions here
    static const std::unordered_multimap<std::string, std::string> map = {
        {"age-gender-recognition-retail-0013",
         "2020.3.0/intel/age-gender-recognition-retail-0013/FP32"},
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
    ii->setLayout(IE::Layout::NHWC);
    ii->getPreProcess().setResizeAlgorithm(IE::RESIZE_BILINEAR);
}

#if INF_ENGINE_RELEASE < 2020000000  // < 2020.1
IE::CNNNetwork readNetwork(const std::string& topology_path,
                           const std::string& weights_path) {
    IE::CNNNetReader reader;
    reader.ReadNetwork(topology_path);
    reader.ReadWeights(weights_path);
    return reader.getNetwork();
}

IE::CNNNetwork readNetwork(IE::Core&,
                           const std::string& topology_path,
                           const std::string& weights_path) {
    return readNetwork(topology_path, weights_path);
}
#else // >= 2020.1
inline IE::CNNNetwork readNetwork(      IE::Core&    core,
                                  const std::string& topology_path,
                                  const std::string& weights_path) {
    return core.ReadNetwork(topology_path, weights_path);
}
#endif // INF_ENGINE_RELEASE < 2020000000
#if INF_ENGINE_RELEASE < 2019020000  // < 2019.R2
inline IE::InferencePlugin getPlugin(const std::string& device) {
    return IE::PluginDispatcher().getPluginByDevice(device);
}

IE::CNNNetwork readNetwork(IE::InferencePlugin&,
                           const std::string& topology_path,
                           const std::string& weights_path) {
    return readNetwork(topology_path, weights_path);
}

inline IE::ExecutableNetwork loadNetwork(      IE::InferencePlugin& plugin,
                                         const IE::CNNNetwork&      net,
                                         const std::string&) {
    return plugin.LoadNetwork(net, {});
}
#else // >= 2019.R2
inline IE::Core getPlugin(const std::string&) {
    return IE::Core();
}

inline IE::ExecutableNetwork loadNetwork(      IE::Core&       core,
                                         const IE::CNNNetwork& net,
                                         const std::string&    device) {
    return core.LoadNetwork(net, device);
}
#endif // INF_ENGINE_RELEASE < 2019020000
} // anonymous namespace

// TODO: Probably DNN/IE part can be further parametrized with a template
// NOTE: here ".." is used to leave the default "gapi/" search scope
TEST(TestAgeGenderIE, InferBasicTensor)
{
    initDLDTDataPath();

    std::string topology_path, weights_path;
    std::tie(topology_path, weights_path) = findModel("age-gender-recognition-retail-0013");

    // Load IE network, initialize input data using that.
    namespace IE = InferenceEngine;
    cv::Mat in_mat;
    cv::Mat gapi_age, gapi_gender;

    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin        = getPlugin("CPU");
        auto net           = readNetwork(plugin, topology_path, weights_path);
        // setNetParameters(net);
        auto plugin_net    = loadNetwork(plugin, net, "CPU");
        auto infer_request = plugin_net.CreateInferRequest();

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

    std::string topology_path, weights_path;
    std::tie(topology_path, weights_path) = findModel("age-gender-recognition-retail-0013");

    // FIXME: Ideally it should be an image from disk
    // cv::Mat in_mat = cv::imread(findDataFile("grace_hopper_227.png"));
    cv::Mat in_mat(cv::Size(320, 240), CV_8UC3);
    cv::randu(in_mat, 0, 255);

    cv::Mat gapi_age, gapi_gender;

    // Load & run IE network
    namespace IE = InferenceEngine;
    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin        = getPlugin("CPU");
        auto net           = readNetwork(plugin, topology_path, weights_path);
        setNetParameters(net);
        auto plugin_net    = loadNetwork(plugin, net, "CPU");
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

struct ROIList: public ::testing::Test {
    std::string m_model_path;
    std::string m_weights_path;

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
        std::tie(m_model_path, m_weights_path) = findModel("age-gender-recognition-retail-0013");

        // FIXME: it must be cv::imread(findDataFile("../dnn/grace_hopper_227.png", false));
        m_in_mat = cv::Mat(cv::Size(320, 240), CV_8UC3);
        cv::randu(m_in_mat, 0, 255);

        // both ROIs point to the same face, with a slightly changed geometry
        m_roi_list = {
            cv::Rect(cv::Point{64, 60}, cv::Size{ 96,  96}),
            cv::Rect(cv::Point{50, 32}, cv::Size{128, 160}),
        };

        // Load & run IE network
        namespace IE = InferenceEngine;
        {
            auto plugin        = getPlugin("CPU");
            auto net           = readNetwork(plugin, topology_path, weights_path);
            setNetParameters(net);
            auto plugin_net    = loadNetwork(plugin, net, "CPU");
            auto infer_request = plugin_net.CreateInferRequest();
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
        m_model_path, m_weights_path, "CPU"
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
        m_model_path, m_weights_path, "CPU"
    }.cfgOutputLayers({ "age_conv3", "prob" });
    comp.apply(cv::gin(m_in_mat, m_roi_list),
               cv::gout(m_out_gapi_ages, m_out_gapi_genders),
               cv::compile_args(cv::gapi::networks(pp)));
    validate();
}

} // namespace opencv_test

#endif //  HAVE_INF_ENGINE
