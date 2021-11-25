// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019-2021 Intel Corporation

#include "../test_precomp.hpp"

#ifdef HAVE_INF_ENGINE

#include <stdexcept>
#include <mutex>
#include <condition_variable>

#include <inference_engine.hpp>

#include <ade/util/iota_range.hpp>

#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/streaming/cap.hpp>

#include "backends/ie/util.hpp"
#include "backends/ie/giebackend/giewrapper.hpp"

#ifdef HAVE_NGRAPH
#if defined(__clang__)  // clang or MSVC clang
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <ngraph/ngraph.hpp>
#endif

namespace opencv_test
{
namespace {

class TestMediaBGR final: public cv::MediaFrame::IAdapter {
    cv::Mat m_mat;
    using Cb = cv::MediaFrame::View::Callback;
    Cb m_cb;

public:
    explicit TestMediaBGR(cv::Mat m, Cb cb = [](){})
        : m_mat(m), m_cb(cb) {
    }
    cv::GFrameDesc meta() const override {
        return cv::GFrameDesc{cv::MediaFormat::BGR, cv::Size(m_mat.cols, m_mat.rows)};
    }
    cv::MediaFrame::View access(cv::MediaFrame::Access) override {
        cv::MediaFrame::View::Ptrs pp = { m_mat.ptr(), nullptr, nullptr, nullptr };
        cv::MediaFrame::View::Strides ss = { m_mat.step, 0u, 0u, 0u };
        return cv::MediaFrame::View(std::move(pp), std::move(ss), Cb{m_cb});
    }
    cv::util::any blobParams() const override {
        return std::make_pair<InferenceEngine::TensorDesc,
                              InferenceEngine::ParamMap>({IE::Precision::U8,
                                                          {1, 3, 300, 300},
                                                          IE::Layout::NCHW},
                                                         {{"HELLO", 42},
                                                          {"COLOR_FORMAT",
                                                           InferenceEngine::ColorFormat::NV12}});
    }
};

class TestMediaNV12 final: public cv::MediaFrame::IAdapter {
    cv::Mat m_y;
    cv::Mat m_uv;
public:
    TestMediaNV12(cv::Mat y, cv::Mat uv) : m_y(y), m_uv(uv) {
    }
    cv::GFrameDesc meta() const override {
        return cv::GFrameDesc{cv::MediaFormat::NV12, cv::Size(m_y.cols, m_y.rows)};
    }
    cv::MediaFrame::View access(cv::MediaFrame::Access) override {
        cv::MediaFrame::View::Ptrs pp = {
            m_y.ptr(), m_uv.ptr(), nullptr, nullptr
        };
        cv::MediaFrame::View::Strides ss = {
            m_y.step, m_uv.step, 0u, 0u
        };
        return cv::MediaFrame::View(std::move(pp), std::move(ss));
    }
};

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

namespace IE = InferenceEngine;

void setNetParameters(IE::CNNNetwork& net, bool is_nv12 = false) {
    auto ii = net.getInputsInfo().at("data");
    ii->setPrecision(IE::Precision::U8);
    ii->getPreProcess().setResizeAlgorithm(IE::RESIZE_BILINEAR);
    if (is_nv12) {
        ii->getPreProcess().setColorFormat(IE::ColorFormat::NV12);
    }
}

bool checkDeviceIsAvailable(const std::string& device) {
    const static auto available_devices = [&](){
        auto devices = cv::gimpl::ie::wrap::getCore().GetAvailableDevices();
        return std::unordered_set<std::string>{devices.begin(), devices.end()};
    }();
    return available_devices.find(device) != available_devices.end();
}

void skipIfDeviceNotAvailable(const std::string& device) {
    if (!checkDeviceIsAvailable(device)) {
        throw SkipTestException("Device: " + device + " isn't available!");
    }
}

void compileBlob(const cv::gapi::ie::detail::ParamDesc& params,
                 const std::string&                     output,
                 const IE::Precision&                   ip) {
    auto plugin = cv::gimpl::ie::wrap::getPlugin(params);
    auto net    = cv::gimpl::ie::wrap::readNetwork(params);
    for (auto&& ii : net.getInputsInfo()) {
        ii.second->setPrecision(ip);
    }
    auto this_network = cv::gimpl::ie::wrap::loadNetwork(plugin, net, params);
    std::ofstream out_file{output, std::ios::out | std::ios::binary};
    GAPI_Assert(out_file.is_open());
    this_network.Export(out_file);
}

std::string compileAgeGenderBlob(const std::string& device) {
    const static std::string blob_path = [&](){
        cv::gapi::ie::detail::ParamDesc params;
        const std::string model_name = "age-gender-recognition-retail-0013";
        const std::string output  = model_name + ".blob";
        params.model_path   = findDataFile(SUBDIR + model_name + ".xml");
        params.weights_path = findDataFile(SUBDIR + model_name + ".bin");
        params.device_id    = device;
        compileBlob(params, output, IE::Precision::U8);
        return output;
    }();
    return blob_path;
}

} // anonymous namespace

// TODO: Probably DNN/IE part can be further parametrized with a template
// NOTE: here ".." is used to leave the default "gapi/" search scope
TEST(TestAgeGenderIE, InferBasicTensor)
{
    initDLDTDataPath();

    cv::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
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
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
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

struct InferWithReshape: public ::testing::Test {
    cv::gapi::ie::detail::ParamDesc params;
    cv::Mat m_in_mat;
    std::vector<cv::Rect> m_roi_list;
    std::vector<size_t> reshape_dims;
    std::vector<cv::Mat> m_out_ie_ages;
    std::vector<cv::Mat> m_out_ie_genders;
    std::vector<cv::Mat> m_out_gapi_ages;
    std::vector<cv::Mat> m_out_gapi_genders;
    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    InferenceEngine::CNNNetwork net;
    InferenceEngine::Core plugin;

    InferWithReshape() {
        // FIXME: it must be cv::imread(findDataFile("../dnn/grace_hopper_227.png", false));
        m_in_mat = cv::Mat(cv::Size(320, 240), CV_8UC3);
        cv::randu(m_in_mat, 0, 255);

        m_out_gapi_ages.resize(1);
        m_out_gapi_genders.resize(1);

        // both ROIs point to the same face, with a slightly changed geometry
        m_roi_list = {
            cv::Rect(cv::Point{64, 60}, cv::Size{ 96,  96}),
            cv::Rect(cv::Point{50, 32}, cv::Size{128, 160}),
        };

        // New dimensions for "data" input
        reshape_dims = {1, 3, 70, 70};

        initDLDTDataPath();
        params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
        params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");

        params.device_id = "CPU";

        plugin = cv::gimpl::ie::wrap::getPlugin(params);
        net    = cv::gimpl::ie::wrap::readNetwork(params);
        setNetParameters(net);
        net.reshape({{"data", reshape_dims}});
    }

    void inferROIs(IE::Blob::Ptr blob) {
        auto this_network  = cv::gimpl::ie::wrap::loadNetwork(plugin, net, params);
        auto infer_request = this_network.CreateInferRequest();
        for (auto &&rc : m_roi_list) {
            const auto ie_rc = IE::ROI {
                0u
                , static_cast<std::size_t>(rc.x)
                , static_cast<std::size_t>(rc.y)
                , static_cast<std::size_t>(rc.width)
                , static_cast<std::size_t>(rc.height)
            };
            infer_request.SetBlob("data", IE::make_shared_blob(blob, ie_rc));
            infer_request.Infer();
            using namespace cv::gapi::ie::util;
            m_out_ie_ages.push_back(to_ocv(infer_request.GetBlob("age_conv3")).clone());
            m_out_ie_genders.push_back(to_ocv(infer_request.GetBlob("prob")).clone());
        }
    }

    void infer(cv::Mat& in, const bool with_roi = false) {
        if (!with_roi) {
            auto this_network  = cv::gimpl::ie::wrap::loadNetwork(plugin, net, params);
            auto infer_request = this_network.CreateInferRequest();
            infer_request.SetBlob("data", cv::gapi::ie::util::to_ie(in));
            infer_request.Infer();
            using namespace cv::gapi::ie::util;
            m_out_ie_ages.push_back(to_ocv(infer_request.GetBlob("age_conv3")).clone());
            m_out_ie_genders.push_back(to_ocv(infer_request.GetBlob("prob")).clone());
        } else {
            auto frame_blob = cv::gapi::ie::util::to_ie(in);
            inferROIs(frame_blob);
        }
    }

    void validate() {
        // Validate with IE itself (avoid DNN module dependency here)
        GAPI_Assert(!m_out_gapi_ages.empty());
        ASSERT_EQ(m_out_gapi_genders.size(), m_out_gapi_ages.size());
        ASSERT_EQ(m_out_gapi_ages.size(), m_out_ie_ages.size());
        ASSERT_EQ(m_out_gapi_genders.size(), m_out_ie_genders.size());

        const size_t size = m_out_gapi_ages.size();
        for (size_t i = 0; i < size; ++i) {
            normAssert(m_out_ie_ages   [i], m_out_gapi_ages   [i], "Test age output");
            normAssert(m_out_ie_genders[i], m_out_gapi_genders[i], "Test gender output");
        }
    }
}; // InferWithReshape

struct InferWithReshapeNV12: public InferWithReshape {
    cv::Mat m_in_uv;
    cv::Mat m_in_y;
    void SetUp() {
        cv::Size sz{320, 240};
        m_in_y = cv::Mat{sz, CV_8UC1};
        cv::randu(m_in_y, 0, 255);
        m_in_uv = cv::Mat{sz / 2, CV_8UC2};
        cv::randu(m_in_uv, 0, 255);
        setNetParameters(net, true);
        net.reshape({{"data", reshape_dims}});
        auto frame_blob = cv::gapi::ie::util::to_ie(m_in_y, m_in_uv);
        inferROIs(frame_blob);
    }
};

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

    void SetUp() {
        initDLDTDataPath();
        params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
        params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
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

struct ROIListNV12: public ::testing::Test {
    cv::gapi::ie::detail::ParamDesc params;

    cv::Mat m_in_uv;
    cv::Mat m_in_y;
    std::vector<cv::Rect> m_roi_list;

    std::vector<cv::Mat> m_out_ie_ages;
    std::vector<cv::Mat> m_out_ie_genders;

    std::vector<cv::Mat> m_out_gapi_ages;
    std::vector<cv::Mat> m_out_gapi_genders;

    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    void SetUp() {
        initDLDTDataPath();
        params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
        params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
        params.device_id = "CPU";

        cv::Size sz{320, 240};
        m_in_y = cv::Mat{sz, CV_8UC1};
        cv::randu(m_in_y, 0, 255);
        m_in_uv = cv::Mat{sz / 2, CV_8UC2};
        cv::randu(m_in_uv, 0, 255);

        // both ROIs point to the same face, with a slightly changed geometry
        m_roi_list = {
            cv::Rect(cv::Point{64, 60}, cv::Size{ 96,  96}),
            cv::Rect(cv::Point{50, 32}, cv::Size{128, 160}),
        };

        // Load & run IE network
        {
            auto plugin        = cv::gimpl::ie::wrap::getPlugin(params);
            auto net           = cv::gimpl::ie::wrap::readNetwork(params);
            setNetParameters(net, true);
            auto this_network  = cv::gimpl::ie::wrap::loadNetwork(plugin, net, params);
            auto infer_request = this_network.CreateInferRequest();
            auto frame_blob = cv::gapi::ie::util::to_ie(m_in_y, m_in_uv);

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
};

struct SingleROI: public ::testing::Test {
    cv::gapi::ie::detail::ParamDesc params;

    cv::Mat m_in_mat;
    cv::Rect m_roi;

    cv::Mat m_out_gapi_age;
    cv::Mat m_out_gapi_gender;

    cv::Mat m_out_ie_age;
    cv::Mat m_out_ie_gender;

    void SetUp() {
        initDLDTDataPath();
        params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
        params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
        params.device_id = "CPU";

        // FIXME: it must be cv::imread(findDataFile("../dnn/grace_hopper_227.png", false));
        m_in_mat = cv::Mat(cv::Size(320, 240), CV_8UC3);
        cv::randu(m_in_mat, 0, 255);

        m_roi = cv::Rect(cv::Point{64, 60}, cv::Size{96, 96});

        // Load & run IE network
        IE::Blob::Ptr ie_age, ie_gender;
        {
            auto plugin        = cv::gimpl::ie::wrap::getPlugin(params);
            auto net           = cv::gimpl::ie::wrap::readNetwork(params);
            setNetParameters(net);
            auto this_network  = cv::gimpl::ie::wrap::loadNetwork(plugin, net, params);
            auto infer_request = this_network.CreateInferRequest();

            const auto ie_rc = IE::ROI {
                0u
                , static_cast<std::size_t>(m_roi.x)
                , static_cast<std::size_t>(m_roi.y)
                , static_cast<std::size_t>(m_roi.width)
                , static_cast<std::size_t>(m_roi.height)
            };

            IE::Blob::Ptr roi_blob = IE::make_shared_blob(cv::gapi::ie::util::to_ie(m_in_mat), ie_rc);
            infer_request.SetBlob("data", roi_blob);
            infer_request.Infer();

            using namespace cv::gapi::ie::util;
            m_out_ie_age    = to_ocv(infer_request.GetBlob("age_conv3")).clone();
            m_out_ie_gender = to_ocv(infer_request.GetBlob("prob")).clone();
        }
    }

    void validate() {
        // Validate with IE itself (avoid DNN module dependency here)
        normAssert(m_out_ie_age   , m_out_gapi_age   , "Test age output");
        normAssert(m_out_ie_gender, m_out_gapi_gender, "Test gender output");
    }
};

struct SingleROINV12: public ::testing::Test {
    cv::gapi::ie::detail::ParamDesc params;

    cv::Mat m_in_y;
    cv::Mat m_in_uv;
    cv::Rect m_roi;

    cv::Mat m_out_gapi_age;
    cv::Mat m_out_gapi_gender;

    cv::Mat m_out_ie_age;
    cv::Mat m_out_ie_gender;

    void SetUp() {
        initDLDTDataPath();
        params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
        params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
        params.device_id = "CPU";

        cv::Size sz{320, 240};
        m_in_y = cv::Mat{sz, CV_8UC1};
        cv::randu(m_in_y, 0, 255);
        m_in_uv = cv::Mat{sz / 2, CV_8UC2};
        cv::randu(m_in_uv, 0, 255);

        m_roi = cv::Rect(cv::Point{64, 60}, cv::Size{96, 96});

        // Load & run IE network
        IE::Blob::Ptr ie_age, ie_gender;
        {
            auto plugin        = cv::gimpl::ie::wrap::getPlugin(params);
            auto net           = cv::gimpl::ie::wrap::readNetwork(params);
            setNetParameters(net, /* NV12 */ true);
            auto this_network  = cv::gimpl::ie::wrap::loadNetwork(plugin, net, params);
            auto infer_request = this_network.CreateInferRequest();
            auto blob = cv::gapi::ie::util::to_ie(m_in_y, m_in_uv);

            const auto ie_rc = IE::ROI {
                0u
                , static_cast<std::size_t>(m_roi.x)
                , static_cast<std::size_t>(m_roi.y)
                , static_cast<std::size_t>(m_roi.width)
                , static_cast<std::size_t>(m_roi.height)
            };

            IE::Blob::Ptr roi_blob = IE::make_shared_blob(blob, ie_rc);
            infer_request.SetBlob("data", roi_blob);
            infer_request.Infer();

            using namespace cv::gapi::ie::util;
            m_out_ie_age    = to_ocv(infer_request.GetBlob("age_conv3")).clone();
            m_out_ie_gender = to_ocv(infer_request.GetBlob("prob")).clone();
        }
    }

    void validate() {
        // Validate with IE itself (avoid DNN module dependency here)
        normAssert(m_out_ie_age   , m_out_gapi_age   , "Test age output");
        normAssert(m_out_ie_gender, m_out_gapi_gender, "Test gender output");
    }
};

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
    AGparams.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
    AGparams.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
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

TEST(TestAgeGenderIE, GenericInfer)
{
    initDLDTDataPath();

    cv::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    params.device_id = "CPU";

    cv::Mat in_mat(cv::Size(320, 240), CV_8UC3);
    cv::randu(in_mat, 0, 255);

    cv::Mat gapi_age, gapi_gender;

    // Load & run IE network
    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin = cv::gimpl::ie::wrap::getPlugin(params);
        auto net    = cv::gimpl::ie::wrap::readNetwork(params);
        setNetParameters(net);
        auto this_network  = cv::gimpl::ie::wrap::loadNetwork(plugin, net, params);
        auto infer_request = this_network.CreateInferRequest();
        infer_request.SetBlob("data", cv::gapi::ie::util::to_ie(in_mat));
        infer_request.Infer();
        ie_age    = infer_request.GetBlob("age_conv3");
        ie_gender = infer_request.GetBlob("prob");
    }

    // Configure & run G-API
    cv::GMat in;
    GInferInputs inputs;
    inputs["data"] = in;

    auto outputs = cv::gapi::infer<cv::gapi::Generic>("age-gender-generic", inputs);

    auto age    = outputs.at("age_conv3");
    auto gender = outputs.at("prob");

    cv::GComputation comp(cv::GIn(in), cv::GOut(age, gender));

    cv::gapi::ie::Params<cv::gapi::Generic> pp{
        "age-gender-generic", params.model_path, params.weights_path, params.device_id};

    comp.apply(cv::gin(in_mat), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(cv::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(cv::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
}

TEST(TestAgeGenderIE, InvalidConfigGeneric)
{
    initDLDTDataPath();

    std::string model_path   = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    std::string weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    std::string device_id    = "CPU";

    // Configure & run G-API
    cv::GMat in;
    GInferInputs inputs;
    inputs["data"] = in;

    auto outputs = cv::gapi::infer<cv::gapi::Generic>("age-gender-generic", inputs);
    auto age     = outputs.at("age_conv3");
    auto gender  = outputs.at("prob");
    cv::GComputation comp(cv::GIn(in), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<cv::gapi::Generic>{
        "age-gender-generic", model_path, weights_path, device_id
    }.pluginConfig({{"unsupported_config", "some_value"}});

    EXPECT_ANY_THROW(comp.compile(cv::GMatDesc{CV_8U,3,cv::Size{320, 240}},
                     cv::compile_args(cv::gapi::networks(pp))));
}

TEST(TestAgeGenderIE, CPUConfigGeneric)
{
    initDLDTDataPath();

    std::string model_path   = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    std::string weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    std::string device_id    = "CPU";

    // Configure & run G-API
    cv::GMat in;
    GInferInputs inputs;
    inputs["data"] = in;

    auto outputs = cv::gapi::infer<cv::gapi::Generic>("age-gender-generic", inputs);
    auto age     = outputs.at("age_conv3");
    auto gender  = outputs.at("prob");
    cv::GComputation comp(cv::GIn(in), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<cv::gapi::Generic> {
        "age-gender-generic", model_path, weights_path, device_id
    }.pluginConfig({{IE::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS,
                     IE::PluginConfigParams::CPU_THROUGHPUT_NUMA}});

    EXPECT_NO_THROW(comp.compile(cv::GMatDesc{CV_8U,3,cv::Size{320, 240}},
                    cv::compile_args(cv::gapi::networks(pp))));
}

TEST(TestAgeGenderIE, InvalidConfig)
{
    initDLDTDataPath();

    std::string model_path   = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    std::string weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    std::string device_id    = "CPU";

    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    cv::GMat in;
    cv::GMat age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<AgeGender> {
        model_path, weights_path, device_id
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .pluginConfig({{"unsupported_config", "some_value"}});

    EXPECT_ANY_THROW(comp.compile(cv::GMatDesc{CV_8U,3,cv::Size{320, 240}},
                     cv::compile_args(cv::gapi::networks(pp))));
}

TEST(TestAgeGenderIE, CPUConfig)
{
    initDLDTDataPath();

    std::string model_path   = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    std::string weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    std::string device_id    = "CPU";

    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    cv::GMat in;
    cv::GMat age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<AgeGender> {
        model_path, weights_path, device_id
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .pluginConfig({{IE::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS,
                     IE::PluginConfigParams::CPU_THROUGHPUT_NUMA}});

    EXPECT_NO_THROW(comp.compile(cv::GMatDesc{CV_8U,3,cv::Size{320, 240}},
                    cv::compile_args(cv::gapi::networks(pp))));
}

TEST_F(ROIList, MediaInputBGR)
{
    initDLDTDataPath();

    cv::GFrame in;
    cv::GArray<cv::Rect> rr;
    cv::GArray<cv::GMat> age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(rr, in);
    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(age, gender));

    auto frame = MediaFrame::Create<TestMediaBGR>(m_in_mat);

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });
    comp.apply(cv::gin(frame, m_roi_list),
               cv::gout(m_out_gapi_ages, m_out_gapi_genders),
               cv::compile_args(cv::gapi::networks(pp)));

    validate();
}

TEST_F(ROIListNV12, MediaInputNV12)
{
    initDLDTDataPath();

    cv::GFrame in;
    cv::GArray<cv::Rect> rr;
    cv::GArray<cv::GMat> age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(rr, in);
    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(age, gender));

    auto frame = MediaFrame::Create<TestMediaNV12>(m_in_y, m_in_uv);

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });
    comp.apply(cv::gin(frame, m_roi_list),
               cv::gout(m_out_gapi_ages, m_out_gapi_genders),
               cv::compile_args(cv::gapi::networks(pp)));

    validate();
}

TEST(TestAgeGenderIE, MediaInputNV12)
{
    initDLDTDataPath();

    cv::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    params.device_id = "CPU";

    cv::Size sz{320, 240};
    cv::Mat in_y_mat(sz, CV_8UC1);
    cv::randu(in_y_mat, 0, 255);
    cv::Mat in_uv_mat(sz / 2, CV_8UC2);
    cv::randu(in_uv_mat, 0, 255);

    cv::Mat gapi_age, gapi_gender;

    // Load & run IE network
    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin        = cv::gimpl::ie::wrap::getPlugin(params);
        auto net           = cv::gimpl::ie::wrap::readNetwork(params);
        setNetParameters(net, true);
        auto this_network  = cv::gimpl::ie::wrap::loadNetwork(plugin, net, params);
        auto infer_request = this_network.CreateInferRequest();
        infer_request.SetBlob("data", cv::gapi::ie::util::to_ie(in_y_mat, in_uv_mat));
        infer_request.Infer();
        ie_age    = infer_request.GetBlob("age_conv3");
        ie_gender = infer_request.GetBlob("prob");
    }

    // Configure & run G-API
    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    cv::GFrame in;
    cv::GMat age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(age, gender));

    auto frame = MediaFrame::Create<TestMediaNV12>(in_y_mat, in_uv_mat);

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });
    comp.apply(cv::gin(frame), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));


    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(cv::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(cv::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
}

TEST(TestAgeGenderIE, MediaInputBGR)
{
    initDLDTDataPath();

    cv::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    params.device_id = "CPU";

    cv::Size sz{320, 240};
    cv::Mat in_mat(sz, CV_8UC3);
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

    cv::GFrame in;
    cv::GMat age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(age, gender));

    auto frame = MediaFrame::Create<TestMediaBGR>(in_mat);

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });
    comp.apply(cv::gin(frame), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));


    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(cv::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(cv::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
}

TEST(InferROI, MediaInputBGR)
{
    initDLDTDataPath();

    cv::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    params.device_id = "CPU";

    cv::Size sz{320, 240};
    cv::Mat in_mat(sz, CV_8UC3);
    cv::randu(in_mat, 0, 255);

    cv::Mat gapi_age, gapi_gender;
    cv::Rect rect(cv::Point{64, 60}, cv::Size{96, 96});

    // Load & run IE network
    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin        = cv::gimpl::ie::wrap::getPlugin(params);
        auto net           = cv::gimpl::ie::wrap::readNetwork(params);
        setNetParameters(net);
        auto this_network  = cv::gimpl::ie::wrap::loadNetwork(plugin, net, params);
        auto infer_request = this_network.CreateInferRequest();
        const auto ie_rc = IE::ROI {
            0u
            , static_cast<std::size_t>(rect.x)
            , static_cast<std::size_t>(rect.y)
            , static_cast<std::size_t>(rect.width)
            , static_cast<std::size_t>(rect.height)
        };
        IE::Blob::Ptr roi_blob = IE::make_shared_blob(cv::gapi::ie::util::to_ie(in_mat), ie_rc);
        infer_request.SetBlob("data", roi_blob);
        infer_request.Infer();
        ie_age    = infer_request.GetBlob("age_conv3");
        ie_gender = infer_request.GetBlob("prob");
    }

    // Configure & run G-API
    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    cv::GFrame in;
    cv::GOpaque<cv::Rect> roi;
    cv::GMat age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(roi, in);
    cv::GComputation comp(cv::GIn(in, roi), cv::GOut(age, gender));

    auto frame = MediaFrame::Create<TestMediaBGR>(in_mat);

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });
    comp.apply(cv::gin(frame, rect), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));


    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(cv::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(cv::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
}

TEST(InferROI, MediaInputNV12)
{
    initDLDTDataPath();

    cv::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    params.device_id = "CPU";

    cv::Size sz{320, 240};
    auto in_y_mat = cv::Mat{sz, CV_8UC1};
    cv::randu(in_y_mat, 0, 255);
    auto in_uv_mat = cv::Mat{sz / 2, CV_8UC2};
    cv::randu(in_uv_mat, 0, 255);

    cv::Mat gapi_age, gapi_gender;
    cv::Rect rect(cv::Point{64, 60}, cv::Size{96, 96});

    // Load & run IE network
    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin        = cv::gimpl::ie::wrap::getPlugin(params);
        auto net           = cv::gimpl::ie::wrap::readNetwork(params);
        setNetParameters(net, true);
        auto this_network  = cv::gimpl::ie::wrap::loadNetwork(plugin, net, params);
        auto infer_request = this_network.CreateInferRequest();
        const auto ie_rc = IE::ROI {
            0u
            , static_cast<std::size_t>(rect.x)
            , static_cast<std::size_t>(rect.y)
            , static_cast<std::size_t>(rect.width)
            , static_cast<std::size_t>(rect.height)
        };
        IE::Blob::Ptr roi_blob = IE::make_shared_blob(cv::gapi::ie::util::to_ie(in_y_mat, in_uv_mat), ie_rc);
        infer_request.SetBlob("data", roi_blob);
        infer_request.Infer();
        ie_age    = infer_request.GetBlob("age_conv3");
        ie_gender = infer_request.GetBlob("prob");
    }

    // Configure & run G-API
    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    cv::GFrame in;
    cv::GOpaque<cv::Rect> roi;
    cv::GMat age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(roi, in);
    cv::GComputation comp(cv::GIn(in, roi), cv::GOut(age, gender));

    auto frame = MediaFrame::Create<TestMediaNV12>(in_y_mat, in_uv_mat);

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });
    comp.apply(cv::gin(frame, rect), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));


    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(cv::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(cv::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
}

TEST_F(ROIList, Infer2MediaInputBGR)
{
    cv::GArray<cv::Rect> rr;
    cv::GFrame in;
    cv::GArray<cv::GMat> age, gender;
    std::tie(age, gender) = cv::gapi::infer2<AgeGender>(in, rr);
    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(age, gender));

    auto frame = MediaFrame::Create<TestMediaBGR>(m_in_mat);

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });
    comp.apply(cv::gin(frame, m_roi_list),
               cv::gout(m_out_gapi_ages, m_out_gapi_genders),
               cv::compile_args(cv::gapi::networks(pp)));
    validate();
}

TEST_F(ROIListNV12, Infer2MediaInputNV12)
{
    cv::GArray<cv::Rect> rr;
    cv::GFrame in;
    cv::GArray<cv::GMat> age, gender;
    std::tie(age, gender) = cv::gapi::infer2<AgeGender>(in, rr);
    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(age, gender));

    auto frame = MediaFrame::Create<TestMediaNV12>(m_in_y, m_in_uv);

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });
    comp.apply(cv::gin(frame, m_roi_list),
               cv::gout(m_out_gapi_ages, m_out_gapi_genders),
               cv::compile_args(cv::gapi::networks(pp)));
    validate();
}

TEST_F(SingleROI, GenericInfer)
{
    // Configure & run G-API
    cv::GMat in;
    cv::GOpaque<cv::Rect> roi;
    cv::GInferInputs inputs;
    inputs["data"] = in;

    auto outputs = cv::gapi::infer<cv::gapi::Generic>("age-gender-generic", roi, inputs);
    auto age     = outputs.at("age_conv3");
    auto gender  = outputs.at("prob");

    cv::GComputation comp(cv::GIn(in, roi), cv::GOut(age, gender));

    cv::gapi::ie::Params<cv::gapi::Generic> pp{
        "age-gender-generic", params.model_path, params.weights_path, params.device_id
    };
    pp.cfgNumRequests(2u);

    comp.apply(cv::gin(m_in_mat, m_roi), cv::gout(m_out_gapi_age, m_out_gapi_gender),
            cv::compile_args(cv::gapi::networks(pp)));

    validate();
}

TEST_F(SingleROI, GenericInferMediaBGR)
{
    // Configure & run G-API
    cv::GFrame in;
    cv::GOpaque<cv::Rect> roi;
    cv::GInferInputs inputs;
    inputs["data"] = in;

    auto outputs = cv::gapi::infer<cv::gapi::Generic>("age-gender-generic", roi, inputs);
    auto age     = outputs.at("age_conv3");
    auto gender  = outputs.at("prob");

    cv::GComputation comp(cv::GIn(in, roi), cv::GOut(age, gender));

    cv::gapi::ie::Params<cv::gapi::Generic> pp{
        "age-gender-generic", params.model_path, params.weights_path, params.device_id
    };
    pp.cfgNumRequests(2u);

    auto frame = MediaFrame::Create<TestMediaBGR>(m_in_mat);
    comp.apply(cv::gin(frame, m_roi), cv::gout(m_out_gapi_age, m_out_gapi_gender),
            cv::compile_args(cv::gapi::networks(pp)));

    validate();
}

TEST_F(SingleROINV12, GenericInferMediaNV12)
{
    // Configure & run G-API
    cv::GFrame in;
    cv::GOpaque<cv::Rect> roi;
    cv::GInferInputs inputs;
    inputs["data"] = in;

    auto outputs = cv::gapi::infer<cv::gapi::Generic>("age-gender-generic", roi, inputs);
    auto age     = outputs.at("age_conv3");
    auto gender  = outputs.at("prob");

    cv::GComputation comp(cv::GIn(in, roi), cv::GOut(age, gender));

    cv::gapi::ie::Params<cv::gapi::Generic> pp{
        "age-gender-generic", params.model_path, params.weights_path, params.device_id
    };
    pp.cfgNumRequests(2u);

    auto frame = MediaFrame::Create<TestMediaNV12>(m_in_y, m_in_uv);
    comp.apply(cv::gin(frame, m_roi), cv::gout(m_out_gapi_age, m_out_gapi_gender),
            cv::compile_args(cv::gapi::networks(pp)));

    validate();
}

TEST_F(ROIList, GenericInfer)
{
    cv::GMat in;
    cv::GArray<cv::Rect> rr;
    cv::GInferInputs inputs;
    inputs["data"] = in;

    auto outputs = cv::gapi::infer<cv::gapi::Generic>("age-gender-generic", rr, inputs);
    auto age     = outputs.at("age_conv3");
    auto gender  = outputs.at("prob");

    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(age, gender));

    cv::gapi::ie::Params<cv::gapi::Generic> pp{
        "age-gender-generic", params.model_path, params.weights_path, params.device_id
    };
    pp.cfgNumRequests(2u);

    comp.apply(cv::gin(m_in_mat, m_roi_list),
            cv::gout(m_out_gapi_ages, m_out_gapi_genders),
            cv::compile_args(cv::gapi::networks(pp)));

    validate();
}

TEST_F(ROIList, GenericInferMediaBGR)
{
    cv::GFrame in;
    cv::GArray<cv::Rect> rr;
    cv::GInferInputs inputs;
    inputs["data"] = in;

    auto outputs = cv::gapi::infer<cv::gapi::Generic>("age-gender-generic", rr, inputs);
    auto age     = outputs.at("age_conv3");
    auto gender  = outputs.at("prob");

    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(age, gender));

    cv::gapi::ie::Params<cv::gapi::Generic> pp{
        "age-gender-generic", params.model_path, params.weights_path, params.device_id
    };
    pp.cfgNumRequests(2u);

    auto frame = MediaFrame::Create<TestMediaBGR>(m_in_mat);
    comp.apply(cv::gin(frame, m_roi_list),
            cv::gout(m_out_gapi_ages, m_out_gapi_genders),
            cv::compile_args(cv::gapi::networks(pp)));

    validate();
}

TEST_F(ROIListNV12, GenericInferMediaNV12)
{
    cv::GFrame in;
    cv::GArray<cv::Rect> rr;
    cv::GInferInputs inputs;
    inputs["data"] = in;

    auto outputs = cv::gapi::infer<cv::gapi::Generic>("age-gender-generic", rr, inputs);
    auto age     = outputs.at("age_conv3");
    auto gender  = outputs.at("prob");

    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(age, gender));

    cv::gapi::ie::Params<cv::gapi::Generic> pp{
        "age-gender-generic", params.model_path, params.weights_path, params.device_id
    };
    pp.cfgNumRequests(2u);

    auto frame = MediaFrame::Create<TestMediaNV12>(m_in_y, m_in_uv);
    comp.apply(cv::gin(frame, m_roi_list),
            cv::gout(m_out_gapi_ages, m_out_gapi_genders),
            cv::compile_args(cv::gapi::networks(pp)));

    validate();
}

TEST_F(ROIList, GenericInfer2)
{
    cv::GArray<cv::Rect> rr;
    cv::GMat in;
    GInferListInputs list;
    list["data"] = rr;

    auto outputs = cv::gapi::infer2<cv::gapi::Generic>("age-gender-generic", in, list);
    auto age     = outputs.at("age_conv3");
    auto gender  = outputs.at("prob");

    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(age, gender));

    cv::gapi::ie::Params<cv::gapi::Generic> pp{
        "age-gender-generic", params.model_path, params.weights_path, params.device_id
    };
    pp.cfgNumRequests(2u);

    comp.apply(cv::gin(m_in_mat, m_roi_list),
            cv::gout(m_out_gapi_ages, m_out_gapi_genders),
            cv::compile_args(cv::gapi::networks(pp)));
    validate();
}

TEST_F(ROIList, GenericInfer2MediaInputBGR)
{
    cv::GArray<cv::Rect> rr;
    cv::GFrame in;
    GInferListInputs inputs;
    inputs["data"] = rr;

    auto outputs = cv::gapi::infer2<cv::gapi::Generic>("age-gender-generic", in, inputs);
    auto age     = outputs.at("age_conv3");
    auto gender  = outputs.at("prob");

    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(age, gender));

    cv::gapi::ie::Params<cv::gapi::Generic> pp{
        "age-gender-generic", params.model_path, params.weights_path, params.device_id
    };
    pp.cfgNumRequests(2u);

    auto frame = MediaFrame::Create<TestMediaBGR>(m_in_mat);
    comp.apply(cv::gin(frame, m_roi_list),
            cv::gout(m_out_gapi_ages, m_out_gapi_genders),
            cv::compile_args(cv::gapi::networks(pp)));
    validate();
}

TEST_F(ROIListNV12, GenericInfer2MediaInputNV12)
{
    cv::GArray<cv::Rect> rr;
    cv::GFrame in;
    GInferListInputs inputs;
    inputs["data"] = rr;

    auto outputs = cv::gapi::infer2<cv::gapi::Generic>("age-gender-generic", in, inputs);
    auto age     = outputs.at("age_conv3");
    auto gender  = outputs.at("prob");

    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(age, gender));

    cv::gapi::ie::Params<cv::gapi::Generic> pp{
        "age-gender-generic", params.model_path, params.weights_path, params.device_id
    };
    pp.cfgNumRequests(2u);

    auto frame = MediaFrame::Create<TestMediaNV12>(m_in_y, m_in_uv);
    comp.apply(cv::gin(frame, m_roi_list),
            cv::gout(m_out_gapi_ages, m_out_gapi_genders),
            cv::compile_args(cv::gapi::networks(pp)));
    validate();
}

TEST(Infer, SetInvalidNumberOfRequests)
{
    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    cv::gapi::ie::Params<AgeGender> pp{"model", "weights", "device"};

    EXPECT_ANY_THROW(pp.cfgNumRequests(0u));
}

TEST(Infer, TestStreamingInfer)
{
    initDLDTDataPath();

    std::string filepath = findDataFile("cv/video/768x576.avi");

    cv::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    params.device_id = "CPU";

    // Load IE network, initialize input data using that.
    cv::Mat in_mat;
    cv::Mat gapi_age, gapi_gender;

    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    cv::GMat in;
    cv::GMat age, gender;

    std::tie(age, gender) = cv::gapi::infer<AgeGender>(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .cfgNumRequests(4u);


    std::size_t num_frames = 0u;
    std::size_t max_frames = 10u;

    cv::VideoCapture cap;
    cap.open(filepath);
    if (!cap.isOpened())
        throw SkipTestException("Video file can not be opened");

    cap >> in_mat;
    auto pipeline = comp.compileStreaming(cv::compile_args(cv::gapi::networks(pp)));
    pipeline.setSource<cv::gapi::wip::GCaptureSource>(filepath);

    pipeline.start();
    while (num_frames < max_frames && pipeline.pull(cv::gout(gapi_age, gapi_gender)))
    {
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
        // Validate with IE itself (avoid DNN module dependency here)
        normAssert(cv::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
        normAssert(cv::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
        ++num_frames;
        cap >> in_mat;
    }
    pipeline.stop();
}

TEST(InferROI, TestStreamingInfer)
{
    initDLDTDataPath();

    std::string filepath = findDataFile("cv/video/768x576.avi");

    cv::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    params.device_id = "CPU";

    // Load IE network, initialize input data using that.
    cv::Mat in_mat;
    cv::Mat gapi_age, gapi_gender;
    cv::Rect rect(cv::Point{64, 60}, cv::Size{96, 96});

    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    cv::GMat in;
    cv::GOpaque<cv::Rect> roi;
    cv::GMat age, gender;

    std::tie(age, gender) = cv::gapi::infer<AgeGender>(roi, in);
    cv::GComputation comp(cv::GIn(in, roi), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .cfgNumRequests(4u);


    std::size_t num_frames = 0u;
    std::size_t max_frames = 10u;

    cv::VideoCapture cap;
    cap.open(filepath);
    if (!cap.isOpened())
        throw SkipTestException("Video file can not be opened");

    cap >> in_mat;
    auto pipeline = comp.compileStreaming(cv::compile_args(cv::gapi::networks(pp)));
    pipeline.setSource(
            cv::gin(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(filepath), rect));

    pipeline.start();
    while (num_frames < max_frames && pipeline.pull(cv::gout(gapi_age, gapi_gender)))
    {
        // Load & run IE network
        IE::Blob::Ptr ie_age, ie_gender;
        {
            auto plugin        = cv::gimpl::ie::wrap::getPlugin(params);
            auto net           = cv::gimpl::ie::wrap::readNetwork(params);
            setNetParameters(net);
            auto this_network  = cv::gimpl::ie::wrap::loadNetwork(plugin, net, params);
            auto infer_request = this_network.CreateInferRequest();
            const auto ie_rc = IE::ROI {
                0u
                , static_cast<std::size_t>(rect.x)
                , static_cast<std::size_t>(rect.y)
                , static_cast<std::size_t>(rect.width)
                , static_cast<std::size_t>(rect.height)
            };
            IE::Blob::Ptr roi_blob = IE::make_shared_blob(cv::gapi::ie::util::to_ie(in_mat), ie_rc);
            infer_request.SetBlob("data", roi_blob);
            infer_request.Infer();
            ie_age    = infer_request.GetBlob("age_conv3");
            ie_gender = infer_request.GetBlob("prob");
        }
        // Validate with IE itself (avoid DNN module dependency here)
        normAssert(cv::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
        normAssert(cv::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
        ++num_frames;
        cap >> in_mat;
    }
    pipeline.stop();
}

TEST(InferList, TestStreamingInfer)
{
    initDLDTDataPath();

    std::string filepath = findDataFile("cv/video/768x576.avi");

    cv::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    params.device_id = "CPU";

    // Load IE network, initialize input data using that.
    cv::Mat in_mat;
    std::vector<cv::Mat> ie_ages, ie_genders, gapi_ages, gapi_genders;

    std::vector<cv::Rect> roi_list = {
        cv::Rect(cv::Point{64, 60}, cv::Size{ 96,  96}),
        cv::Rect(cv::Point{50, 32}, cv::Size{128, 160}),
    };

    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    cv::GMat in;
    cv::GArray<cv::Rect> roi;
    cv::GArray<GMat> age, gender;

    std::tie(age, gender) = cv::gapi::infer<AgeGender>(roi, in);
    cv::GComputation comp(cv::GIn(in, roi), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .cfgNumRequests(4u);

    std::size_t num_frames = 0u;
    std::size_t max_frames = 10u;

    cv::VideoCapture cap;
    cap.open(filepath);
    if (!cap.isOpened())
        throw SkipTestException("Video file can not be opened");

    cap >> in_mat;
    auto pipeline = comp.compileStreaming(cv::compile_args(cv::gapi::networks(pp)));
    pipeline.setSource(
            cv::gin(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(filepath), roi_list));

    pipeline.start();
    while (num_frames < max_frames && pipeline.pull(cv::gout(gapi_ages, gapi_genders)))
    {
        {
            auto plugin        = cv::gimpl::ie::wrap::getPlugin(params);
            auto net           = cv::gimpl::ie::wrap::readNetwork(params);
            setNetParameters(net);
            auto this_network  = cv::gimpl::ie::wrap::loadNetwork(plugin, net, params);
            auto infer_request = this_network.CreateInferRequest();
            auto frame_blob = cv::gapi::ie::util::to_ie(in_mat);

            for (auto &&rc : roi_list) {
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
                ie_ages.push_back(to_ocv(infer_request.GetBlob("age_conv3")).clone());
                ie_genders.push_back(to_ocv(infer_request.GetBlob("prob")).clone());
            }
        } // namespace IE = ..
        // Validate with IE itself (avoid DNN module dependency here)
        normAssert(ie_ages   [0], gapi_ages   [0], "0: Test age output");
        normAssert(ie_genders[0], gapi_genders[0], "0: Test gender output");
        normAssert(ie_ages   [1], gapi_ages   [1], "1: Test age output");
        normAssert(ie_genders[1], gapi_genders[1], "1: Test gender output");

        ie_ages.clear();
        ie_genders.clear();

        ++num_frames;
        cap >> in_mat;
    }
}

TEST(Infer2, TestStreamingInfer)
{
    initDLDTDataPath();

    std::string filepath = findDataFile("cv/video/768x576.avi");

    cv::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    params.device_id = "CPU";

    // Load IE network, initialize input data using that.
    cv::Mat in_mat;
    std::vector<cv::Mat> ie_ages, ie_genders, gapi_ages, gapi_genders;

    std::vector<cv::Rect> roi_list = {
        cv::Rect(cv::Point{64, 60}, cv::Size{ 96,  96}),
        cv::Rect(cv::Point{50, 32}, cv::Size{128, 160}),
    };

    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    cv::GArray<cv::Rect> rr;
    cv::GMat in;
    cv::GArray<cv::GMat> age, gender;
    std::tie(age, gender) = cv::gapi::infer2<AgeGender>(in, rr);

    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .cfgNumRequests(4u);

    std::size_t num_frames = 0u;
    std::size_t max_frames = 10u;

    cv::VideoCapture cap;
    cap.open(filepath);
    if (!cap.isOpened())
        throw SkipTestException("Video file can not be opened");

    cap >> in_mat;
    auto pipeline = comp.compileStreaming(cv::compile_args(cv::gapi::networks(pp)));
    pipeline.setSource(
            cv::gin(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(filepath), roi_list));

    pipeline.start();
    while (num_frames < max_frames && pipeline.pull(cv::gout(gapi_ages, gapi_genders)))
    {
        {
            auto plugin        = cv::gimpl::ie::wrap::getPlugin(params);
            auto net           = cv::gimpl::ie::wrap::readNetwork(params);
            setNetParameters(net);
            auto this_network  = cv::gimpl::ie::wrap::loadNetwork(plugin, net, params);
            auto infer_request = this_network.CreateInferRequest();
            auto frame_blob = cv::gapi::ie::util::to_ie(in_mat);

            for (auto &&rc : roi_list) {
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
                ie_ages.push_back(to_ocv(infer_request.GetBlob("age_conv3")).clone());
                ie_genders.push_back(to_ocv(infer_request.GetBlob("prob")).clone());
            }
        } // namespace IE = ..
        // Validate with IE itself (avoid DNN module dependency here)
        normAssert(ie_ages   [0], gapi_ages   [0], "0: Test age output");
        normAssert(ie_genders[0], gapi_genders[0], "0: Test gender output");
        normAssert(ie_ages   [1], gapi_ages   [1], "1: Test age output");
        normAssert(ie_genders[1], gapi_genders[1], "1: Test gender output");

        ie_ages.clear();
        ie_genders.clear();

        ++num_frames;
        cap >> in_mat;
    }
    pipeline.stop();
}

TEST(InferEmptyList, TestStreamingInfer)
{
    initDLDTDataPath();

    std::string filepath = findDataFile("cv/video/768x576.avi");

    cv::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    params.device_id = "CPU";

    // Load IE network, initialize input data using that.
    cv::Mat in_mat;
    std::vector<cv::Mat> ie_ages, ie_genders, gapi_ages, gapi_genders;

    // NB: Empty list of roi
    std::vector<cv::Rect> roi_list;

    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    cv::GMat in;
    cv::GArray<cv::Rect> roi;
    cv::GArray<GMat> age, gender;

    std::tie(age, gender) = cv::gapi::infer<AgeGender>(roi, in);
    cv::GComputation comp(cv::GIn(in, roi), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .cfgNumRequests(4u);

    std::size_t num_frames = 0u;
    std::size_t max_frames = 1u;

    cv::VideoCapture cap;
    cap.open(filepath);
    if (!cap.isOpened())
        throw SkipTestException("Video file can not be opened");

    cap >> in_mat;
    auto pipeline = comp.compileStreaming(cv::compile_args(cv::gapi::networks(pp)));
    pipeline.setSource(
            cv::gin(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(filepath), roi_list));

    pipeline.start();
    while (num_frames < max_frames && pipeline.pull(cv::gout(gapi_ages, gapi_genders)))
    {
        EXPECT_TRUE(gapi_ages.empty());
        EXPECT_TRUE(gapi_genders.empty());
    }
}

TEST(Infer2EmptyList, TestStreamingInfer)
{
    initDLDTDataPath();

    std::string filepath = findDataFile("cv/video/768x576.avi");

    cv::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    params.device_id = "CPU";

    // Load IE network, initialize input data using that.
    cv::Mat in_mat;
    std::vector<cv::Mat> ie_ages, ie_genders, gapi_ages, gapi_genders;

    // NB: Empty list of roi
    std::vector<cv::Rect> roi_list;

    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    cv::GArray<cv::Rect> rr;
    cv::GMat in;
    cv::GArray<cv::GMat> age, gender;
    std::tie(age, gender) = cv::gapi::infer2<AgeGender>(in, rr);

    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .cfgNumRequests(4u);

    std::size_t num_frames = 0u;
    std::size_t max_frames = 1u;

    cv::VideoCapture cap;
    cap.open(filepath);
    if (!cap.isOpened())
        throw SkipTestException("Video file can not be opened");

    cap >> in_mat;
    auto pipeline = comp.compileStreaming(cv::compile_args(cv::gapi::networks(pp)));
    pipeline.setSource(
            cv::gin(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(filepath), roi_list));

    pipeline.start();
    while (num_frames < max_frames && pipeline.pull(cv::gout(gapi_ages, gapi_genders)))
    {
        EXPECT_TRUE(gapi_ages.empty());
        EXPECT_TRUE(gapi_genders.empty());
    }
}

TEST_F(InferWithReshape, TestInfer)
{
    // IE code
    infer(m_in_mat);
    // G-API code
    cv::GMat in;
    cv::GMat age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" }).cfgInputReshape({{"data", reshape_dims}});
    comp.apply(cv::gin(m_in_mat), cv::gout(m_out_gapi_ages.front(), m_out_gapi_genders.front()),
               cv::compile_args(cv::gapi::networks(pp)));
    // Validate
    validate();
}

TEST_F(InferWithReshape, TestInferInImage)
{
    // Input image already has 70x70 size
    cv::Mat rsz;
    cv::resize(m_in_mat, rsz, cv::Size(70, 70));
    // IE code
    infer(rsz);
    // G-API code
    cv::GMat in;
    cv::GMat age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" }).cfgInputReshape({"data"});
    // Reshape CNN input by input image size
    comp.apply(cv::gin(rsz), cv::gout(m_out_gapi_ages.front(), m_out_gapi_genders.front()),
               cv::compile_args(cv::gapi::networks(pp)));
    // Validate
    validate();
}

TEST_F(InferWithReshape, TestInferForSingleLayer)
{
    // IE code
    infer(m_in_mat);
    // G-API code
    cv::GMat in;
    cv::GMat age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .cfgInputReshape("data", reshape_dims);
    comp.apply(cv::gin(m_in_mat), cv::gout(m_out_gapi_ages.front(), m_out_gapi_genders.front()),
               cv::compile_args(cv::gapi::networks(pp)));
    // Validate
    validate();
}

TEST_F(InferWithReshape, TestInferList)
{
    // IE code
    infer(m_in_mat, true);
    // G-API code
    cv::GArray<cv::Rect> rr;
    cv::GMat in;
    cv::GArray<cv::GMat> age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(rr, in);
    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" }).cfgInputReshape({{"data", reshape_dims}});
    comp.apply(cv::gin(m_in_mat, m_roi_list),
               cv::gout(m_out_gapi_ages, m_out_gapi_genders),
               cv::compile_args(cv::gapi::networks(pp)));
    // Validate
    validate();
}

TEST_F(InferWithReshape, TestInferList2)
{
    // IE code
    infer(m_in_mat, true);
    // G-API code
    cv::GArray<cv::Rect> rr;
    cv::GMat in;
    cv::GArray<cv::GMat> age, gender;
    std::tie(age, gender) = cv::gapi::infer2<AgeGender>(in, rr);
    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" }).cfgInputReshape({{"data", reshape_dims}});
    comp.apply(cv::gin(m_in_mat, m_roi_list),
               cv::gout(m_out_gapi_ages, m_out_gapi_genders),
               cv::compile_args(cv::gapi::networks(pp)));
    // Validate
    validate();
}

TEST_F(InferWithReshape, TestInferListBGR)
{
    // IE code
    infer(m_in_mat, true);
    // G-API code
    cv::GArray<cv::Rect> rr;
    cv::GFrame in;
    cv::GArray<cv::GMat> age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(rr, in);
    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(age, gender));

    auto frame = MediaFrame::Create<TestMediaBGR>(m_in_mat);

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" }).cfgInputReshape({{"data", reshape_dims}});
    comp.apply(cv::gin(frame, m_roi_list),
               cv::gout(m_out_gapi_ages, m_out_gapi_genders),
               cv::compile_args(cv::gapi::networks(pp)));
    // Validate
    validate();
}

TEST_F(InferWithReshapeNV12, TestInferListYUV)
{
    // G-API code
    cv::GFrame in;
    cv::GArray<cv::Rect> rr;
    cv::GArray<cv::GMat> age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(rr, in);
    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(age, gender));

    auto frame = MediaFrame::Create<TestMediaNV12>(m_in_y, m_in_uv);

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" }).cfgInputReshape({{"data", reshape_dims}});
    comp.apply(cv::gin(frame, m_roi_list),
               cv::gout(m_out_gapi_ages, m_out_gapi_genders),
               cv::compile_args(cv::gapi::networks(pp)));
    // Validate
    validate();
}

TEST_F(ROIList, CallInferMultipleTimes)
{
    cv::GArray<cv::Rect> rr;
    cv::GMat in;
    cv::GArray<cv::GMat> age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(rr, in);
    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.weights_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });

    auto cc = comp.compile(cv::descr_of(cv::gin(m_in_mat, m_roi_list)),
                           cv::compile_args(cv::gapi::networks(pp)));

    for (int i = 0; i < 10; ++i) {
        cc(cv::gin(m_in_mat, m_roi_list), cv::gout(m_out_gapi_ages, m_out_gapi_genders));
    }

    validate();
}

TEST(IEFrameAdapter, blobParams)
{
    cv::Mat bgr = cv::Mat::eye(240, 320, CV_8UC3);
    cv::MediaFrame frame = cv::MediaFrame::Create<TestMediaBGR>(bgr);

    auto expected = std::make_pair(IE::TensorDesc{IE::Precision::U8, {1, 3, 300, 300},
                                                  IE::Layout::NCHW},
                                   IE::ParamMap{{"HELLO", 42}, {"COLOR_FORMAT",
                                                                IE::ColorFormat::NV12}});

    auto actual = cv::util::any_cast<decltype(expected)>(frame.blobParams());

    EXPECT_EQ(expected, actual);
}

namespace
{

struct Sync {
    std::mutex              m;
    std::condition_variable cv;
    int                     counter = 0;
};

class GMockMediaAdapter final: public cv::MediaFrame::IAdapter {
public:
    explicit GMockMediaAdapter(cv::Mat m, std::shared_ptr<Sync> sync)
        : m_mat(m), m_sync(sync) {
    }

    cv::GFrameDesc meta() const override {
        return cv::GFrameDesc{cv::MediaFormat::BGR, m_mat.size()};
    }

    cv::MediaFrame::View access(cv::MediaFrame::Access) override {
        cv::MediaFrame::View::Ptrs pp = { m_mat.ptr(), nullptr, nullptr, nullptr };
        cv::MediaFrame::View::Strides ss = { m_mat.step, 0u, 0u, 0u };
        return cv::MediaFrame::View(std::move(pp), std::move(ss));
    }

    ~GMockMediaAdapter() {
        {
            std::lock_guard<std::mutex> lk{m_sync->m};
            m_sync->counter--;
        }
        m_sync->cv.notify_one();
    }

private:
    cv::Mat               m_mat;
    std::shared_ptr<Sync> m_sync;
};

// NB: This source is needed to simulate real
// cases where the memory resources are limited.
// GMockSource(int limit) - accept the number of MediaFrames that
// the source can produce until resources are over.
class GMockSource : public cv::gapi::wip::IStreamSource {
public:
    explicit GMockSource(int limit)
        : m_limit(limit), m_mat(cv::Size(1920, 1080), CV_8UC3),
          m_sync(new Sync{}) {
        cv::randu(m_mat, cv::Scalar::all(0), cv::Scalar::all(255));
    }

    bool pull(cv::gapi::wip::Data& data) {
        std::unique_lock<std::mutex> lk(m_sync->m);
        m_sync->counter++;
        // NB: Can't produce new frames until old ones are released.
        m_sync->cv.wait(lk, [this]{return m_sync->counter <= m_limit;});

        data = cv::MediaFrame::Create<GMockMediaAdapter>(m_mat, m_sync);
        return true;
    }

    GMetaArg descr_of() const override {
        return GMetaArg{cv::GFrameDesc{cv::MediaFormat::BGR, m_mat.size()}};
    }

private:
    int                   m_limit;
    cv::Mat               m_mat;
    std::shared_ptr<Sync> m_sync;
};

struct LimitedSourceInfer: public ::testing::Test {
    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    LimitedSourceInfer()
        : comp([](){
            cv::GFrame in;
            cv::GMat age, gender;
            std::tie(age, gender) = cv::gapi::infer<AgeGender>(in);
            return cv::GComputation(cv::GIn(in), cv::GOut(age, gender));
        }) {
        initDLDTDataPath();
    }

    GStreamingCompiled compileStreaming(int nireq) {
        cv::gapi::ie::detail::ParamDesc params;
        params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
        params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
        params.device_id = "CPU";

        auto pp = cv::gapi::ie::Params<AgeGender> {
            params.model_path, params.weights_path, params.device_id }
        .cfgOutputLayers({ "age_conv3", "prob" })
        .cfgNumRequests(nireq);

        return comp.compileStreaming(cv::compile_args(cv::gapi::networks(pp)));
    }

    void run(const int max_frames, const int limit, const int nireq) {
        auto pipeline = compileStreaming(nireq);
        pipeline.setSource<GMockSource>(limit);
        pipeline.start();

        int num_frames = 0;
        while (num_frames != max_frames &&
               pipeline.pull(cv::gout(out_age, out_gender))) {
            ++num_frames;
        }
    }

    cv::GComputation comp;
    cv::Mat          out_age, out_gender;
};

} // anonymous namespace

TEST_F(LimitedSourceInfer, ReleaseFrame)
{
    constexpr int max_frames      = 50;
    constexpr int resources_limit = 1;
    constexpr int nireq           = 1;

    run(max_frames, resources_limit, nireq);
}

TEST_F(LimitedSourceInfer, ReleaseFrameAsync)
{
    constexpr int max_frames      = 50;
    constexpr int resources_limit = 4;
    constexpr int nireq           = 8;

    run(max_frames, resources_limit, nireq);
}

TEST(TestAgeGenderIE, InferWithBatch)
{
    initDLDTDataPath();

    constexpr int batch_size = 4;
    cv::gapi::ie::detail::ParamDesc params;
    params.model_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    params.weights_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    params.device_id = "CPU";

    cv::Mat in_mat({batch_size, 3, 320, 240}, CV_8U);
    cv::randu(in_mat, 0, 255);

    cv::Mat gapi_age, gapi_gender;

    // Load & run IE network
    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin = cv::gimpl::ie::wrap::getPlugin(params);
        auto net    = cv::gimpl::ie::wrap::readNetwork(params);
        setNetParameters(net);
        net.setBatchSize(batch_size);
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
    }.cfgOutputLayers({ "age_conv3", "prob" })
     .cfgBatchSize(batch_size);

    comp.apply(cv::gin(in_mat), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(cv::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(cv::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
}

TEST(ImportNetwork, Infer)
{
    const std::string device = "MYRIAD";
    skipIfDeviceNotAvailable(device);

    initDLDTDataPath();

    cv::gapi::ie::detail::ParamDesc params;
    params.model_path = compileAgeGenderBlob(device);
    params.device_id = device;

    cv::Mat in_mat(320, 240, CV_8UC3);
    cv::randu(in_mat, 0, 255);
    cv::Mat gapi_age, gapi_gender;

    // Load & run IE network
    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin = cv::gimpl::ie::wrap::getPlugin(params);
        auto this_network  = cv::gimpl::ie::wrap::importNetwork(plugin, params);
        auto infer_request = this_network.CreateInferRequest();
        IE::PreProcessInfo info;
        info.setResizeAlgorithm(IE::RESIZE_BILINEAR);
        infer_request.SetBlob("data", cv::gapi::ie::util::to_ie(in_mat), info);
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
        params.model_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });

    comp.apply(cv::gin(in_mat), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(cv::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(cv::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
}

TEST(ImportNetwork, InferNV12)
{
    const std::string device = "MYRIAD";
    skipIfDeviceNotAvailable(device);

    initDLDTDataPath();

    cv::gapi::ie::detail::ParamDesc params;
    params.model_path= compileAgeGenderBlob(device);
    params.device_id = device;

    cv::Size sz{320, 240};
    cv::Mat in_y_mat(sz, CV_8UC1);
    cv::randu(in_y_mat, 0, 255);
    cv::Mat in_uv_mat(sz / 2, CV_8UC2);
    cv::randu(in_uv_mat, 0, 255);

    cv::Mat gapi_age, gapi_gender;

    // Load & run IE network
    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin        = cv::gimpl::ie::wrap::getPlugin(params);
        auto this_network  = cv::gimpl::ie::wrap::importNetwork(plugin, params);
        auto infer_request = this_network.CreateInferRequest();
        IE::PreProcessInfo info;
        info.setResizeAlgorithm(IE::RESIZE_BILINEAR);
        info.setColorFormat(IE::ColorFormat::NV12);
        infer_request.SetBlob("data", cv::gapi::ie::util::to_ie(in_y_mat, in_uv_mat), info);
        infer_request.Infer();
        ie_age    = infer_request.GetBlob("age_conv3");
        ie_gender = infer_request.GetBlob("prob");
    }

    // Configure & run G-API
    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    cv::GFrame in;
    cv::GMat age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(age, gender));

    auto frame = MediaFrame::Create<TestMediaNV12>(in_y_mat, in_uv_mat);

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });
    comp.apply(cv::gin(frame), cv::gout(gapi_age, gapi_gender),
            cv::compile_args(cv::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(cv::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(cv::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
}

TEST(ImportNetwork, InferROI)
{
    const std::string device = "MYRIAD";
    skipIfDeviceNotAvailable(device);

    initDLDTDataPath();

    cv::gapi::ie::detail::ParamDesc params;
    params.model_path = compileAgeGenderBlob(device);
    params.device_id = device;

    cv::Mat in_mat(320, 240, CV_8UC3);
    cv::randu(in_mat, 0, 255);
    cv::Mat gapi_age, gapi_gender;
    cv::Rect rect(cv::Point{64, 60}, cv::Size{96, 96});

    // Load & run IE network
    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin = cv::gimpl::ie::wrap::getPlugin(params);
        auto this_network = cv::gimpl::ie::wrap::importNetwork(plugin, params);
        auto infer_request = this_network.CreateInferRequest();
        const auto ie_rc = IE::ROI {
            0u
            , static_cast<std::size_t>(rect.x)
            , static_cast<std::size_t>(rect.y)
            , static_cast<std::size_t>(rect.width)
            , static_cast<std::size_t>(rect.height)
        };
        IE::Blob::Ptr roi_blob = IE::make_shared_blob(cv::gapi::ie::util::to_ie(in_mat), ie_rc);
        IE::PreProcessInfo info;
        info.setResizeAlgorithm(IE::RESIZE_BILINEAR);
        infer_request.SetBlob("data", roi_blob, info);
        infer_request.Infer();
        ie_age    = infer_request.GetBlob("age_conv3");
        ie_gender = infer_request.GetBlob("prob");
    }

    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    cv::GMat in;
    cv::GOpaque<cv::Rect> roi;
    cv::GMat age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(roi, in);
    cv::GComputation comp(cv::GIn(in, roi), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });

    comp.apply(cv::gin(in_mat, rect), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(cv::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(cv::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
}

TEST(ImportNetwork, InferROINV12)
{
    const std::string device = "MYRIAD";
    skipIfDeviceNotAvailable(device);

    initDLDTDataPath();

    cv::gapi::ie::detail::ParamDesc params;
    params.model_path = compileAgeGenderBlob(device);
    params.device_id = device;

    cv::Size sz{320, 240};
    cv::Mat in_y_mat(sz, CV_8UC1);
    cv::randu(in_y_mat, 0, 255);
    cv::Mat in_uv_mat(sz / 2, CV_8UC2);
    cv::randu(in_uv_mat, 0, 255);
    cv::Rect rect(cv::Point{64, 60}, cv::Size{96, 96});

    cv::Mat gapi_age, gapi_gender;

    // Load & run IE network
    IE::Blob::Ptr ie_age, ie_gender;
    {
        auto plugin = cv::gimpl::ie::wrap::getPlugin(params);
        auto this_network = cv::gimpl::ie::wrap::importNetwork(plugin, params);
        auto infer_request = this_network.CreateInferRequest();
        const auto ie_rc = IE::ROI {
            0u
            , static_cast<std::size_t>(rect.x)
            , static_cast<std::size_t>(rect.y)
            , static_cast<std::size_t>(rect.width)
            , static_cast<std::size_t>(rect.height)
        };
        IE::Blob::Ptr roi_blob =
            IE::make_shared_blob(cv::gapi::ie::util::to_ie(in_y_mat, in_uv_mat), ie_rc);
        IE::PreProcessInfo info;
        info.setResizeAlgorithm(IE::RESIZE_BILINEAR);
        info.setColorFormat(IE::ColorFormat::NV12);
        infer_request.SetBlob("data", roi_blob, info);
        infer_request.Infer();
        ie_age    = infer_request.GetBlob("age_conv3");
        ie_gender = infer_request.GetBlob("prob");
    }

    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    cv::GFrame in;
    cv::GOpaque<cv::Rect> roi;
    cv::GMat age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(roi, in);
    cv::GComputation comp(cv::GIn(in, roi), cv::GOut(age, gender));

    auto frame = MediaFrame::Create<TestMediaNV12>(in_y_mat, in_uv_mat);

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });

    comp.apply(cv::gin(frame, rect), cv::gout(gapi_age, gapi_gender),
            cv::compile_args(cv::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    normAssert(cv::gapi::ie::util::to_ocv(ie_age),    gapi_age,    "Test age output"   );
    normAssert(cv::gapi::ie::util::to_ocv(ie_gender), gapi_gender, "Test gender output");
}

TEST(ImportNetwork, InferList)
{
    const std::string device = "MYRIAD";
    skipIfDeviceNotAvailable(device);

    initDLDTDataPath();

    cv::gapi::ie::detail::ParamDesc params;
    params.model_path = compileAgeGenderBlob(device);
    params.device_id = device;

    cv::Mat in_mat(320, 240, CV_8UC3);
    cv::randu(in_mat, 0, 255);
    std::vector<cv::Rect> roi_list = {
        cv::Rect(cv::Point{64, 60}, cv::Size{ 96,  96}),
        cv::Rect(cv::Point{50, 32}, cv::Size{128, 160}),
    };
    std::vector<cv::Mat> out_ie_ages, out_ie_genders, out_gapi_ages, out_gapi_genders;

    // Load & run IE network
    {
        auto plugin = cv::gimpl::ie::wrap::getPlugin(params);
        auto this_network  = cv::gimpl::ie::wrap::importNetwork(plugin, params);
        auto infer_request = this_network.CreateInferRequest();
        for (auto &&rc : roi_list) {
            const auto ie_rc = IE::ROI {
                0u
                , static_cast<std::size_t>(rc.x)
                , static_cast<std::size_t>(rc.y)
                , static_cast<std::size_t>(rc.width)
                , static_cast<std::size_t>(rc.height)
            };
            IE::Blob::Ptr roi_blob =
                IE::make_shared_blob(cv::gapi::ie::util::to_ie(in_mat), ie_rc);
            IE::PreProcessInfo info;
            info.setResizeAlgorithm(IE::RESIZE_BILINEAR);
            infer_request.SetBlob("data", roi_blob, info);
            infer_request.Infer();
            using namespace cv::gapi::ie::util;
            out_ie_ages.push_back(to_ocv(infer_request.GetBlob("age_conv3")).clone());
            out_ie_genders.push_back(to_ocv(infer_request.GetBlob("prob")).clone());
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
        params.model_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });

    comp.apply(cv::gin(in_mat, roi_list), cv::gout(out_gapi_ages, out_gapi_genders),
               cv::compile_args(cv::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    GAPI_Assert(!out_gapi_ages.empty());
    ASSERT_EQ(out_gapi_genders.size(), out_gapi_ages.size());
    ASSERT_EQ(out_gapi_ages.size(), out_ie_ages.size());
    ASSERT_EQ(out_gapi_genders.size(), out_ie_genders.size());

    const size_t size = out_gapi_ages.size();
    for (size_t i = 0; i < size; ++i) {
        normAssert(out_ie_ages   [i], out_gapi_ages   [i], "Test age output");
        normAssert(out_ie_genders[i], out_gapi_genders[i], "Test gender output");
    }
}

TEST(ImportNetwork, InferListNV12)
{
    const std::string device = "MYRIAD";
    skipIfDeviceNotAvailable(device);

    initDLDTDataPath();

    cv::gapi::ie::detail::ParamDesc params;
    params.model_path = compileAgeGenderBlob(device);
    params.device_id = device;

    cv::Size sz{320, 240};
    cv::Mat in_y_mat(sz, CV_8UC1);
    cv::randu(in_y_mat, 0, 255);
    cv::Mat in_uv_mat(sz / 2, CV_8UC2);
    cv::randu(in_uv_mat, 0, 255);
    std::vector<cv::Rect> roi_list = {
        cv::Rect(cv::Point{64, 60}, cv::Size{ 96,  96}),
        cv::Rect(cv::Point{50, 32}, cv::Size{128, 160}),
    };
    std::vector<cv::Mat> out_ie_ages, out_ie_genders, out_gapi_ages, out_gapi_genders;

    // Load & run IE network
    {
        auto plugin = cv::gimpl::ie::wrap::getPlugin(params);
        auto this_network  = cv::gimpl::ie::wrap::importNetwork(plugin, params);
        auto infer_request = this_network.CreateInferRequest();
        for (auto &&rc : roi_list) {
            const auto ie_rc = IE::ROI {
                0u
                , static_cast<std::size_t>(rc.x)
                , static_cast<std::size_t>(rc.y)
                , static_cast<std::size_t>(rc.width)
                , static_cast<std::size_t>(rc.height)
            };
            IE::Blob::Ptr roi_blob =
                IE::make_shared_blob(cv::gapi::ie::util::to_ie(in_y_mat, in_uv_mat), ie_rc);
            IE::PreProcessInfo info;
            info.setResizeAlgorithm(IE::RESIZE_BILINEAR);
            info.setColorFormat(IE::ColorFormat::NV12);
            infer_request.SetBlob("data", roi_blob, info);
            infer_request.Infer();
            using namespace cv::gapi::ie::util;
            out_ie_ages.push_back(to_ocv(infer_request.GetBlob("age_conv3")).clone());
            out_ie_genders.push_back(to_ocv(infer_request.GetBlob("prob")).clone());
        }
    }

    // Configure & run G-API
    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    cv::GArray<cv::Rect> rr;
    cv::GFrame in;
    cv::GArray<cv::GMat> age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(rr, in);
    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });

    auto frame = MediaFrame::Create<TestMediaNV12>(in_y_mat, in_uv_mat);

    comp.apply(cv::gin(frame, roi_list), cv::gout(out_gapi_ages, out_gapi_genders),
               cv::compile_args(cv::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    GAPI_Assert(!out_gapi_ages.empty());
    ASSERT_EQ(out_gapi_genders.size(), out_gapi_ages.size());
    ASSERT_EQ(out_gapi_ages.size(), out_ie_ages.size());
    ASSERT_EQ(out_gapi_genders.size(), out_ie_genders.size());

    const size_t size = out_gapi_ages.size();
    for (size_t i = 0; i < size; ++i) {
        normAssert(out_ie_ages   [i], out_gapi_ages   [i], "Test age output");
        normAssert(out_ie_genders[i], out_gapi_genders[i], "Test gender output");
    }
}

TEST(ImportNetwork, InferList2)
{
    const std::string device = "MYRIAD";
    skipIfDeviceNotAvailable(device);

    initDLDTDataPath();

    cv::gapi::ie::detail::ParamDesc params;
    params.model_path = compileAgeGenderBlob(device);
    params.device_id = device;

    cv::Mat in_mat(320, 240, CV_8UC3);
    cv::randu(in_mat, 0, 255);
    std::vector<cv::Rect> roi_list = {
        cv::Rect(cv::Point{64, 60}, cv::Size{ 96,  96}),
        cv::Rect(cv::Point{50, 32}, cv::Size{128, 160}),
    };
    std::vector<cv::Mat> out_ie_ages, out_ie_genders, out_gapi_ages, out_gapi_genders;

    // Load & run IE network
    {
        auto plugin = cv::gimpl::ie::wrap::getPlugin(params);
        auto this_network  = cv::gimpl::ie::wrap::importNetwork(plugin, params);
        auto infer_request = this_network.CreateInferRequest();
        for (auto &&rc : roi_list) {
            const auto ie_rc = IE::ROI {
                0u
                , static_cast<std::size_t>(rc.x)
                , static_cast<std::size_t>(rc.y)
                , static_cast<std::size_t>(rc.width)
                , static_cast<std::size_t>(rc.height)
            };
            IE::Blob::Ptr roi_blob =
                IE::make_shared_blob(cv::gapi::ie::util::to_ie(in_mat), ie_rc);
            IE::PreProcessInfo info;
            info.setResizeAlgorithm(IE::RESIZE_BILINEAR);
            infer_request.SetBlob("data", roi_blob, info);
            infer_request.Infer();
            using namespace cv::gapi::ie::util;
            out_ie_ages.push_back(to_ocv(infer_request.GetBlob("age_conv3")).clone());
            out_ie_genders.push_back(to_ocv(infer_request.GetBlob("prob")).clone());
        }
    }

    // Configure & run G-API
    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    cv::GArray<cv::Rect> rr;
    cv::GMat in;
    cv::GArray<cv::GMat> age, gender;
    std::tie(age, gender) = cv::gapi::infer2<AgeGender>(in, rr);
    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });

    comp.apply(cv::gin(in_mat, roi_list), cv::gout(out_gapi_ages, out_gapi_genders),
               cv::compile_args(cv::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    GAPI_Assert(!out_gapi_ages.empty());
    ASSERT_EQ(out_gapi_genders.size(), out_gapi_ages.size());
    ASSERT_EQ(out_gapi_ages.size(), out_ie_ages.size());
    ASSERT_EQ(out_gapi_genders.size(), out_ie_genders.size());

    const size_t size = out_gapi_ages.size();
    for (size_t i = 0; i < size; ++i) {
        normAssert(out_ie_ages   [i], out_gapi_ages   [i], "Test age output");
        normAssert(out_ie_genders[i], out_gapi_genders[i], "Test gender output");
    }
}

TEST(ImportNetwork, InferList2NV12)
{
    const std::string device = "MYRIAD";
    skipIfDeviceNotAvailable(device);

    initDLDTDataPath();

    cv::gapi::ie::detail::ParamDesc params;
    params.model_path = compileAgeGenderBlob(device);
    params.device_id = device;

    cv::Size sz{320, 240};
    cv::Mat in_y_mat(sz, CV_8UC1);
    cv::randu(in_y_mat, 0, 255);
    cv::Mat in_uv_mat(sz / 2, CV_8UC2);
    cv::randu(in_uv_mat, 0, 255);
    std::vector<cv::Rect> roi_list = {
        cv::Rect(cv::Point{64, 60}, cv::Size{ 96,  96}),
        cv::Rect(cv::Point{50, 32}, cv::Size{128, 160}),
    };
    std::vector<cv::Mat> out_ie_ages, out_ie_genders, out_gapi_ages, out_gapi_genders;

    // Load & run IE network
    {
        auto plugin = cv::gimpl::ie::wrap::getPlugin(params);
        auto this_network  = cv::gimpl::ie::wrap::importNetwork(plugin, params);
        auto infer_request = this_network.CreateInferRequest();
        for (auto &&rc : roi_list) {
            const auto ie_rc = IE::ROI {
                0u
                , static_cast<std::size_t>(rc.x)
                , static_cast<std::size_t>(rc.y)
                , static_cast<std::size_t>(rc.width)
                , static_cast<std::size_t>(rc.height)
            };
            IE::Blob::Ptr roi_blob =
                IE::make_shared_blob(cv::gapi::ie::util::to_ie(in_y_mat, in_uv_mat), ie_rc);
            IE::PreProcessInfo info;
            info.setResizeAlgorithm(IE::RESIZE_BILINEAR);
            info.setColorFormat(IE::ColorFormat::NV12);
            infer_request.SetBlob("data", roi_blob, info);
            infer_request.Infer();
            using namespace cv::gapi::ie::util;
            out_ie_ages.push_back(to_ocv(infer_request.GetBlob("age_conv3")).clone());
            out_ie_genders.push_back(to_ocv(infer_request.GetBlob("prob")).clone());
        }
    }

    // Configure & run G-API
    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    cv::GArray<cv::Rect> rr;
    cv::GFrame in;
    cv::GArray<cv::GMat> age, gender;
    std::tie(age, gender) = cv::gapi::infer2<AgeGender>(in, rr);
    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });

    auto frame = MediaFrame::Create<TestMediaNV12>(in_y_mat, in_uv_mat);

    comp.apply(cv::gin(frame, roi_list), cv::gout(out_gapi_ages, out_gapi_genders),
            cv::compile_args(cv::gapi::networks(pp)));

    // Validate with IE itself (avoid DNN module dependency here)
    GAPI_Assert(!out_gapi_ages.empty());
    ASSERT_EQ(out_gapi_genders.size(), out_gapi_ages.size());
    ASSERT_EQ(out_gapi_ages.size(), out_ie_ages.size());
    ASSERT_EQ(out_gapi_genders.size(), out_ie_genders.size());

    const size_t size = out_gapi_ages.size();
    for (size_t i = 0; i < size; ++i) {
        normAssert(out_ie_ages   [i], out_gapi_ages   [i], "Test age output");
        normAssert(out_ie_genders[i], out_gapi_genders[i], "Test gender output");
    }
}

TEST(TestAgeGender, ThrowBlobAndInputPrecisionMismatch)
{
    const std::string device = "MYRIAD";
    skipIfDeviceNotAvailable(device);

    initDLDTDataPath();

    cv::gapi::ie::detail::ParamDesc params;
    // NB: Precision for inputs is U8.
    params.model_path = compileAgeGenderBlob(device);
    params.device_id = device;

    // Configure & run G-API
    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "test-age-gender");

    cv::GMat in, age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(age, gender));

    auto pp = cv::gapi::ie::Params<AgeGender> {
        params.model_path, params.device_id
    }.cfgOutputLayers({ "age_conv3", "prob" });

    cv::Mat in_mat(320, 240, CV_32FC3);
    cv::randu(in_mat, 0, 1);
    cv::Mat gapi_age, gapi_gender;

    // NB: Blob precision is U8, but user pass FP32 data, so exception will be thrown.
    // Now exception comes directly from IE, but since G-API has information
    // about data precision at the compile stage, consider the possibility of
    // throwing exception from there.
    EXPECT_ANY_THROW(comp.apply(cv::gin(in_mat), cv::gout(gapi_age, gapi_gender),
                     cv::compile_args(cv::gapi::networks(pp))));
}

#ifdef HAVE_NGRAPH

TEST(Infer, ModelWith2DInputs)
{
    const std::string model_name   = "ModelWith2DInputs";
    const std::string model_path   = model_name + ".xml";
    const std::string weights_path = model_name + ".bin";
    const std::string device_id    = "CPU";
    const int W                    = 10;
    const int H                    = 5;

    // NB: Define model with 2D inputs.
    auto in1 = std::make_shared<ngraph::op::Parameter>(
        ngraph::element::Type_t::u8,
        ngraph::Shape(std::vector<size_t>{{H, W}})
    );
    auto in2 = std::make_shared<ngraph::op::Parameter>(
        ngraph::element::Type_t::u8,
        ngraph::Shape(std::vector<size_t>{{H, W}})
    );
    auto result = std::make_shared<ngraph::op::v1::Add>(in1, in2);
    auto func   = std::make_shared<ngraph::Function>(
        ngraph::OutputVector{result},
        ngraph::ParameterVector{in1, in2}
    );

    cv::Mat in_mat1(std::vector<int>{H, W}, CV_8U),
            in_mat2(std::vector<int>{H, W}, CV_8U),
            gapi_mat, ref_mat;

    cv::randu(in_mat1, 0, 100);
    cv::randu(in_mat2, 0, 100);
    cv::add(in_mat1, in_mat2, ref_mat, cv::noArray(), CV_32F);

    // Compile xml file
    IE::CNNNetwork(func).serialize(model_path);

    // Configure & run G-API
    cv::GMat g_in1, g_in2;
    cv::GInferInputs inputs;
    inputs[in1->get_name()] = g_in1;
    inputs[in2->get_name()] = g_in2;
    auto outputs = cv::gapi::infer<cv::gapi::Generic>(model_name, inputs);
    auto out = outputs.at(result->get_name());

    cv::GComputation comp(cv::GIn(g_in1, g_in2), cv::GOut(out));

    auto pp = cv::gapi::ie::Params<cv::gapi::Generic>(model_name,
                                                      model_path,
                                                      weights_path,
                                                      device_id);

    comp.apply(cv::gin(in_mat1, in_mat2), cv::gout(gapi_mat),
               cv::compile_args(cv::gapi::networks(pp)));

    normAssert(ref_mat, gapi_mat, "Test model output");
}

#endif // HAVE_NGRAPH

} // namespace opencv_test

#endif //  HAVE_INF_ENGINE
