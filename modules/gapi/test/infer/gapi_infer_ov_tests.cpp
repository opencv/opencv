// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation

#if defined HAVE_INF_ENGINE && INF_ENGINE_RELEASE >= 2022010000

#include "../test_precomp.hpp"

#include "backends/ov/util.hpp"

#include <opencv2/gapi/infer/ov.hpp>

#include <openvino/openvino.hpp>

namespace opencv_test
{

namespace {
// FIXME: taken from DNN module
void initDLDTDataPath()
{
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
}

static const std::string SUBDIR = "intel/age-gender-recognition-retail-0013/FP32/";

// FIXME: taken from the DNN module
void normAssert(cv::InputArray ref, cv::InputArray test,
                const char *comment /*= ""*/,
                double l1 = 0.00001, double lInf = 0.0001) {
    double normL1 = cvtest::norm(ref, test, cv::NORM_L1) / ref.getMat().total();
    EXPECT_LE(normL1, l1) << comment;

    double normInf = cvtest::norm(ref, test, cv::NORM_INF);
    EXPECT_LE(normInf, lInf) << comment;
}

// TODO: AGNetGenComp, AGNetTypedComp, AGNetOVComp, AGNetOVCompiled
// can be generalized to work with any model and used as parameters for tests.

struct AGNetGenParams {
    static constexpr const char* tag = "age-gender-generic";
    using Params = cv::gapi::ov::Params<cv::gapi::Generic>;

    static Params params(const std::string &xml,
                         const std::string &bin,
                         const std::string &device) {
        return {tag, xml, bin, device};
    }

    static Params params(const std::string &blob_path,
                         const std::string &device) {
        return {tag, blob_path, device};
    }
};

struct AGNetTypedParams {
    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "typed-age-gender");
    using Params = cv::gapi::ov::Params<AgeGender>;

    static Params params(const std::string &xml_path,
                         const std::string &bin_path,
                         const std::string &device) {
        return Params {
            xml_path, bin_path, device
        }.cfgOutputLayers({ "age_conv3", "prob" });
    }
};

struct AGNetTypedComp : AGNetTypedParams {
    static cv::GComputation create() {
        cv::GMat in;
        cv::GMat age, gender;
        std::tie(age, gender) = cv::gapi::infer<AgeGender>(in);
        return cv::GComputation{cv::GIn(in), cv::GOut(age, gender)};
    }
};

struct AGNetGenComp : public AGNetGenParams {
    static cv::GComputation create() {
        cv::GMat in;
        GInferInputs inputs;
        inputs["data"] = in;
        auto outputs = cv::gapi::infer<cv::gapi::Generic>(tag, inputs);
        auto age = outputs.at("age_conv3");
        auto gender = outputs.at("prob");
        return cv::GComputation{cv::GIn(in), cv::GOut(age, gender)};
    }
};

struct AGNetROIGenComp : AGNetGenParams {
    static cv::GComputation create() {
        cv::GMat in;
        cv::GOpaque<cv::Rect> roi;
        GInferInputs inputs;
        inputs["data"] = in;
        auto outputs = cv::gapi::infer<cv::gapi::Generic>(tag, roi, inputs);
        auto age = outputs.at("age_conv3");
        auto gender = outputs.at("prob");
        return cv::GComputation{cv::GIn(in, roi), cv::GOut(age, gender)};
    }
};

struct AGNetListGenComp : AGNetGenParams {
    static cv::GComputation create() {
        cv::GMat in;
        cv::GArray<cv::Rect> rois;
        GInferInputs inputs;
        inputs["data"] = in;
        auto outputs = cv::gapi::infer<cv::gapi::Generic>(tag, rois, inputs);
        auto age = outputs.at("age_conv3");
        auto gender = outputs.at("prob");
        return cv::GComputation{cv::GIn(in, rois), cv::GOut(age, gender)};
    }
};

struct AGNetList2GenComp : AGNetGenParams {
    static cv::GComputation create() {
        cv::GMat in;
        cv::GArray<cv::Rect> rois;
        GInferListInputs list;
        list["data"] = rois;
        auto outputs = cv::gapi::infer2<cv::gapi::Generic>(tag, in, list);
        auto age = outputs.at("age_conv3");
        auto gender = outputs.at("prob");
        return cv::GComputation{cv::GIn(in, rois), cv::GOut(age, gender)};
    }
};

class AGNetOVCompiled {
public:
    AGNetOVCompiled(ov::CompiledModel &&compiled_model)
        : m_compiled_model(std::move(compiled_model)),
          m_infer_request(m_compiled_model.create_infer_request()) {
    }

    void operator()(const cv::Mat  &in_mat,
                    const cv::Rect &roi,
                          cv::Mat  &age_mat,
                          cv::Mat  &gender_mat) {
        // FIXME: W & H could be extracted from model shape
        // but it's anyway used only for Age Gender model.
        // (Well won't work in case of reshape)
        const int W = 62;
        const int H = 62;
        cv::Mat resized_roi;
        cv::resize(in_mat(roi), resized_roi, cv::Size(W, H));
        (*this)(resized_roi, age_mat, gender_mat);
    }

    void operator()(const cv::Mat               &in_mat,
                    const std::vector<cv::Rect> &rois,
                    std::vector<cv::Mat>        &age_mats,
                    std::vector<cv::Mat>        &gender_mats) {
        for (size_t i = 0; i < rois.size(); ++i) {
            (*this)(in_mat, rois[i], age_mats[i], gender_mats[i]);
        }
    }

    void operator()(const cv::Mat &in_mat,
                          cv::Mat &age_mat,
                          cv::Mat &gender_mat) {
        auto input_tensor   = m_infer_request.get_input_tensor();
        cv::gapi::ov::util::to_ov(in_mat, input_tensor);

        m_infer_request.infer();

        auto age_tensor = m_infer_request.get_tensor("age_conv3");
        age_mat.create(cv::gapi::ov::util::to_ocv(age_tensor.get_shape()),
                       cv::gapi::ov::util::to_ocv(age_tensor.get_element_type()));
        cv::gapi::ov::util::to_ocv(age_tensor, age_mat);

        auto gender_tensor = m_infer_request.get_tensor("prob");
        gender_mat.create(cv::gapi::ov::util::to_ocv(gender_tensor.get_shape()),
                          cv::gapi::ov::util::to_ocv(gender_tensor.get_element_type()));
        cv::gapi::ov::util::to_ocv(gender_tensor, gender_mat);
    }

    void export_model(const std::string &outpath) {
        std::ofstream file{outpath, std::ios::out | std::ios::binary};
        GAPI_Assert(file.is_open());
        m_compiled_model.export_model(file);
    }

private:
    ov::CompiledModel m_compiled_model;
    ov::InferRequest  m_infer_request;
};

struct ImageInputPreproc {
    void operator()(ov::preprocess::PrePostProcessor &ppp) {
        ppp.input().tensor().set_layout(ov::Layout("NHWC"))
                            .set_element_type(ov::element::u8)
                            .set_shape({1, size.height, size.width, 3});
        ppp.input().model().set_layout(ov::Layout("NCHW"));
        ppp.input().preprocess().resize(::ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
    }

    cv::Size size;
};

class AGNetOVComp {
public:
    AGNetOVComp(const std::string &xml_path,
                const std::string &bin_path,
                const std::string &device)
        : m_device(device) {
        m_model = cv::gapi::ov::wrap::getCore()
            .read_model(xml_path, bin_path);
    }

    using PrePostProcessF = std::function<void(ov::preprocess::PrePostProcessor&)>;

    void cfgPrePostProcessing(PrePostProcessF f) {
        ov::preprocess::PrePostProcessor ppp(m_model);
        f(ppp);
        m_model = ppp.build();
    }

    AGNetOVCompiled compile() {
        auto compiled_model = cv::gapi::ov::wrap::getCore()
            .compile_model(m_model, m_device);
        return {std::move(compiled_model)};
    }

    void apply(const cv::Mat &in_mat,
                     cv::Mat &age_mat,
                     cv::Mat &gender_mat) {
        compile()(in_mat, age_mat, gender_mat);
    }

private:
    std::string m_device;
    std::shared_ptr<ov::Model> m_model;
};

struct BaseAgeGenderOV: public ::testing::Test {
    BaseAgeGenderOV() {
        initDLDTDataPath();
        xml_path  = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
        bin_path  = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
        device    = "CPU";
        blob_path = "age-gender-recognition-retail-0013.blob";
    }

    cv::Mat getRandomImage(const cv::Size &sz) {
        cv::Mat image(sz, CV_8UC3);
        cv::randu(image, 0, 255);
        return image;
    }

    cv::Mat getRandomTensor(const std::vector<int> &dims,
                            const int              depth) {
        cv::Mat tensor(dims, depth);
        cv::randu(tensor, -1, 1);
        return tensor;
    }

    std::string xml_path;
    std::string bin_path;
    std::string blob_path;
    std::string device;

};

struct TestAgeGenderOV : public BaseAgeGenderOV {
    cv::Mat ov_age, ov_gender, gapi_age, gapi_gender;

    void validate() {
        normAssert(ov_age,    gapi_age,    "Test age output"   );
        normAssert(ov_gender, gapi_gender, "Test gender output");
    }
};

struct TestAgeGenderListOV : public BaseAgeGenderOV {
    std::vector<cv::Mat> ov_age, ov_gender,
                         gapi_age, gapi_gender;

    std::vector<cv::Rect> roi_list = {
        cv::Rect(cv::Point{64, 60}, cv::Size{ 96,  96}),
        cv::Rect(cv::Point{50, 32}, cv::Size{128, 160}),
    };

    TestAgeGenderListOV() {
        ov_age.resize(roi_list.size());
        ov_gender.resize(roi_list.size());
        gapi_age.resize(roi_list.size());
        gapi_gender.resize(roi_list.size());
    }

    void validate() {
        ASSERT_EQ(ov_age.size(), ov_gender.size());

        ASSERT_EQ(ov_age.size(), gapi_age.size());
        ASSERT_EQ(ov_gender.size(), gapi_gender.size());

        for (size_t i = 0; i < ov_age.size(); ++i) {
            normAssert(ov_age[i], gapi_age[i], "Test age output");
            normAssert(ov_gender[i], gapi_gender[i], "Test gender output");
        }
    }
};

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
};

struct MediaFrameTestAgeGenderOV: public ::testing::Test {
    MediaFrameTestAgeGenderOV() {
        initDLDTDataPath();
        xml_path  = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml", false);
        bin_path  = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin", false);
        device    = "CPU";
        blob_path = "age-gender-recognition-retail-0013.blob";

        cv::Size sz{62, 62};
        m_in_mat = cv::Mat(sz, CV_8UC3);
        cv::resize(m_in_mat, m_in_mat, sz);

        m_in_y = cv::Mat{sz, CV_8UC1};
        cv::randu(m_in_y, 0, 255);
        m_in_uv = cv::Mat{sz / 2, CV_8UC2};
        cv::randu(m_in_uv, 0, 255);
    }

    cv::Mat m_in_y;
    cv::Mat m_in_uv;

    cv::Mat m_in_mat;

    cv::Mat m_out_ov_age;
    cv::Mat m_out_ov_gender;

    cv::Mat m_out_gapi_age;
    cv::Mat m_out_gapi_gender;

    std::string xml_path;
    std::string bin_path;
    std::string blob_path;
    std::string device;
    std::string image_path;

    using AGInfo = std::tuple<cv::GMat, cv::GMat>;
    G_API_NET(AgeGender, <AGInfo(cv::GMat)>, "typed-age-gender");

    void validate() {
        normAssert(m_out_ov_age,    m_out_gapi_age,    "0: Test age output");
        normAssert(m_out_ov_gender, m_out_gapi_gender, "0: Test gender output");
    }
}; // MediaFrameTestAgeGenderOV

} // anonymous namespace

TEST_F(MediaFrameTestAgeGenderOV, InferMediaInputBGR)
{
    // OpenVINO
    AGNetOVComp ref(xml_path, bin_path, device);
    ref.cfgPrePostProcessing([](ov::preprocess::PrePostProcessor &ppp) {
        ppp.input().tensor().set_element_type(ov::element::u8);
        ppp.input().tensor().set_layout("NHWC");
    });
    ref.compile()(m_in_mat, m_out_ov_age, m_out_ov_gender);

    // G-API
    cv::GFrame in;
    cv::GMat age, gender;
    std::tie(age, gender) = cv::gapi::infer<AgeGender>(in);
    cv::GComputation comp{cv::GIn(in), cv::GOut(age, gender)};

    auto frame = MediaFrame::Create<TestMediaBGR>(m_in_mat);
    auto pp = cv::gapi::ov::Params<AgeGender> {
        xml_path, bin_path, device
    }.cfgOutputLayers({ "age_conv3", "prob" });

    comp.apply(cv::gin(frame),
               cv::gout(m_out_gapi_age, m_out_gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    validate();
}

TEST_F(MediaFrameTestAgeGenderOV, InferROIGenericMediaInputBGR) {
    // OpenVINO
    cv::Rect roi(cv::Rect(cv::Point{20, 25}, cv::Size{16, 16}));
    auto frame = MediaFrame::Create<TestMediaBGR>(m_in_mat);
    static constexpr const char* tag = "age-gender-generic";

    // OpenVINO
    AGNetOVComp ref(xml_path, bin_path, device);
    ref.cfgPrePostProcessing([](ov::preprocess::PrePostProcessor &ppp) {
        ppp.input().tensor().set_element_type(ov::element::u8);
        ppp.input().tensor().set_layout("NHWC");
    });
    ref.compile()(m_in_mat, roi, m_out_ov_age, m_out_ov_gender);

    // G-API
    cv::GFrame in;
    cv::GOpaque<cv::Rect> rr;
    GInferInputs inputs;
    inputs["data"] = in;
    auto outputs = cv::gapi::infer<cv::gapi::Generic>(tag, rr, inputs);
    auto age = outputs.at("age_conv3");
    auto gender = outputs.at("prob");
    cv::GComputation comp{cv::GIn(in, rr), cv::GOut(age, gender)};

    auto pp = AGNetROIGenComp::params(xml_path, bin_path, device);

    comp.apply(cv::gin(frame, roi), cv::gout(m_out_gapi_age, m_out_gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    validate();
}

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

TEST_F(MediaFrameTestAgeGenderOV, TestMediaNV12AgeGenderOV)
{
    cv::GFrame in;
    cv::GOpaque<cv::Rect> rr;
    GInferInputs inputs;
    inputs["data"] = in;
    static constexpr const char* tag = "age-gender-generic";
    auto outputs = cv::gapi::infer<cv::gapi::Generic>(tag, rr, inputs);
    auto age = outputs.at("age_conv3");
    auto gender = outputs.at("prob");
    cv::GComputation comp{cv::GIn(in, rr), cv::GOut(age, gender)};

    auto frame = MediaFrame::Create<TestMediaNV12>(m_in_y, m_in_uv);
    auto pp = AGNetROIGenComp::params(xml_path, bin_path, device);

    cv::Rect roi(cv::Rect(cv::Point{20, 25}, cv::Size{16, 16}));

    EXPECT_NO_THROW(comp.apply(cv::gin(frame, roi),
                    cv::gout(m_out_gapi_age, m_out_gapi_gender),
                    cv::compile_args(cv::gapi::networks(pp))));
}

// TODO: Make all of tests below parmetrized to avoid code duplication
TEST_F(TestAgeGenderOV, Infer_Tensor) {
    const auto in_mat = getRandomTensor({1, 3, 62, 62}, CV_32F);
    // OpenVINO
    AGNetOVComp ref(xml_path, bin_path, device);
    ref.apply(in_mat, ov_age, ov_gender);

    // G-API
    auto comp = AGNetTypedComp::create();
    auto pp   = AGNetTypedComp::params(xml_path, bin_path, device);
    comp.apply(cv::gin(in_mat), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    // Assert
    validate();
}

TEST_F(TestAgeGenderOV, Infer_Image) {
    const auto in_mat = getRandomImage({300, 300});

    // OpenVINO
    AGNetOVComp ref(xml_path, bin_path, device);
    ref.cfgPrePostProcessing(ImageInputPreproc{in_mat.size()});
    ref.apply(in_mat, ov_age, ov_gender);

    // G-API
    auto comp = AGNetTypedComp::create();
    auto pp   = AGNetTypedComp::params(xml_path, bin_path, device);
    comp.apply(cv::gin(in_mat), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    // Assert
    validate();
}

TEST_F(TestAgeGenderOV, InferGeneric_Tensor) {
    const auto in_mat = getRandomTensor({1, 3, 62, 62}, CV_32F);

    // OpenVINO
    AGNetOVComp ref(xml_path, bin_path, device);
    ref.apply(in_mat, ov_age, ov_gender);

    // G-API
    auto comp = AGNetGenComp::create();
    auto pp   = AGNetGenComp::params(xml_path, bin_path, device);
    comp.apply(cv::gin(in_mat), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    // Assert
    validate();
}

TEST_F(TestAgeGenderOV, InferGenericImage) {
    const auto in_mat = getRandomImage({300, 300});

    // OpenVINO
    AGNetOVComp ref(xml_path, bin_path, device);
    ref.cfgPrePostProcessing(ImageInputPreproc{in_mat.size()});
    ref.apply(in_mat, ov_age, ov_gender);

    // G-API
    auto comp = AGNetGenComp::create();
    auto pp   = AGNetGenComp::params(xml_path, bin_path, device);
    comp.apply(cv::gin(in_mat), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    // Assert
    validate();
}

TEST_F(TestAgeGenderOV, InferGeneric_ImageBlob) {
    const auto in_mat = getRandomImage({300, 300});

    // OpenVINO
    AGNetOVComp ref(xml_path, bin_path, device);
    ref.cfgPrePostProcessing(ImageInputPreproc{in_mat.size()});
    auto cc_ref = ref.compile();
    // NB: Output blob will contain preprocessing inside.
    cc_ref.export_model(blob_path);
    cc_ref(in_mat, ov_age, ov_gender);

    // G-API
    auto comp = AGNetGenComp::create();
    auto pp   = AGNetGenComp::params(blob_path, device);
    comp.apply(cv::gin(in_mat), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    // Assert
    validate();
}

TEST_F(TestAgeGenderOV, InferGeneric_TensorBlob) {
    const auto in_mat = getRandomTensor({1, 3, 62, 62}, CV_32F);

    // OpenVINO
    AGNetOVComp ref(xml_path, bin_path, device);
    auto cc_ref = ref.compile();
    cc_ref.export_model(blob_path);
    cc_ref(in_mat, ov_age, ov_gender);

    // G-API
    auto comp = AGNetGenComp::create();
    auto pp   = AGNetGenComp::params(blob_path, device);
    comp.apply(cv::gin(in_mat), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    // Assert
    validate();
}

TEST_F(TestAgeGenderOV, InferGeneric_BothOutputsFP16) {
    const auto in_mat = getRandomTensor({1, 3, 62, 62}, CV_32F);

    // OpenVINO
    AGNetOVComp ref(xml_path, bin_path, device);
    ref.cfgPrePostProcessing([](ov::preprocess::PrePostProcessor &ppp){
        ppp.output(0).tensor().set_element_type(ov::element::f16);
        ppp.output(1).tensor().set_element_type(ov::element::f16);
    });
    ref.apply(in_mat, ov_age, ov_gender);

    // G-API
    auto comp = AGNetGenComp::create();
    auto pp   = AGNetGenComp::params(xml_path, bin_path, device);
    pp.cfgOutputTensorPrecision(CV_16F);

    comp.apply(cv::gin(in_mat), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    // Assert
    validate();
}

TEST_F(TestAgeGenderOV, InferGeneric_OneOutputFP16) {
    const auto in_mat = getRandomTensor({1, 3, 62, 62}, CV_32F);

    // OpenVINO
    const std::string fp16_output_name = "prob";
    AGNetOVComp ref(xml_path, bin_path, device);
    ref.cfgPrePostProcessing([&](ov::preprocess::PrePostProcessor &ppp){
        ppp.output(fp16_output_name).tensor().set_element_type(ov::element::f16);
    });
    ref.apply(in_mat, ov_age, ov_gender);

    // G-API
    auto comp = AGNetGenComp::create();
    auto pp   = AGNetGenComp::params(xml_path, bin_path, device);
    pp.cfgOutputTensorPrecision({{fp16_output_name, CV_16F}});

    comp.apply(cv::gin(in_mat), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    // Assert
    validate();
}

TEST_F(TestAgeGenderOV, InferGeneric_ThrowCfgOutputPrecForBlob) {
    // OpenVINO (Just for blob compilation)
    AGNetOVComp ref(xml_path, bin_path, device);
    auto cc_ref = ref.compile();
    cc_ref.export_model(blob_path);

    // G-API
    auto comp = AGNetGenComp::create();
    auto pp   = AGNetGenComp::params(blob_path, device);

    EXPECT_ANY_THROW(pp.cfgOutputTensorPrecision(CV_16F));
}

TEST_F(TestAgeGenderOV, InferGeneric_ThrowInvalidConfigIR) {
    // G-API
    auto comp = AGNetGenComp::create();
    auto pp   = AGNetGenComp::params(xml_path, bin_path, device);
    pp.cfgPluginConfig({{"some_key", "some_value"}});

    EXPECT_ANY_THROW(comp.compile(cv::GMatDesc{CV_8U,3,cv::Size{320, 240}},
                                  cv::compile_args(cv::gapi::networks(pp))));
}

TEST_F(TestAgeGenderOV, InferGeneric_ThrowInvalidConfigBlob) {
    // OpenVINO (Just for blob compilation)
    AGNetOVComp ref(xml_path, bin_path, device);
    auto cc_ref = ref.compile();
    cc_ref.export_model(blob_path);

    // G-API
    auto comp = AGNetGenComp::create();
    auto pp   = AGNetGenComp::params(blob_path, device);
    pp.cfgPluginConfig({{"some_key", "some_value"}});

    EXPECT_ANY_THROW(comp.compile(cv::GMatDesc{CV_8U,3,cv::Size{320, 240}},
                                  cv::compile_args(cv::gapi::networks(pp))));
}

TEST_F(TestAgeGenderOV, Infer_ThrowInvalidImageLayout) {
    const auto in_mat = getRandomImage({300, 300});
    auto comp = AGNetTypedComp::create();
    auto pp = AGNetTypedComp::params(xml_path, bin_path, device);

    pp.cfgInputTensorLayout("NCHW");

    EXPECT_ANY_THROW(comp.compile(cv::descr_of(in_mat),
                     cv::compile_args(cv::gapi::networks(pp))));
}

TEST_F(TestAgeGenderOV, Infer_TensorWithPreproc) {
    const auto in_mat = getRandomTensor({1, 240, 320, 3}, CV_32F);

    // OpenVINO
    AGNetOVComp ref(xml_path, bin_path, device);
    ref.cfgPrePostProcessing([](ov::preprocess::PrePostProcessor &ppp) {
        auto& input = ppp.input();
        input.tensor().set_spatial_static_shape(240, 320)
                      .set_layout("NHWC");
        input.preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
    });
    ref.apply(in_mat, ov_age, ov_gender);

    // G-API
    auto comp = AGNetTypedComp::create();
    auto pp = AGNetTypedComp::params(xml_path, bin_path, device);
    pp.cfgResize(cv::INTER_LINEAR)
      .cfgInputTensorLayout("NHWC");

    comp.apply(cv::gin(in_mat), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    // Assert
    validate();
}

TEST_F(TestAgeGenderOV, InferROIGeneric_Image) {
    const auto in_mat = getRandomImage({300, 300});
    cv::Rect roi(cv::Rect(cv::Point{64, 60}, cv::Size{96, 96}));

    // OpenVINO
    AGNetOVComp ref(xml_path, bin_path, device);
    ref.cfgPrePostProcessing([](ov::preprocess::PrePostProcessor &ppp) {
        ppp.input().tensor().set_element_type(ov::element::u8);
        ppp.input().tensor().set_layout("NHWC");
    });
    ref.compile()(in_mat, roi, ov_age, ov_gender);

    // G-API
    auto comp = AGNetROIGenComp::create();
    auto pp   = AGNetROIGenComp::params(xml_path, bin_path, device);

    comp.apply(cv::gin(in_mat, roi), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    // Assert
    validate();
}

TEST_F(TestAgeGenderOV, InferROIGeneric_ThrowIncorrectLayout) {
    const auto in_mat = getRandomImage({300, 300});
    cv::Rect roi(cv::Rect(cv::Point{64, 60}, cv::Size{96, 96}));

    // G-API
    auto comp = AGNetROIGenComp::create();
    auto pp   = AGNetROIGenComp::params(xml_path, bin_path, device);

    pp.cfgInputTensorLayout("NCHW");
    EXPECT_ANY_THROW(comp.apply(cv::gin(in_mat, roi), cv::gout(gapi_age, gapi_gender),
                     cv::compile_args(cv::gapi::networks(pp))));
}

TEST_F(TestAgeGenderOV, InferROIGeneric_ThrowTensorInput) {
    const auto in_mat = getRandomTensor({1, 3, 62, 62}, CV_32F);
    cv::Rect roi(cv::Rect(cv::Point{64, 60}, cv::Size{96, 96}));

    // G-API
    auto comp = AGNetROIGenComp::create();
    auto pp   = AGNetROIGenComp::params(xml_path, bin_path, device);

    EXPECT_ANY_THROW(comp.apply(cv::gin(in_mat, roi), cv::gout(gapi_age, gapi_gender),
                                cv::compile_args(cv::gapi::networks(pp))));
}

TEST_F(TestAgeGenderOV, InferROIGeneric_ThrowExplicitResize) {
    const auto in_mat = getRandomImage({300, 300});
    cv::Rect roi(cv::Rect(cv::Point{64, 60}, cv::Size{96, 96}));

    // G-API
    auto comp = AGNetROIGenComp::create();
    auto pp   = AGNetROIGenComp::params(xml_path, bin_path, device);

    pp.cfgResize(cv::INTER_LINEAR);
    EXPECT_ANY_THROW(comp.apply(cv::gin(in_mat, roi), cv::gout(gapi_age, gapi_gender),
                     cv::compile_args(cv::gapi::networks(pp))));
}

TEST_F(TestAgeGenderListOV, InferListGeneric_Image) {
    const auto in_mat = getRandomImage({300, 300});

    // OpenVINO
    AGNetOVComp ref(xml_path, bin_path, device);
    ref.cfgPrePostProcessing([](ov::preprocess::PrePostProcessor &ppp) {
        ppp.input().tensor().set_element_type(ov::element::u8);
        ppp.input().tensor().set_layout("NHWC");
    });
    ref.compile()(in_mat, roi_list, ov_age, ov_gender);

    // G-API
    auto comp = AGNetListGenComp::create();
    auto pp   = AGNetListGenComp::params(xml_path, bin_path, device);

    comp.apply(cv::gin(in_mat, roi_list), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    // Assert
    validate();
}

TEST_F(TestAgeGenderListOV, InferList2Generic_Image) {
    const auto in_mat = getRandomImage({300, 300});

    // OpenVINO
    AGNetOVComp ref(xml_path, bin_path, device);
    ref.cfgPrePostProcessing([](ov::preprocess::PrePostProcessor &ppp) {
        ppp.input().tensor().set_element_type(ov::element::u8);
        ppp.input().tensor().set_layout("NHWC");
    });
    ref.compile()(in_mat, roi_list, ov_age, ov_gender);

    // G-API
    auto comp = AGNetList2GenComp::create();
    auto pp   = AGNetList2GenComp::params(xml_path, bin_path, device);

    comp.apply(cv::gin(in_mat, roi_list), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    // Assert
    validate();
}

static ov::element::Type toOV(int depth) {
    switch (depth) {
    case CV_8U:  return ov::element::u8;
    case CV_32S: return ov::element::i32;
    case CV_32F: return ov::element::f32;
    case CV_16F: return ov::element::f16;
    default: GAPI_Error("OV Backend: Unsupported data type");
    }
    return ov::element::undefined;
}

struct TestMeanScaleOV : public ::testing::TestWithParam<int>{
    G_API_NET(IdentityNet, <cv::GMat(cv::GMat)>, "test-identity-net");

    static cv::GComputation create() {
        cv::GMat in;
        cv::GMat out;
        out = cv::gapi::infer<IdentityNet>(in);

        return cv::GComputation{cv::GIn(in), cv::GOut(out)};
    }

    using Params = cv::gapi::ov::Params<IdentityNet>;
    static Params params(const std::string &xml_path,
                         const std::string &bin_path,
                         const std::string &device) {
        return Params {
            xml_path, bin_path, device
        }.cfgInputModelLayout("NHWC")
         .cfgOutputLayers({ "output" });
    }

    TestMeanScaleOV() {
        initDLDTDataPath();

        m_model_path = findDataFile("gapi/ov/identity_net_100x100.xml");
        m_weights_path = findDataFile("gapi/ov/identity_net_100x100.bin");
        m_device_id = "CPU";

        m_ov_model = cv::gapi::ov::wrap::getCore()
            .read_model(m_model_path, m_weights_path);

        auto input_depth = GetParam();
        auto input = cv::imread(findDataFile("gapi/gapi_logo.jpg"));
        input.convertTo(m_in_mat, input_depth);
    }

    void addPreprocToOV(
        std::function<void(ov::preprocess::PrePostProcessor&)> f) {

        auto input_depth = GetParam();

        ov::preprocess::PrePostProcessor ppp(m_ov_model);
        ppp.input().tensor().set_layout(ov::Layout("NHWC"))
                            .set_element_type(toOV(input_depth))
                            .set_shape({ 1, 100, 100, 3 });
        ppp.input().model().set_layout(ov::Layout("NHWC"));
        f(ppp);
        m_ov_model = ppp.build();
    }

    void runOV() {
        auto compiled_model = cv::gapi::ov::wrap::getCore()
            .compile_model(m_ov_model, m_device_id);
        auto infer_request = compiled_model.create_infer_request();

        auto input_tensor = infer_request.get_input_tensor();
        cv::gapi::ov::util::to_ov(m_in_mat, input_tensor);

        infer_request.infer();

        auto out_tensor = infer_request.get_tensor("output");
        m_out_mat_ov.create(cv::gapi::ov::util::to_ocv(out_tensor.get_shape()),
                            cv::gapi::ov::util::to_ocv(out_tensor.get_element_type()));
        cv::gapi::ov::util::to_ocv(out_tensor, m_out_mat_ov);
    }

    std::string m_model_path;
    std::string m_weights_path;
    std::string m_device_id;

    std::shared_ptr<ov::Model> m_ov_model;

    cv::Mat m_in_mat;
    cv::Mat m_out_mat_gapi;
    cv::Mat m_out_mat_ov;
};

TEST_P(TestMeanScaleOV, Mean)
{
    int input_depth = GetParam();

    std::vector<float> mean_values{ 220.1779, 218.9857, 217.8986 };

    // Run OV reference pipeline:
    {
        addPreprocToOV([&](ov::preprocess::PrePostProcessor& ppp) {
            if (input_depth == CV_8U || input_depth == CV_32S) {
                ppp.input().preprocess().convert_element_type(ov::element::f32);
            }
            ppp.input().preprocess().mean(mean_values);
            });
        runOV();
    }

    // Run G-API
    GComputation comp = create();
    auto pp = params(m_model_path, m_weights_path, m_device_id);
    pp.cfgMean(mean_values);

    comp.apply(cv::gin(m_in_mat), cv::gout(m_out_mat_gapi),
               cv::compile_args(cv::gapi::networks(pp)));

    // Validate OV results against G-API ones:
    normAssert(m_out_mat_ov, m_out_mat_gapi, "Test output");
}

TEST_P(TestMeanScaleOV, Scale)
{
    int input_depth = GetParam();

    std::vector<float> scale_values{ 2., 2., 2. };

    // Run OV reference pipeline:
    {
        addPreprocToOV([&](ov::preprocess::PrePostProcessor& ppp) {
            if (input_depth == CV_8U || input_depth == CV_32S) {
                ppp.input().preprocess().convert_element_type(ov::element::f32);
            }
            ppp.input().preprocess().scale(scale_values);
            });
        runOV();
    }

    // Run G-API
    GComputation comp = create();
    auto pp = params(m_model_path, m_weights_path, m_device_id);
    pp.cfgScale(scale_values);

    comp.apply(cv::gin(m_in_mat), cv::gout(m_out_mat_gapi),
        cv::compile_args(cv::gapi::networks(pp)));

    // Validate OV results against G-API ones:
    normAssert(m_out_mat_ov, m_out_mat_gapi, "Test output");
}

TEST_P(TestMeanScaleOV, MeanAndScale)
{
    int input_depth = GetParam();

    std::vector<float> mean_values{ 220.1779, 218.9857, 217.8986 };
    std::vector<float> scale_values{ 2., 2., 2. };

    // Run OV reference pipeline:
    {
        addPreprocToOV([&](ov::preprocess::PrePostProcessor& ppp) {
            if (input_depth == CV_8U || input_depth == CV_32S) {
                ppp.input().preprocess().convert_element_type(ov::element::f32);
            }
            ppp.input().preprocess().mean(mean_values);
            ppp.input().preprocess().scale(scale_values);
            });
        runOV();
    }

    // Run G-API
    GComputation comp = create();
    auto pp = params(m_model_path, m_weights_path, m_device_id);
    pp.cfgMean(mean_values);
    pp.cfgScale(scale_values);

    comp.apply(cv::gin(m_in_mat), cv::gout(m_out_mat_gapi),
        cv::compile_args(cv::gapi::networks(pp)));

    // Validate OV results against G-API ones:
    normAssert(m_out_mat_ov, m_out_mat_gapi, "Test output");
}

INSTANTIATE_TEST_CASE_P(Instantiation, TestMeanScaleOV,
                        Values(CV_8U, CV_32S, CV_16F, CV_32F));

} // namespace opencv_test

#endif // HAVE_INF_ENGINE && INF_ENGINE_RELEASE >= 2022010000
