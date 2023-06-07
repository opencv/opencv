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

static const std::string SUBDIR = "intel/age-gender-recognition-retail-0013/FP32/";

void copyFromOV(ov::Tensor &tensor, cv::Mat &mat) {
    GAPI_Assert(tensor.get_byte_size() == mat.total() * mat.elemSize());
    std::copy_n(reinterpret_cast<uint8_t*>(tensor.data()),
                tensor.get_byte_size(),
                mat.ptr<uint8_t>());
}

void copyToOV(const cv::Mat &mat, ov::Tensor &tensor) {
    GAPI_Assert(tensor.get_byte_size() == mat.total() * mat.elemSize());
    std::copy_n(mat.ptr<uint8_t>(),
                tensor.get_byte_size(),
                reinterpret_cast<uint8_t*>(tensor.data()));
}

// FIXME: taken from the DNN module
void normAssert(cv::InputArray ref, cv::InputArray test,
                const char *comment /*= ""*/,
                double l1 = 0.00001, double lInf = 0.0001) {
    double normL1 = cvtest::norm(ref, test, cv::NORM_L1) / ref.getMat().total();
    EXPECT_LE(normL1, l1) << comment;

    double normInf = cvtest::norm(ref, test, cv::NORM_INF);
    EXPECT_LE(normInf, lInf) << comment;
}

ov::Core getCore() {
    static ov::Core core;
    return core;
}

// TODO: AGNetGenComp, AGNetTypedComp, AGNetOVComp, AGNetOVCompiled
// can be generalized to work with any model and used as parameters for tests.

struct AGNetGenComp {
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

struct AGNetTypedComp {
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

    static cv::GComputation create() {
        cv::GMat in;
        cv::GMat age, gender;
        std::tie(age, gender) = cv::gapi::infer<AgeGender>(in);
        return cv::GComputation{cv::GIn(in), cv::GOut(age, gender)};
    }
};

class AGNetOVCompiled {
public:
    AGNetOVCompiled(ov::CompiledModel &&compiled_model)
        : m_compiled_model(std::move(compiled_model)) {
    }

    void operator()(const cv::Mat &in_mat,
                          cv::Mat &age_mat,
                          cv::Mat &gender_mat) {
        auto infer_request = m_compiled_model.create_infer_request();
        auto input_tensor   = infer_request.get_input_tensor();
        copyToOV(in_mat, input_tensor);

        infer_request.infer();

        auto age_tensor = infer_request.get_tensor("age_conv3");
        age_mat.create(cv::gapi::ov::util::to_ocv(age_tensor.get_shape()),
                       cv::gapi::ov::util::to_ocv(age_tensor.get_element_type()));
        copyFromOV(age_tensor, age_mat);

        auto gender_tensor = infer_request.get_tensor("prob");
        gender_mat.create(cv::gapi::ov::util::to_ocv(gender_tensor.get_shape()),
                          cv::gapi::ov::util::to_ocv(gender_tensor.get_element_type()));
        copyFromOV(gender_tensor, gender_mat);
    }

    void export_model(const std::string &outpath) {
        std::ofstream file{outpath, std::ios::out | std::ios::binary};
        GAPI_Assert(file.is_open());
        m_compiled_model.export_model(file);
    }

private:
    ov::CompiledModel m_compiled_model;
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
        m_model = getCore().read_model(xml_path, bin_path);
    }

    using PrePostProcessF = std::function<void(ov::preprocess::PrePostProcessor&)>;

    void cfgPrePostProcessing(PrePostProcessF f) {
        ov::preprocess::PrePostProcessor ppp(m_model);
        f(ppp);
        m_model = ppp.build();
    }

    AGNetOVCompiled compile() {
        auto compiled_model = getCore().compile_model(m_model, m_device);
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

} // anonymous namespace

// TODO: Make all of tests below parmetrized to avoid code duplication
TEST(TestAgeGenderOV, InferTypedTensor) {
    initDLDTDataPath();
    const std::string xml_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    const std::string bin_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    const std::string device   = "CPU";

    cv::Mat in_mat({1, 3, 62, 62}, CV_32F);
    cv::randu(in_mat, -1, 1);
    cv::Mat ov_age, ov_gender, gapi_age, gapi_gender;

    // OpenVINO
    AGNetOVComp ref(xml_path, bin_path, device);
    ref.apply(in_mat, ov_age, ov_gender);

    // G-API
    auto comp = AGNetTypedComp::create();
    auto pp   = AGNetTypedComp::params(xml_path, bin_path, device);
    comp.apply(cv::gin(in_mat), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    // Assert
    normAssert(ov_age,    gapi_age,    "Test age output"   );
    normAssert(ov_gender, gapi_gender, "Test gender output");
}

TEST(TestAgeGenderOV, InferTypedImage) {
    initDLDTDataPath();
    const std::string xml_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    const std::string bin_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    const std::string device   = "CPU";

    cv::Mat in_mat(300, 300, CV_8UC3);
    cv::randu(in_mat, 0, 255);
    cv::Mat ov_age, ov_gender, gapi_age, gapi_gender;

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
    normAssert(ov_age,    gapi_age,    "Test age output"   );
    normAssert(ov_gender, gapi_gender, "Test gender output");
}

TEST(TestAgeGenderOV, InferGenericTensor) {
    initDLDTDataPath();
    const std::string xml_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    const std::string bin_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    const std::string device   = "CPU";

    cv::Mat in_mat({1, 3, 62, 62}, CV_32F);
    cv::randu(in_mat, -1, 1);
    cv::Mat ov_age, ov_gender, gapi_age, gapi_gender;

    // OpenVINO
    AGNetOVComp ref(xml_path, bin_path, device);
    ref.apply(in_mat, ov_age, ov_gender);

    // G-API
    auto comp = AGNetGenComp::create();
    auto pp   = AGNetGenComp::params(xml_path, bin_path, device);
    comp.apply(cv::gin(in_mat), cv::gout(gapi_age, gapi_gender),
               cv::compile_args(cv::gapi::networks(pp)));

    // Assert
    normAssert(ov_age,    gapi_age,    "Test age output"   );
    normAssert(ov_gender, gapi_gender, "Test gender output");
}

TEST(TestAgeGenderOV, InferGenericImage) {
    initDLDTDataPath();
    const std::string xml_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    const std::string bin_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    const std::string device   = "CPU";

    cv::Mat in_mat(300, 300, CV_8UC3);
    cv::randu(in_mat, 0, 255);
    cv::Mat ov_age, ov_gender, gapi_age, gapi_gender;

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
    normAssert(ov_age,    gapi_age,    "Test age output"   );
    normAssert(ov_gender, gapi_gender, "Test gender output");
}

TEST(TestAgeGenderOV, InferGenericImageBlob) {
    initDLDTDataPath();
    const std::string xml_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    const std::string bin_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    const std::string blob_path = "age-gender-recognition-retail-0013.blob";
    const std::string device   = "CPU";

    cv::Mat in_mat(300, 300, CV_8UC3);
    cv::randu(in_mat, 0, 255);
    cv::Mat ov_age, ov_gender, gapi_age, gapi_gender;

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
    normAssert(ov_age,    gapi_age,    "Test age output"   );
    normAssert(ov_gender, gapi_gender, "Test gender output");
}

TEST(TestAgeGenderOV, InferGenericTensorBlob) {
    initDLDTDataPath();
    const std::string xml_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    const std::string bin_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    const std::string blob_path = "age-gender-recognition-retail-0013.blob";
    const std::string device   = "CPU";

    cv::Mat in_mat({1, 3, 62, 62}, CV_32F);
    cv::randu(in_mat, -1, 1);
    cv::Mat ov_age, ov_gender, gapi_age, gapi_gender;

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
    normAssert(ov_age,    gapi_age,    "Test age output"   );
    normAssert(ov_gender, gapi_gender, "Test gender output");
}

TEST(TestAgeGenderOV, InferBothOutputsFP16) {
    initDLDTDataPath();
    const std::string xml_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    const std::string bin_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    const std::string device   = "CPU";

    cv::Mat in_mat({1, 3, 62, 62}, CV_32F);
    cv::randu(in_mat, -1, 1);
    cv::Mat ov_age, ov_gender, gapi_age, gapi_gender;

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
    normAssert(ov_age,    gapi_age,    "Test age output"   );
    normAssert(ov_gender, gapi_gender, "Test gender output");
}

TEST(TestAgeGenderOV, InferOneOutputFP16) {
    initDLDTDataPath();
    const std::string xml_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    const std::string bin_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    const std::string device   = "CPU";

    cv::Mat in_mat({1, 3, 62, 62}, CV_32F);
    cv::randu(in_mat, -1, 1);
    cv::Mat ov_age, ov_gender, gapi_age, gapi_gender;

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
    normAssert(ov_age,    gapi_age,    "Test age output"   );
    normAssert(ov_gender, gapi_gender, "Test gender output");
}

TEST(TestAgeGenderOV, ThrowCfgOutputPrecForBlob) {
    initDLDTDataPath();
    const std::string xml_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    const std::string bin_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    const std::string blob_path = "age-gender-recognition-retail-0013.blob";
    const std::string device   = "CPU";

    // OpenVINO (Just for blob compilation)
    AGNetOVComp ref(xml_path, bin_path, device);
    auto cc_ref = ref.compile();
    cc_ref.export_model(blob_path);

    // G-API
    auto comp = AGNetGenComp::create();
    auto pp   = AGNetGenComp::params(blob_path, device);

    EXPECT_ANY_THROW(pp.cfgOutputTensorPrecision(CV_16F));
}

TEST(TestAgeGenderOV, ThrowInvalidConfigIR) {
    initDLDTDataPath();
    const std::string xml_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    const std::string bin_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    const std::string device   = "CPU";

    // G-API
    auto comp = AGNetGenComp::create();
    auto pp   = AGNetGenComp::params(xml_path, bin_path, device);
    pp.cfgPluginConfig({{"some_key", "some_value"}});

    EXPECT_ANY_THROW(comp.compile(cv::GMatDesc{CV_8U,3,cv::Size{320, 240}},
                                  cv::compile_args(cv::gapi::networks(pp))));
}

TEST(TestAgeGenderOV, ThrowInvalidConfigBlob) {
    initDLDTDataPath();
    const std::string xml_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    const std::string bin_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    const std::string blob_path = "age-gender-recognition-retail-0013.blob";
    const std::string device   = "CPU";

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

TEST(TestAgeGenderOV, ThrowInvalidImageLayout) {
    initDLDTDataPath();
    const std::string xml_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    const std::string bin_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    const std::string device   = "CPU";

    // NB: This mat may only have "NHWC" layout.
    cv::Mat in_mat(300, 300, CV_8UC3);
    cv::randu(in_mat, 0, 255);
    cv::Mat gender, gapi_age, gapi_gender;
    auto comp = AGNetTypedComp::create();
    auto pp = AGNetTypedComp::params(xml_path, bin_path, device);

    pp.cfgInputTensorLayout("NCHW");

    EXPECT_ANY_THROW(comp.compile(cv::descr_of(in_mat),
                     cv::compile_args(cv::gapi::networks(pp))));
}

TEST(TestAgeGenderOV, InferTensorWithPreproc) {
    initDLDTDataPath();
    const std::string xml_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.xml");
    const std::string bin_path = findDataFile(SUBDIR + "age-gender-recognition-retail-0013.bin");
    const std::string device   = "CPU";

    cv::Mat in_mat({1, 240, 320, 3}, CV_32F);
    cv::randu(in_mat, -1, 1);
    cv::Mat ov_age, ov_gender, gapi_age, gapi_gender;

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
    normAssert(ov_age,    gapi_age,    "Test age output"   );
    normAssert(ov_gender, gapi_gender, "Test gender output");
}

} // namespace opencv_test

#endif // HAVE_INF_ENGINE && INF_ENGINE_RELEASE >= 2022010000
