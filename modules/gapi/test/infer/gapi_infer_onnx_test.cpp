// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "../test_precomp.hpp"

#ifdef HAVE_ONNX

#include <stdexcept>
#include <onnxruntime_cxx_api.h>
#include <ade/util/iota_range.hpp>

#include <opencv2/gapi/own/convert.hpp>
#include <opencv2/gapi/infer/onnx.hpp>

namespace {

struct ONNXInitPath {
    ONNXInitPath() {
        const char* env_path = getenv("OPENCV_GAPI_ONNX_MODEL_PATH");
        if (env_path)
            cvtest::addDataSearchPath(env_path);
    }
};
static ONNXInitPath g_init_path;

cv::Mat initMatrixRandU(int type, cv::Size sz_in)
{
    cv::Mat in_mat1 = cv::Mat(sz_in, type);

    if (CV_MAT_DEPTH(type) < CV_32F)
    {
        cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));
    }
    else
    {
        const int fscale = 256;  // avoid bits near ULP, generate stable test input
        cv::Mat in_mat32s(in_mat1.size(), CV_MAKE_TYPE(CV_32S, CV_MAT_CN(type)));
        cv::randu(in_mat32s, cv::Scalar::all(0), cv::Scalar::all(255 * fscale));
        in_mat32s.convertTo(in_mat1, type, 1.0f / fscale, 0);
    }
    return in_mat1;
}
}
namespace opencv_test
{
namespace {
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

std::string findModel(const std::string &model_name)
{
    return findDataFile("vision/" + model_name + ".onnx", false);
}

void remap_yolo(const std::unordered_map<std::string, cv::Mat> &onnx,
                       std::unordered_map<std::string, cv::Mat> &gapi) {
    GAPI_Assert(onnx.size() == 1u);
    GAPI_Assert(gapi.size() == 1u);
    const auto size = onnx.begin()->second.total();
    GAPI_Assert(onnx.begin()->second.size == gapi.begin()->second.size);
    const float* onnx_ptr = onnx.begin()->second.ptr<float>();
          float* gapi_ptr = gapi.begin()->second.ptr<float>();

    // Simple copy. Same sizes.
    for (size_t i = 0; i < size; ++i) {
       gapi_ptr[i] = onnx_ptr[i];
    }
}

void toCHW(cv::Mat& src, cv::Mat& dst, int new_h, int new_w, int new_c) {
    dst.create(cv::Size(new_w, new_h * new_c), CV_32F);
    std::vector<cv::Mat> planes;
    for (int i = 0; i < new_c; ++i) {
        planes.push_back(dst.rowRange(i * new_h, (i + 1) * new_h));
    }
    cv::split(src, planes);
}

inline int toCV(ONNXTensorElementDataType prec) {
    switch (prec) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return CV_8U;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return CV_32F;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return CV_32S;
    default: GAPI_Assert(false && "Unsupported data type");
    }
    return -1;
}

inline std::vector<int64_t> toORT(const cv::MatSize &sz) {
    return cv::to_own<int64_t>(sz);
}

inline std::vector<const char*> getCharNames(const std::vector<std::string>& names) {
    std::vector<const char*> out_vec;
    for (const auto& el : names) {
            out_vec.push_back(el.data());
    }
    return out_vec;
}

class ONNXtest : public ::testing::Test {
public:
    Ort::Env env{nullptr};
    Ort::MemoryInfo memory_info{nullptr};
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::SessionOptions session_options;
    Ort::Session session{nullptr};

    std::string model_path;
    cv::Mat in_mat1;
    std::vector<cv::Mat> out_gapi;
    std::vector<cv::Mat> out_onnx;

    std::vector<int64_t> output_node_dims;
    std::vector<int64_t> input_node_dims;
    std::vector<std::string> in_node_names;
    std::vector<std::string> out_node_names;

    ONNXtest() {
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
        memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        out_gapi.resize(1);
        out_onnx.resize(1);
        in_mat1 = initMatrixRandU(CV_8UC3, cv::Size{640, 480});
    }

    void validate() {
        GAPI_Assert(!out_gapi.empty() && !out_onnx.empty());
        ASSERT_EQ(out_gapi.size(), out_onnx.size());
        auto size = out_gapi.size();
        for (size_t i = 0; i < size; ++i) {
            normAssert(out_onnx[i], out_gapi[i], "Test output");
        }
    }

    void useModel(const std::string& model_name) {
        model_path = findModel(model_name);
    }
};

class ONNXSimpleTest : public ONNXtest {
public:
    cv::Scalar mean, std;

    virtual void SetUp() {
        mean = { 0.485, 0.456, 0.406 };
        std = { 0.229, 0.224, 0.225 };
    }

    virtual void prepareInfer() {
        session = Ort::Session(env, model_path.data(), session_options);
        char* in_node_name_p = session.GetInputName(0, allocator);
        char* out_node_name_p = session.GetOutputName(0, allocator);
        in_node_names = {std::string(in_node_name_p)};
        out_node_names = {std::string(out_node_name_p)};
        allocator.Free(in_node_name_p);
        allocator.Free(out_node_name_p);
    }

    void infer(const cv::Mat& in, cv::Mat& out) {
        prepareInfer();
        input_node_dims.clear();
        for (int i = 0; i < in.size.dims(); ++i) {
            input_node_dims.push_back(in.size[i]);
        }
        auto in_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                         const_cast<float*>(in.ptr<float>()),
                                                         in.total(),
                                                         input_node_dims.data(),
                                                         input_node_dims.size());

        std::vector<const char *> in_names = {in_node_names[0].data()};
        std::vector<const char *> out_names = {out_node_names[0].data()};
        auto result = session.Run(Ort::RunOptions{nullptr},
                                  in_names.data(),
                                  &in_tensor,
                                  session.GetInputCount(),
                                  out_names.data(),
                                  session.GetOutputCount());

        auto info = result.front().GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        auto type = info.GetElementType();
        cv::Mat mt(std::vector<int>(shape.begin(), shape.end()), toCV(type),
                   reinterpret_cast<void*>(result.front().GetTensorMutableData<uint8_t*>()));
        mt.copyTo(out);
    }

    virtual void preprocess(const cv::Mat& src, cv::Mat& dst) {
        const int new_h = 224;
        const int new_w = 224;
        cv::Mat tmp, nmat, cvt;
        cv::resize(src, dst, cv::Size(new_w, new_h));
        dst.convertTo(cvt, CV_32F, 1.f / 255);
        nmat = cvt - mean;
        tmp = nmat / std;
        toCHW(tmp, dst, new_h, new_w, 3);
        dst = dst.reshape(1, {1, 3, new_h, new_w});
    }
};

class ONNXGRayScaleTest : public ONNXSimpleTest {
public:
    virtual void preprocess(const cv::Mat& src, cv::Mat& dst) {
        const int new_h = 64;
        const int new_w = 64;
        cv::Mat csc, rsc, cvt;
        cv::cvtColor(src, csc, cv::COLOR_BGR2GRAY);
        cv::resize(csc, rsc, cv::Size(new_w, new_h));
        rsc.convertTo(cvt, CV_32F);
        toCHW(cvt, dst, new_h, new_w, 1);
        dst = dst.reshape(1, {1, 1, new_h, new_w});
    }
};

} // anonymous namespace

TEST_F(ONNXSimpleTest, Infer)
{
    useModel("classification/squeezenet/model/squeezenet1.0-9");
    // ONNX_API code
    cv::Mat processed_mat;
    preprocess(in_mat1, processed_mat);
    infer(processed_mat, out_onnx.front());
    // G_API code
    G_API_NET(SqueezNet, <cv::GMat(cv::GMat)>, "squeeznet");
    cv::GMat in;
    cv::GMat out = cv::gapi::infer<SqueezNet>(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(out));
    auto net = cv::gapi::onnx::Params<SqueezNet> { model_path }.cfgMeanStd({mean},{std});
    comp.apply(cv::gin(in_mat1),
               cv::gout(out_gapi.front()),
               cv::compile_args(cv::gapi::networks(net)));
    // Validate
    validate();
}

TEST_F(ONNXSimpleTest, InferTensor)
{
    useModel("classification/squeezenet/model/squeezenet1.0-9");
    // Create tensor
    const cv::Mat rand_mat = initMatrixRandU(CV_32FC3, cv::Size{224, 224});
    const std::vector<int> dims = {1, rand_mat.channels(), rand_mat.rows, rand_mat.cols};
    const cv::Mat tensor(dims, CV_32F, rand_mat.data);
    // ONNX_API code
    infer(tensor, out_onnx.front());
    // G_API code
    G_API_NET(SqueezNet, <cv::GMat(cv::GMat)>, "squeeznet");
    cv::GMat in;
    cv::GMat out = cv::gapi::infer<SqueezNet>(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(out));

    auto net = cv::gapi::onnx::Params<SqueezNet> { model_path };
    comp.apply(cv::gin(tensor),
               cv::gout(out_gapi.front()),
               cv::compile_args(cv::gapi::networks(net)));
    // Validate
    validate();
}

TEST_F(ONNXSimpleTest, InferROI)
{
    useModel("classification/squeezenet/model/squeezenet1.0-9");
    const cv::Rect ROI(cv::Point{0, 0}, cv::Size{250, 250});
    // ONNX_API code
    cv::Mat roi_mat;
    preprocess(in_mat1(ROI), roi_mat);
    infer(roi_mat, out_onnx.front());
    // G_API code
    G_API_NET(SqueezNet, <cv::GMat(cv::GMat)>, "squeeznet");
    cv::GMat in;
    cv::GOpaque<cv::Rect> rect;
    cv::GMat out = cv::gapi::infer<SqueezNet>(rect, in);
    cv::GComputation comp(cv::GIn(in, rect), cv::GOut(out));
    auto net = cv::gapi::onnx::Params<SqueezNet> { model_path }.cfgMeanStd({mean},{std});
    comp.apply(cv::gin(in_mat1(ROI), ROI),
               cv::gout(out_gapi.front()),
               cv::compile_args(cv::gapi::networks(net)));
    // Validate
    validate();
}

TEST_F(ONNXSimpleTest, InferROIList)
{
    useModel("classification/squeezenet/model/squeezenet1.0-9");
    const std::vector<cv::Rect> rois = {
        cv::Rect(cv::Point{ 0,   0}, cv::Size{80, 120}),
        cv::Rect(cv::Point{50, 100}, cv::Size{250, 360}),
    };
    // ONNX_API code
    out_onnx.resize(rois.size());
    for (size_t i = 0; i < rois.size(); ++i) {
        cv::Mat roi_mat;
        preprocess(in_mat1(rois[i]), roi_mat);
        infer(roi_mat, out_onnx[i]);
    }
    // G_API code
    G_API_NET(SqueezNet, <cv::GMat(cv::GMat)>, "squeeznet");
    cv::GMat in;
    cv::GArray<cv::Rect> rr;
    cv::GArray<cv::GMat> out = cv::gapi::infer<SqueezNet>(rr, in);
    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(out));
    auto net = cv::gapi::onnx::Params<SqueezNet> { model_path }.cfgMeanStd({mean},{std});
    comp.apply(cv::gin(in_mat1, rois),
               cv::gout(out_gapi),
               cv::compile_args(cv::gapi::networks(net)));
    // Validate
    validate();
}

TEST_F(ONNXSimpleTest, Infer2ROIList)
{
    useModel("classification/squeezenet/model/squeezenet1.0-9");
    const std::vector<cv::Rect> rois = {
        cv::Rect(cv::Point{ 0,   0}, cv::Size{80, 120}),
        cv::Rect(cv::Point{50, 100}, cv::Size{250, 360}),
    };
    // ONNX_API code
    out_onnx.resize(rois.size());
    for (size_t i = 0; i < rois.size(); ++i) {
        cv::Mat roi_mat;
        preprocess(in_mat1(rois[i]), roi_mat);
        infer(roi_mat, out_onnx[i]);
    }
    // G_API code
    G_API_NET(SqueezNet, <cv::GMat(cv::GMat)>, "squeeznet");
    cv::GMat in;
    cv::GArray<cv::Rect> rr;
    cv::GArray<cv::GMat> out = cv::gapi::infer2<SqueezNet>(in,rr);
    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(out));
    auto net = cv::gapi::onnx::Params<SqueezNet> { model_path }.cfgMeanStd({mean},{std});
    comp.apply(cv::gin(in_mat1, rois),
               cv::gout(out_gapi),
               cv::compile_args(cv::gapi::networks(net)));

    // Validate
    validate();
}

TEST_F(ONNXSimpleTest, InferDynamicInputTensor)
{
    useModel("object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8");
    // Create tensor
    const cv::Mat rand_mat = initMatrixRandU(CV_32FC3, cv::Size{416, 416});
    const std::vector<int> dims = {1, rand_mat.channels(), rand_mat.rows, rand_mat.cols};
    cv::Mat tensor(dims, CV_32F, rand_mat.data);
    const cv::Mat in_tensor = tensor / 255.f;
    // ONNX_API code
    infer(in_tensor, out_onnx.front());
    // G_API code
    G_API_NET(YoloNet, <cv::GMat(cv::GMat)>, "YoloNet");
    auto net = cv::gapi::onnx::Params<YoloNet>{model_path}
        .cfgPostProc({cv::GMatDesc{CV_32F, {1,125,13,13}}}, remap_yolo)
        .cfgOutputLayers({"out"});

    cv::GMat in;
    cv::GMat out = cv::gapi::infer<YoloNet>(in);

    cv::GComputation comp(cv::GIn(in), cv::GOut(out));
    comp.apply(cv::gin(in_tensor),
               cv::gout(out_gapi.front()),
               cv::compile_args(cv::gapi::networks(net)));
    // Validate
    validate();
}

TEST_F(ONNXGRayScaleTest, InferImage)
{
    useModel("body_analysis/emotion_ferplus/model/emotion-ferplus-8");
    // ONNX_API code
    cv::Mat prep_mat;
    preprocess(in_mat1, prep_mat);
    infer(prep_mat, out_onnx.front());
    // G_API code
    G_API_NET(EmotionNet, <cv::GMat(cv::GMat)>, "emotion-ferplus");
    cv::GMat in;
    cv::GMat out = cv::gapi::infer<EmotionNet>(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(out));
    auto net = cv::gapi::onnx::Params<EmotionNet> { model_path }
    .cfgNormalize({false}); // model accepts 0..255 range in FP32;
    comp.apply(cv::gin(in_mat1),
               cv::gout(out_gapi.front()),
               cv::compile_args(cv::gapi::networks(net)));
    // Validate
    validate();
}

} // namespace opencv_test

#endif //  HAVE_ONNX
