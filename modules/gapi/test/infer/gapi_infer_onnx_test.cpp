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
        if (env_path) {
            cvtest::addDataSearchPath(env_path);
        }
    }
};
static ONNXInitPath g_init_path;

cv::Mat initMatrixRandU(const int type, const cv::Size& sz_in) {
    const cv::Mat in_mat1 = cv::Mat(sz_in, type);

    if (CV_MAT_DEPTH(type) < CV_32F) {
        cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));
    } else {
        const int fscale = 256;  // avoid bits near ULP, generate stable test input
        cv::Mat in_mat32s(in_mat1.size(), CV_MAKE_TYPE(CV_32S, CV_MAT_CN(type)));
        cv::randu(in_mat32s, cv::Scalar::all(0), cv::Scalar::all(255 * fscale));
        in_mat32s.convertTo(in_mat1, type, 1.0f / fscale, 0);
    }
    return in_mat1;
}
} // anonymous namespace
namespace opencv_test
{
namespace {
// FIXME: taken from the DNN module
void normAssert(const cv::InputArray& ref, const cv::InputArray& test,
                const char *comment /*= ""*/,
                const double l1 = 0.00001, const double lInf = 0.0001) {
    const double normL1 = cvtest::norm(ref, test, cv::NORM_L1) / ref.getMat().total();
    EXPECT_LE(normL1, l1) << comment;

    const double normInf = cvtest::norm(ref, test, cv::NORM_INF);
    EXPECT_LE(normInf, lInf) << comment;
}

inline std::string findModel(const std::string &model_name) {
    return findDataFile("vision/" + model_name + ".onnx", false);
}

inline void toCHW(const cv::Mat& src, cv::Mat& dst) {
    dst.create(cv::Size(src.cols, src.rows * src.channels()), CV_32F);
    std::vector<cv::Mat> planes;
    for (int i = 0; i < src.channels(); ++i) {
        planes.push_back(dst.rowRange(i * src.rows, (i + 1) * src.rows));
    }
    cv::split(src, planes);
}

inline int toCV(const ONNXTensorElementDataType prec) {
    switch (prec) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return CV_8U;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return CV_32F;
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

inline void copyToOut(const cv::Mat& in, cv::Mat& out) {
    GAPI_Assert(in.depth() == CV_32F);
    GAPI_Assert(in.size == out.size);
    const float* const inptr = in.ptr<float>();
    float* const optr = out.ptr<float>();
    const int size = in.total();
    for (int i = 0; i < size; ++i) {
        optr[i] = inptr[i];
    }
}

void remapYolo(const std::unordered_map<std::string, cv::Mat> &onnx,
                      std::unordered_map<std::string, cv::Mat> &gapi) {
    GAPI_Assert(onnx.size() == 1u);
    GAPI_Assert(gapi.size() == 1u);
    // Result from Run method
    const cv::Mat& in = onnx.begin()->second;
    // Configured output
    cv::Mat& out = gapi.begin()->second;
    // Simple copy
    copyToOut(in, out);
}

void remapSsdPorts(const std::unordered_map<std::string, cv::Mat> &onnx,
                           std::unordered_map<std::string, cv::Mat> &gapi) {
    // Result from Run method
    const cv::Mat& in_num     = onnx.at("num_detections:0");
    const cv::Mat& in_boxes   = onnx.at("detection_boxes:0");
    const cv::Mat& in_scores  = onnx.at("detection_scores:0");
    const cv::Mat& in_classes = onnx.at("detection_classes:0");
    // Configured outputs
    cv::Mat& out_boxes   = gapi.at("out1");
    cv::Mat& out_classes = gapi.at("out2");
    cv::Mat& out_scores  = gapi.at("out3");
    cv::Mat& out_num     = gapi.at("out4");
    // Simple copy for outputs
    copyToOut(in_num, out_num);
    copyToOut(in_boxes, out_boxes);
    copyToOut(in_scores, out_scores);
    copyToOut(in_classes, out_classes);
}

class ONNXtest : public ::testing::Test {
public:
    std::string model_path;
    size_t num_in, num_out;
    std::vector<cv::Mat> out_gapi;
    std::vector<cv::Mat> out_onnx;
    cv::Mat in_mat1;

    ONNXtest() {
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
        memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        out_gapi.resize(1);
        out_onnx.resize(1);
        // FIXME: All tests chek "random" image
        // Ideally it should be a real image
        in_mat1 = initMatrixRandU(CV_8UC3, cv::Size{640, 480});
    }

    template<typename T>
    void infer(const std::vector<cv::Mat>& ins,
                     std::vector<cv::Mat>& outs) {
        // Prepare session
        session = Ort::Session(env, model_path.data(), session_options);
        num_in = session.GetInputCount();
        num_out = session.GetOutputCount();
        GAPI_Assert(num_in == ins.size());
        in_node_names.clear();
        out_node_names.clear();
        // Inputs Run params
        std::vector<Ort::Value> in_tensors;
        for(size_t i = 0; i < num_in; ++i) {
            char* in_node_name_p = session.GetInputName(i, allocator);
            in_node_names.push_back(std::string(in_node_name_p));
            allocator.Free(in_node_name_p);
            in_node_dims = toORT(ins[i].size);
            in_tensors.emplace_back(Ort::Value::CreateTensor<T>(memory_info,
                                                                const_cast<T*>(ins[i].ptr<T>()),
                                                                ins[i].total(),
                                                                in_node_dims.data(),
                                                                in_node_dims.size()));
        }
        // Outputs Run params
        for(size_t i = 0; i < num_out; ++i) {
            char* out_node_name_p = session.GetOutputName(i, allocator);
            out_node_names.push_back(std::string(out_node_name_p));
            allocator.Free(out_node_name_p);
        }
        // Input/output order by names
        const auto in_run_names  = getCharNames(in_node_names);
        const auto out_run_names = getCharNames(out_node_names);
        // Run
        auto result = session.Run(Ort::RunOptions{nullptr},
                                  in_run_names.data(),
                                  &in_tensors.front(),
                                  num_in,
                                  out_run_names.data(),
                                  num_out);
        // Copy outputs
        GAPI_Assert(result.size() == num_out);
        outs.resize(num_out);
        for (size_t i = 0; i < num_out; ++i) {
            const auto info = result[i].GetTensorTypeAndShapeInfo();
            const auto shape = info.GetShape();
            const auto type = info.GetElementType();
            cv::Mat mt(std::vector<int>(shape.begin(), shape.end()), toCV(type),
                       reinterpret_cast<void*>(result[i].GetTensorMutableData<uint8_t*>()));
            mt.copyTo(outs[i]);
        }
    }
    // One input/output overload
    template<typename T>
    void infer(const cv::Mat& in, cv::Mat& out) {
        std::vector<cv::Mat> result;
        infer<T>({in}, result);
        GAPI_Assert(result.size() == 1u);
        out = result.front();
    }

    void validate() {
        GAPI_Assert(!out_gapi.empty() && !out_onnx.empty());
        ASSERT_EQ(out_gapi.size(), out_onnx.size());
        const auto size = out_gapi.size();
        for (size_t i = 0; i < size; ++i) {
            normAssert(out_onnx[i], out_gapi[i], "Test outputs");
        }
    }

    void useModel(const std::string& model_name) {
        model_path = findModel(model_name);
    }
private:
    Ort::Env env{nullptr};
    Ort::MemoryInfo memory_info{nullptr};
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::SessionOptions session_options;
    Ort::Session session{nullptr};

    std::vector<int64_t> in_node_dims;
    std::vector<std::string> in_node_names;
    std::vector<std::string> out_node_names;
};

class ONNXClassificationTest : public ONNXtest {
public:
    const cv::Scalar mean = { 0.485, 0.456, 0.406 };
    const cv::Scalar std  = { 0.229, 0.224, 0.225 };

    void preprocess(const cv::Mat& src, cv::Mat& dst) {
        const int new_h = 224;
        const int new_w = 224;
        cv::Mat tmp, cvt, rsz;
        cv::resize(src, rsz, cv::Size(new_w, new_h));
        rsz.convertTo(cvt, CV_32F, 1.f / 255);
        tmp = (cvt - mean) / std;
        toCHW(tmp, dst);
        dst = dst.reshape(1, {1, 3, new_h, new_w});
    }
};

class ONNXGRayScaleTest : public ONNXtest {
public:
    void preprocess(const cv::Mat& src, cv::Mat& dst) {
        const int new_h = 64;
        const int new_w = 64;
        cv::Mat cvc, rsz, cvt;
        cv::cvtColor(src, cvc, cv::COLOR_BGR2GRAY);
        cv::resize(cvc, rsz, cv::Size(new_w, new_h));
        rsz.convertTo(cvt, CV_32F);
        toCHW(cvt, dst);
        dst = dst.reshape(1, {1, 1, new_h, new_w});
    }
};
} // anonymous namespace

TEST_F(ONNXClassificationTest, Infer)
{
    useModel("classification/squeezenet/model/squeezenet1.0-9");
    // ONNX_API code
    cv::Mat processed_mat;
    preprocess(in_mat1, processed_mat);
    infer<float>(processed_mat, out_onnx.front());
    // G_API code
    G_API_NET(SqueezNet, <cv::GMat(cv::GMat)>, "squeeznet");
    cv::GMat in;
    cv::GMat out = cv::gapi::infer<SqueezNet>(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(out));
    // NOTE: We have to normalize U8 tensor
    // so cfgMeanStd() is here
    auto net = cv::gapi::onnx::Params<SqueezNet> { model_path }.cfgMeanStd({ mean }, { std });
    comp.apply(cv::gin(in_mat1),
               cv::gout(out_gapi.front()),
               cv::compile_args(cv::gapi::networks(net)));
    // Validate
    validate();
}

TEST_F(ONNXtest, InferTensor)
{
    useModel("classification/squeezenet/model/squeezenet1.0-9");
    // Create tensor
    // FIXME: Test cheks "random" image
    // Ideally it should be a real image
    const cv::Mat rand_mat = initMatrixRandU(CV_32FC3, cv::Size{224, 224});
    const std::vector<int> dims = {1, rand_mat.channels(), rand_mat.rows, rand_mat.cols};
    const cv::Mat tensor(dims, CV_32F, rand_mat.data);
    // ONNX_API code
    infer<float>(tensor, out_onnx.front());
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

TEST_F(ONNXClassificationTest, InferROI)
{
    useModel("classification/squeezenet/model/squeezenet1.0-9");
    const cv::Rect ROI(cv::Point{0, 0}, cv::Size{250, 250});
    // ONNX_API code
    cv::Mat roi_mat;
    preprocess(in_mat1(ROI), roi_mat);
    infer<float>(roi_mat, out_onnx.front());
    // G_API code
    G_API_NET(SqueezNet, <cv::GMat(cv::GMat)>, "squeeznet");
    cv::GMat in;
    cv::GOpaque<cv::Rect> rect;
    cv::GMat out = cv::gapi::infer<SqueezNet>(rect, in);
    cv::GComputation comp(cv::GIn(in, rect), cv::GOut(out));
    // NOTE: We have to normalize U8 tensor
    // so cfgMeanStd() is here
    auto net = cv::gapi::onnx::Params<SqueezNet> { model_path }.cfgMeanStd({ mean }, { std });
    comp.apply(cv::gin(in_mat1, ROI),
               cv::gout(out_gapi.front()),
               cv::compile_args(cv::gapi::networks(net)));
    // Validate
    validate();
}

TEST_F(ONNXClassificationTest, InferROIList)
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
        infer<float>(roi_mat, out_onnx[i]);
    }
    // G_API code
    G_API_NET(SqueezNet, <cv::GMat(cv::GMat)>, "squeeznet");
    cv::GMat in;
    cv::GArray<cv::Rect> rr;
    cv::GArray<cv::GMat> out = cv::gapi::infer<SqueezNet>(rr, in);
    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(out));
    // NOTE: We have to normalize U8 tensor
    // so cfgMeanStd() is here
    auto net = cv::gapi::onnx::Params<SqueezNet> { model_path }.cfgMeanStd({ mean }, { std });
    comp.apply(cv::gin(in_mat1, rois),
               cv::gout(out_gapi),
               cv::compile_args(cv::gapi::networks(net)));
    // Validate
    validate();
}

TEST_F(ONNXClassificationTest, Infer2ROIList)
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
        infer<float>(roi_mat, out_onnx[i]);
    }
    // G_API code
    G_API_NET(SqueezNet, <cv::GMat(cv::GMat)>, "squeeznet");
    cv::GMat in;
    cv::GArray<cv::Rect> rr;
    cv::GArray<cv::GMat> out = cv::gapi::infer2<SqueezNet>(in, rr);
    cv::GComputation comp(cv::GIn(in, rr), cv::GOut(out));
    // NOTE: We have to normalize U8 tensor
    // so cfgMeanStd() is here
    auto net = cv::gapi::onnx::Params<SqueezNet> { model_path }.cfgMeanStd({ mean }, { std });
    comp.apply(cv::gin(in_mat1, rois),
               cv::gout(out_gapi),
               cv::compile_args(cv::gapi::networks(net)));
    // Validate
    validate();
}

TEST_F(ONNXtest, InferDynamicInputTensor)
{
    useModel("object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8");
    // Create tensor
    // FIXME: Test cheks "random" image
    // Ideally it should be a real image
    const cv::Mat rand_mat = initMatrixRandU(CV_32FC3, cv::Size{416, 416});
    const std::vector<int> dims = {1, rand_mat.channels(), rand_mat.rows, rand_mat.cols};
    cv::Mat tensor(dims, CV_32F, rand_mat.data);
    const cv::Mat in_tensor = tensor / 255.f;
    // ONNX_API code
    infer<float>(in_tensor, out_onnx.front());
    // G_API code
    G_API_NET(YoloNet, <cv::GMat(cv::GMat)>, "YoloNet");
    cv::GMat in;
    cv::GMat out = cv::gapi::infer<YoloNet>(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(out));
    auto net = cv::gapi::onnx::Params<YoloNet>{model_path}
        .cfgPostProc({cv::GMatDesc{CV_32F, {1, 125, 13, 13}}}, remapYolo)
        .cfgOutputLayers({"out"});
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
    infer<float>(prep_mat, out_onnx.front());
    // G_API code
    G_API_NET(EmotionNet, <cv::GMat(cv::GMat)>, "emotion-ferplus");
    cv::GMat in;
    cv::GMat out = cv::gapi::infer<EmotionNet>(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(out));
    auto net = cv::gapi::onnx::Params<EmotionNet> { model_path }
        .cfgNormalize({ false }); // model accepts 0..255 range in FP32;
    comp.apply(cv::gin(in_mat1),
               cv::gout(out_gapi.front()),
               cv::compile_args(cv::gapi::networks(net)));
    // Validate
    validate();
}

TEST_F(ONNXtest, InferMultOutput)
{
    useModel("object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10");
    // ONNX_API code
    const auto prep_mat = in_mat1.reshape(1, {1, in_mat1.rows, in_mat1.cols, in_mat1.channels()});
    infer<uint8_t>({prep_mat}, out_onnx);
    // G_API code
    using SSDOut = std::tuple<cv::GMat, cv::GMat, cv::GMat, cv::GMat>;
    G_API_NET(MobileNet, <SSDOut(cv::GMat)>, "ssd_mobilenet");
    cv::GMat in;
    cv::GMat out1, out2, out3, out4;
    std::tie(out1, out2, out3, out4) = cv::gapi::infer<MobileNet>(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(out1, out2, out3, out4));
    auto net = cv::gapi::onnx::Params<MobileNet>{model_path}
        .cfgOutputLayers({"out1", "out2", "out3", "out4"})
        .cfgPostProc({cv::GMatDesc{CV_32F, {1, 100, 4}},
                      cv::GMatDesc{CV_32F, {1, 100}},
                      cv::GMatDesc{CV_32F, {1, 100}},
                      cv::GMatDesc{CV_32F, {1, 1}}}, remapSsdPorts);
    out_gapi.resize(num_out);
    comp.apply(cv::gin(in_mat1),
               cv::gout(out_gapi[0], out_gapi[1], out_gapi[2], out_gapi[3]),
               cv::compile_args(cv::gapi::networks(net)));
    // Validate
    validate();
}
} // namespace opencv_test

#endif //  HAVE_ONNX
