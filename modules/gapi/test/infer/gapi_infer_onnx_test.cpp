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
    return findDataFile("vision/classification/squeezenet/model/" + model_name + ".onnx", false);
}

inline void preprocess(const cv::Mat& src,
                             cv::Mat& dst,
                       const cv::Scalar& mean,
                       const cv::Scalar& std) {
    int new_h = 224;
    int new_w = 224;
    cv::Mat tmp, nmat, cvt;
    cv::resize(src, dst, cv::Size(new_w, new_h));
    dst.convertTo(cvt, CV_32F, 1.f / 255);
    nmat = cvt - mean;
    tmp = nmat / std;
    dst.create(cv::Size(new_w, new_h * src.channels()), CV_32F);
    std::vector<cv::Mat> planes;
    for (int i = 0; i < src.channels(); ++i) {
        planes.push_back(dst.rowRange(i * new_h, (i + 1) * new_h));
    }
    cv::split(tmp, planes);
}

void InferONNX(const std::string& model_path,
               const cv::Mat& in,
                     cv::Mat& out,
               const cv::Scalar& mean,
               const cv::Scalar& std)
{
    // FIXME: It must be a FIXTURE test!
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path.data(), session_options);
    auto input_node_dims = //    0 - one input
        session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto output_node_dims = //    0 - one output
        session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    Ort::AllocatorWithDefaultOptions allocator;
    char* in_node_name_p = session.GetInputName(0, allocator);
    char* out_node_name_p = session.GetOutputName(0, allocator);
    std::string in_node_name(in_node_name_p);
    std::string out_node_name(out_node_name_p);
    allocator.Free(in_node_name_p);
    allocator.Free(out_node_name_p);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    cv::Mat dst;
    preprocess(in, dst, mean, std);

    out.create(std::vector<int>(output_node_dims.begin(),
                                output_node_dims.end()), CV_32F); // empty output Mat
    auto in_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                     dst.ptr<float>(),
                                                     dst.total(),
                                                     input_node_dims.data(),
                                                     input_node_dims.size());
    auto out_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                      out.ptr<float>(),
                                                      out.total(),
                                                      output_node_dims.data(),
                                                      output_node_dims.size());
    std::vector<const char *> in_names = {in_node_name.data()};
    std::vector<const char *> out_names = {out_node_name.data()};
    session.Run(Ort::RunOptions{nullptr},
                in_names.data(),
                &in_tensor,
                session.GetInputCount(),
                out_names.data(),
                &out_tensor,
                session.GetOutputCount());
}

} // anonymous namespace

TEST(ONNX, Infer)
{
    cv::Mat in_mat1, out_gapi, out_onnx;
    std::string model_path = findModel("squeezenet1.0-9");
    // NOTE: All tests chek "random" image
    // Ideally it should be a real image
    in_mat1 = initMatrixRandU(CV_8UC3, cv::Size{640, 480});

    cv::Scalar mean = { 0.485, 0.456, 0.406 };
    cv::Scalar std  = { 0.229, 0.224, 0.225 };

    // ONNX_API code
    InferONNX(model_path, in_mat1, out_onnx, mean, std);

    // G_API code
    G_API_NET(SqueezNet, <cv::GMat(cv::GMat)>, "squeeznet");
    cv::GMat in;
    cv::GMat out = cv::gapi::infer<SqueezNet>(in);
    cv::GComputation comp(cv::GIn(in), cv::GOut(out));
    // NOTE: We have to normalize U8 tensor
    // so cfgMeanStd() is here
    auto net = cv::gapi::onnx::Params<SqueezNet> { model_path }.cfgMeanStd({mean},{std});
    comp.apply(cv::gin(in_mat1),
               cv::gout(out_gapi),
               cv::compile_args(cv::gapi::networks(net)));

    // Validate
    ASSERT_EQ(1000u, out_onnx.total());
    ASSERT_EQ(1000u, out_gapi.total());
    normAssert(out_onnx, out_gapi, "Test classification output");
}

TEST(ONNX, InferROI)
{
    cv::Mat in_mat1, out_gapi, out_onnx;
    std::string model_path = findModel("squeezenet1.0-9");
    in_mat1 = initMatrixRandU(CV_8UC3, cv::Size{640, 480});

    cv::Scalar mean = { 0.485, 0.456, 0.406 }; // squeeznet mean
    cv::Scalar std  = { 0.229, 0.224, 0.225 }; // squeeznet std

    cv::Rect ROI(cv::Point{0, 0}, cv::Size{250, 250});
    // ONNX_API code
    InferONNX(model_path, in_mat1(ROI), out_onnx, mean, std);

    // G_API code
    G_API_NET(SqueezNet, <cv::GMat(cv::GMat)>, "squeeznet");
    cv::GMat in;
    cv::GOpaque<cv::Rect> rect;
    cv::GMat out = cv::gapi::infer<SqueezNet>(rect, in);
    cv::GComputation comp(cv::GIn(in, rect), cv::GOut(out));
    auto net = cv::gapi::onnx::Params<SqueezNet> { model_path }.cfgMeanStd({mean},{std});
    comp.apply(cv::gin(in_mat1, ROI),
               cv::gout(out_gapi),
               cv::compile_args(cv::gapi::networks(net)));

    // Validate
    ASSERT_EQ(1000u, out_onnx.total());
    ASSERT_EQ(1000u, out_gapi.total());
    normAssert(out_onnx, out_gapi, "Test classification output");
}

TEST(ONNX, InferROIList)
{
    cv::Mat in_mat1;
    std::string model_path = findModel("squeezenet1.0-9");
    in_mat1 = initMatrixRandU(CV_8UC3, cv::Size{640, 480});

    cv::Scalar mean = { 0.485, 0.456, 0.406 }; // squeeznet mean
    cv::Scalar std  = { 0.229, 0.224, 0.225 }; // squeeznet std

    std::vector<cv::Rect> rois = {
        cv::Rect(cv::Point{ 0,   0}, cv::Size{80, 120}),
        cv::Rect(cv::Point{50, 100}, cv::Size{250, 360}),
    };
    std::vector<cv::Mat> out_gapi;
    std::vector<cv::Mat> out_onnx(rois.size());
    // ONNX_API code
    for (size_t i = 0; i < rois.size(); ++i) {
        InferONNX(model_path, in_mat1(rois[i]), out_onnx[i], mean, std);
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
    for (size_t i = 0; i < rois.size(); ++i) {
        ASSERT_EQ(1000u, out_onnx[i].total());
        ASSERT_EQ(1000u, out_gapi[i].total());
        normAssert(out_onnx[i], out_gapi[i], "Test classification output");
    }
}

TEST(ONNX, Infer2ROIList)
{
    cv::Mat in_mat1;
    std::string model_path = findModel("squeezenet1.0-9");
    in_mat1 = initMatrixRandU(CV_8UC3, cv::Size{640, 480});

    cv::Scalar mean = { 0.485, 0.456, 0.406 }; // squeeznet mean
    cv::Scalar std  = { 0.229, 0.224, 0.225 }; // squeeznet std

    std::vector<cv::Rect> rois = {
        cv::Rect(cv::Point{ 0,   0}, cv::Size{80, 120}),
        cv::Rect(cv::Point{50, 100}, cv::Size{250, 360}),
    };
    std::vector<cv::Mat> out_gapi;
    std::vector<cv::Mat> out_onnx(rois.size());
    // ONNX_API code
    for (size_t i = 0; i < rois.size(); ++i) {
        InferONNX(model_path, in_mat1(rois[i]), out_onnx[i], mean, std);
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
    for (size_t i = 0; i < rois.size(); ++i) {
        ASSERT_EQ(1000u, out_onnx[i].total());
        ASSERT_EQ(1000u, out_gapi[i].total());
        normAssert(out_onnx[i], out_gapi[i], "Test classification output");
    }
}

} // namespace opencv_test

#endif //  HAVE_ONNX
