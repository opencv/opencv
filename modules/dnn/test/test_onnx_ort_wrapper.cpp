// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "npy_blob.hpp"

namespace opencv_test { namespace {

#ifdef HAVE_ONNXRUNTIME

static std::string _tf(const std::string& filename, bool required = true)
{
    return findDataFile(std::string("dnn/onnx/") + filename, required);
}

static cv::dnn::Net readNetFromONNX_ORT(const std::string& onnxModelPath)
{
    cv::dnn::Net net = cv::dnn::readNetFromONNX(onnxModelPath, cv::dnn::ENGINE_ORT);
    EXPECT_FALSE(net.empty());
    return net;
}

TEST(Test_ONNX_ORT_Wrapper, SingleInputSingleOutput)
{
    const std::string basename = "convolution";
    const std::string onnxmodel = _tf("models/" + basename + ".onnx", true);

    cv::Mat input = blobFromNPY(_tf("data/input_" + basename + ".npy"));
    cv::Mat ref = blobFromNPY(_tf("data/output_" + basename + ".npy"));

    cv::dnn::Net net = readNetFromONNX_ORT(onnxmodel);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    net.setInput(input);
    cv::Mat out = net.forward();

    normAssert(ref, out, "ORT 1in/1out convolution", 1e-5, 1e-4);
}

TEST(Test_ONNX_ORT_Wrapper, MultipleInputSingleOutput)
{
    const std::string basename = "min";
    const std::string onnxmodel = _tf("models/" + basename + ".onnx", true);

    cv::Mat inp0 = blobFromNPY(_tf("data/input_" + basename + "_0.npy"));
    cv::Mat inp1 = blobFromNPY(_tf("data/input_" + basename + "_1.npy"));
    cv::Mat ref = blobFromNPY(_tf("data/output_" + basename + ".npy"));

    cv::dnn::Net net = readNetFromONNX_ORT(onnxmodel);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    net.setInput(inp0, "0");
    net.setInput(inp1, "1");
    cv::Mat out = net.forward();

    normAssert(ref, out, "ORT 2in/1out min", 1e-5, 1e-4);
}

TEST(Test_ONNX_ORT_Wrapper, SingleInputMultipleOutput)
{
    const std::string basename = "top_k";
    const std::string onnxmodel = _tf("models/" + basename + ".onnx", true);

    cv::Mat input = cv::dnn::readTensorFromONNX(_tf("data/input_" + basename + ".pb"));
    cv::Mat ref_val = cv::dnn::readTensorFromONNX(_tf("data/output_" + basename + "_0.pb"));
    cv::Mat ref_ind = cv::dnn::readTensorFromONNX(_tf("data/output_" + basename + "_1.pb"));

    cv::dnn::Net net = readNetFromONNX_ORT(onnxmodel);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    net.setInput(input);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, std::vector<std::string>{"values", "indices"});
    ASSERT_EQ(outputs.size(), 2u);

    normAssert(ref_val, outputs[0], "ORT top_k values", 1e-5, 1e-4);
    normAssert(ref_ind, outputs[1], "ORT top_k indices", 0.0, 0.0);
}

#else  // HAVE_ONNXRUNTIME

TEST(Test_ONNX_ORT_Wrapper, DISABLED_NoONNXRuntime) {}

#endif

}} // namespace
