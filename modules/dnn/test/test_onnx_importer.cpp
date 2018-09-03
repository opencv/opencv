// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.


#include "test_precomp.hpp"
#include "npy_blob.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace opencv_test { namespace {

template<typename TString>
static std::string _tf(TString filename)
{
    String rootFolder = "dnn/onnx/";
    return findDataFile(rootFolder + filename, false);
}

class Test_ONNX_layers : public DNNTestLayer
{
public:
    enum Extension
    {
        npy,
        pb
    };

    void testONNXModels(const String& basename, const Extension ext = npy, const double l1 = 0, const float lInf = 0)
    {
        String onnxmodel = _tf("models/" + basename + ".onnx");
        Mat inp, ref;
        if (ext == npy) {
            inp = blobFromNPY(_tf("data/input_" + basename + ".npy"));
            ref = blobFromNPY(_tf("data/output_" + basename + ".npy"));
        }
        else if (ext == pb) {
            inp = readTensorFromONNX(_tf("data/input_" + basename + ".pb"));
            ref = readTensorFromONNX(_tf("data/output_" + basename + ".pb"));
        }
        else
            CV_Error(Error::StsUnsupportedFormat, "Unsupported extension");

        checkBackend(&inp, &ref);
        Net net = readNetFromONNX(onnxmodel);
        ASSERT_FALSE(net.empty());

        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);

        net.setInput(inp);
        Mat out = net.forward();
        normAssert(ref, out, "", l1 ? l1 : default_l1, lInf ? lInf : default_lInf);
    }
};

TEST_P(Test_ONNX_layers, MaxPooling)
{
    testONNXModels("maxpooling");
    testONNXModels("two_maxpooling");
}

TEST_P(Test_ONNX_layers, Convolution)
{
    testONNXModels("convolution");
    testONNXModels("two_convolution");
}

TEST_P(Test_ONNX_layers, Dropout)
{
    testONNXModels("dropout");
}

TEST_P(Test_ONNX_layers, Linear)
{
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        throw SkipTestException("");
    testONNXModels("linear");
}

TEST_P(Test_ONNX_layers, ReLU)
{
    testONNXModels("ReLU");
}

TEST_P(Test_ONNX_layers, MaxPooling_Sigmoid)
{
    testONNXModels("maxpooling_sigmoid");
}

TEST_P(Test_ONNX_layers, Concatenation)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE &&
         (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_OPENCL))
        throw SkipTestException("");
    testONNXModels("concatenation");
}

TEST_P(Test_ONNX_layers, AveragePooling)
{
    testONNXModels("average_pooling");
}

TEST_P(Test_ONNX_layers, BatchNormalization)
{
    testONNXModels("batch_norm");
}

TEST_P(Test_ONNX_layers, Multiplication)
{
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        throw SkipTestException("");
    testONNXModels("mul");
}

TEST_P(Test_ONNX_layers, Constant)
{
    testONNXModels("constant");
}

TEST_P(Test_ONNX_layers, MultyInputs)
{
    const String model =  _tf("models/multy_inputs.onnx");

    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat inp1 = blobFromNPY(_tf("data/input_multy_inputs_0.npy"));
    Mat inp2 = blobFromNPY(_tf("data/input_multy_inputs_1.npy"));
    Mat ref  = blobFromNPY(_tf("data/output_multy_inputs.npy"));
    checkBackend(&inp1, &ref);

    net.setInput(inp1, "0");
    net.setInput(inp2, "1");
    Mat out = net.forward();

    normAssert(ref, out, "", default_l1,  default_lInf);
}


INSTANTIATE_TEST_CASE_P(/*nothing*/, Test_ONNX_layers, dnnBackendsAndTargets());

class Test_ONNX_nets : public Test_ONNX_layers {};
TEST_P(Test_ONNX_nets, Alexnet)
{
    const String model =  _tf("models/alexnet.onnx");

    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat inp = imread(_tf("../grace_hopper_227.png"));
    Mat ref = blobFromNPY(_tf("../caffe_alexnet_prob.npy"));
    checkBackend(&inp, &ref);

    net.setInput(blobFromImage(inp, 1.0f, Size(227, 227), Scalar(), false));
    ASSERT_FALSE(net.empty());
    Mat out = net.forward();

    normAssert(out, ref, "", default_l1,  default_lInf);
}

TEST_P(Test_ONNX_nets, Squeezenet)
{
    testONNXModels("squeezenet", pb);
}

TEST_P(Test_ONNX_nets, Googlenet)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE)
        throw SkipTestException("");

    const String model = _tf("models/googlenet.onnx");

    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    std::vector<Mat> images;
    images.push_back( imread(_tf("../googlenet_0.png")) );
    images.push_back( imread(_tf("../googlenet_1.png")) );
    Mat inp = blobFromImages(images, 1.0f, Size(), Scalar(), false);
    Mat ref = blobFromNPY(_tf("../googlenet_prob.npy"));
    checkBackend(&inp, &ref);

    net.setInput(inp);
    ASSERT_FALSE(net.empty());
    Mat out = net.forward();

    normAssert(ref, out, "", default_l1,  default_lInf);
}

TEST_P(Test_ONNX_nets, CaffeNet)
{
    testONNXModels("caffenet", pb);
}

TEST_P(Test_ONNX_nets, RCNN_ILSVRC13)
{
    testONNXModels("rcnn_ilsvrc13", pb);
}

TEST_P(Test_ONNX_nets, VGG16)
{
    double l1 = default_l1;
    double lInf = default_lInf;
    // output range: [-69; 72]
    if (target == DNN_TARGET_OPENCL_FP16) {
        l1 = 0.087;
        lInf = 0.585;
    }
    else if (backend == DNN_BACKEND_INFERENCE_ENGINE && target == DNN_TARGET_OPENCL) {
        lInf = 1.2e-4;
    }
    testONNXModels("vgg16", pb, l1, lInf);
}

TEST_P(Test_ONNX_nets, VGG16_bn)
{
    double l1 = default_l1;
    double lInf = default_lInf;
    // output range: [-16; 27]
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16) {
        l1 = 0.0086;
        lInf = 0.037;
    }
    else if (backend == DNN_BACKEND_INFERENCE_ENGINE && target == DNN_TARGET_OPENCL_FP16) {
        l1 = 0.031;
        lInf = 0.2;
    }
    testONNXModels("vgg16-bn", pb, l1, lInf);
}

TEST_P(Test_ONNX_nets, ZFNet)
{
    testONNXModels("zfnet512", pb);
}

TEST_P(Test_ONNX_nets, ResNet18v1)
{
    double l1 = default_l1;
    double lInf = default_lInf;
    // output range: [-16; 22]
    if (target == DNN_TARGET_OPENCL_FP16) {
        l1 = 0.022;
        lInf = 0.12;
    }
    testONNXModels("resnet18v1", pb, l1, lInf);
}

TEST_P(Test_ONNX_nets, ResNet50v1)
{
    double l1 = default_l1;
    double lInf = default_lInf;
    // output range: [-67; 75]
    if (target == DNN_TARGET_OPENCL_FP16) {
        l1 = 0.6;
        lInf = 0.51;
    }
    else if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_CPU) {
        l1 = 1.24e-5;
        lInf = 1.1e-4;
    }
    else if (backend == DNN_BACKEND_INFERENCE_ENGINE && target == DNN_TARGET_OPENCL) {
        l1 = 1.25e-5;
        lInf = 1.2e-4;
    }
    testONNXModels("resnet50v1", pb, l1, lInf);
}

TEST_P(Test_ONNX_nets, ResNet101_DUC_HDC)
{
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_OPENCL) {
        throw SkipTestException("");
    }
    testONNXModels("resnet101_duc_hdc", pb);
}

TEST_P(Test_ONNX_nets, TinyYolov2)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE &&
         (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_OPENCL)) {
        throw SkipTestException("");
    }

    double l1 = default_l1;
    double lInf = default_lInf;
    // output range: [-11; 8]
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16) {
        l1 = 0.017;
        lInf = 0.14;
    }
    testONNXModels("tiny_yolo2", pb, l1, lInf);
}

TEST_P(Test_ONNX_nets, CNN_MNIST)
{
    double l1 = default_l1;
    double lInf = default_lInf;
    // output range: [-1952; 6574]
    if (target == DNN_TARGET_CPU) {
        l1 = 4.2e-4;
        lInf = 1e-3;
    }
    else if (target == DNN_TARGET_OPENCL) {
        l1 = 4.3e-4;
        lInf = 1e-3;
    }
    else if (target == DNN_TARGET_OPENCL_FP16) {
        l1 = 3.82;
        lInf = 13.5;
    }
    testONNXModels("cnn_mnist", pb, l1, lInf);
}

TEST_P(Test_ONNX_nets, MobileNet_v2)
{
    double l1 = default_l1;
    double lInf = default_lInf;
    // output range: [-166; 317]
    if (target == DNN_TARGET_CPU) {
        l1 = 6.8e-5;
        lInf = 5e-4;
    }
    else if (target == DNN_TARGET_OPENCL) {
        l1 = 7e-5;
        lInf = 5e-4;
    }
    else if (target == DNN_TARGET_OPENCL_FP16) {
        l1 = 0.38;
        lInf = 2.87;
    }
    testONNXModels("mobilenetv2", pb, l1, lInf);
}

TEST_P(Test_ONNX_nets, LResNet100E_IR)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE &&
         (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_OPENCL))
        throw SkipTestException("");

    double l1 = default_l1;
    double lInf = default_lInf;
    // output range: [-3; 3]
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16) {
        l1 = 0.009;
        lInf = 0.035;
    }
    testONNXModels("LResNet100E_IR", pb, l1, lInf);
}

TEST_P(Test_ONNX_nets, Emotion_ferplus)
{
    testONNXModels("emotion_ferplus", pb);
}

TEST_P(Test_ONNX_nets, Inception_v2)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE)
        throw SkipTestException("");

    testONNXModels("inception_v2", pb);
}

TEST_P(Test_ONNX_nets, DenseNet121)
{
    double l1 = default_l1;
    double lInf = default_lInf;
    // output range: [-87; 138]
    if (target == DNN_TARGET_CPU) {
        l1 = 1.7e-05;
    }
    else if (target == DNN_TARGET_OPENCL) {
        l1 = 1.88e-5;
        lInf = 1.2e-4;
    }
    else if (target == DNN_TARGET_OPENCL_FP16) {
        l1 = 0.11;
        lInf = 0.74;
    }
    testONNXModels("densenet121", pb, l1, lInf);
}

INSTANTIATE_TEST_CASE_P(/**/, Test_ONNX_nets, dnnBackendsAndTargets());

}} // namespace
