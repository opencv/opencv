// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.


#include "test_precomp.hpp"
#include "npy_blob.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace opencv_test { namespace {

class Test_ONNX_layers : public DNNTestLayer
{
public:

    void testLayer(const String& layer)
    {
        testONNXModels("onnx/models/" + layer + ".onnx",
                       "onnx/data/input_" + layer + ".npy",
                       "onnx/data/output_" + layer + ".npy");
    }

    void testModel(const String& family, const String& name)
    {
        testONNXModels(family + "/" + name + ".onnx",
                       family + "/input_" + name + ".pb",
                       family + "/output_" + name + ".pb");
    }

private:

    void testONNXModels(const String& model, const String& input, const String &output)
    {
        string onnxmodel = findDataFile(model, false);
        string inputFile = findDataFile(input, false);
        string outputFile = findDataFile(output, false);

        Mat inp, ref;
        if (inputFile.rfind(".npy") != string::npos && outputFile.rfind(".npy") != string::npos)
        {
            inp = blobFromNPY(inputFile);
            ref = blobFromNPY(outputFile);
        }
        else if (inputFile.rfind(".pb") != string::npos && outputFile.rfind(".pb") != string::npos)
        {
            inp = readTensorFromONNX(inputFile);
            ref = readTensorFromONNX(outputFile);
        }
        ASSERT_FALSE(inp.empty()) << "Failed to read input sample";
        ASSERT_FALSE(ref.empty()) << "Failed to read output sample";

        checkBackend(&inp, &ref);
        Net net = readNetFromONNX(onnxmodel);
        ASSERT_FALSE(net.empty());

        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);

        net.setInput(inp);
        Mat out = net.forward();
        normAssert(ref, out, "", default_l1, default_lInf);
    }
};

TEST_P(Test_ONNX_layers, MaxPooling)
{
    testLayer("maxpooling");
    testLayer("two_maxpooling");
}

TEST_P(Test_ONNX_layers, Convolution)
{
    testLayer("convolution");
    testLayer("two_convolution");
}

TEST_P(Test_ONNX_layers, Dropout)
{
    testLayer("dropout");
}

TEST_P(Test_ONNX_layers, Linear)
{
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        throw SkipTestException("");
    testLayer("linear");
}

TEST_P(Test_ONNX_layers, ReLU)
{
    testLayer("ReLU");
}

TEST_P(Test_ONNX_layers, MaxPooling_Sigmoid)
{
    testLayer("maxpooling_sigmoid");
}

TEST_P(Test_ONNX_layers, Concatenation)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE &&
         (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_OPENCL || target == DNN_TARGET_MYRIAD))
        throw SkipTestException("");
    testLayer("concatenation");
}

TEST_P(Test_ONNX_layers, AveragePooling)
{
    testLayer("average_pooling");
}

TEST_P(Test_ONNX_layers, BatchNormalization)
{
    testLayer("batch_norm");
}

TEST_P(Test_ONNX_layers, Multiplication)
{
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16 ||
        backend == DNN_BACKEND_INFERENCE_ENGINE && target == DNN_TARGET_MYRIAD)
        throw SkipTestException("");
    testLayer("mul");
}

TEST_P(Test_ONNX_layers, Constant)
{
    testLayer("constant");
}

TEST_P(Test_ONNX_layers, MultyInputs)
{
    const String model =  findDataFile("onnx/models/multy_inputs.onnx", false);

    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat inp1 = blobFromNPY(findDataFile("onnx/data/input_multy_inputs_0.npy", false));
    Mat inp2 = blobFromNPY(findDataFile("onnx/data/input_multy_inputs_1.npy", false));
    Mat ref  = blobFromNPY(findDataFile("onnx/data/output_multy_inputs.npy", false));
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
    const String model =  findDataFile("alexnet/alexnet.onnx", false);

    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat inp = imread(findDataFile("grace_hopper_227.png", false));
    Mat ref = blobFromNPY(findDataFile("caffe_alexnet_prob.npy", false));
    checkBackend(&inp, &ref);

    net.setInput(blobFromImage(inp, 1.0f, Size(227, 227), Scalar(), false));
    ASSERT_FALSE(net.empty());
    Mat out = net.forward();

    normAssert(out, ref, "", default_l1,  default_lInf);
}

TEST_P(Test_ONNX_nets, Squeezenet)
{
    testModel("squeezenet", "squeezenet");
}

TEST_P(Test_ONNX_nets, Googlenet)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE)
        throw SkipTestException("");

    const String model = findDataFile("inception/googlenet.onnx", false);

    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    std::vector<Mat> images;
    images.push_back( imread(findDataFile("googlenet_0.png", false)) );
    images.push_back( imread(findDataFile("googlenet_1.png", false)) );
    Mat inp = blobFromImages(images, 1.0f, Size(), Scalar(), false);
    Mat ref = blobFromNPY(findDataFile("googlenet_prob.npy", false));
    checkBackend(&inp, &ref);

    net.setInput(inp);
    ASSERT_FALSE(net.empty());
    Mat out = net.forward();

    normAssert(ref, out, "", default_l1,  default_lInf);
}

TEST_P(Test_ONNX_nets, CaffeNet)
{
    testModel("alexnet", "caffenet");
}

TEST_P(Test_ONNX_nets, RCNN_ILSVRC13)
{
    testModel("rcnn", "rcnn_ilsvrc13");
}

#ifdef OPENCV_32BIT_CONFIGURATION
TEST_P(Test_ONNX_nets, DISABLED_VGG16)  // memory usage >2Gb
#else
TEST_P(Test_ONNX_nets, VGG16)
#endif
{
    // output range: [-69; 72]
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) {
        default_l1 = 0.087;
        default_lInf = 0.585;
    }
    else if (backend == DNN_BACKEND_INFERENCE_ENGINE && target == DNN_TARGET_OPENCL) {
        default_lInf = 1.2e-4;
    }
    testModel("vgg16", "vgg16");
}

#ifdef OPENCV_32BIT_CONFIGURATION
TEST_P(Test_ONNX_nets, DISABLED_VGG16_bn)  // memory usage >2Gb
#else
TEST_P(Test_ONNX_nets, VGG16_bn)
#endif
{
    // output range: [-16; 27]
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16) {
        default_l1 = 0.0086;
        default_lInf = 0.037;
    }
    else if (backend == DNN_BACKEND_INFERENCE_ENGINE &&
             (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD)) {
        default_l1 = 0.031;
        default_lInf = 0.2;
    }
    testModel("vgg16", "vgg16-bn");
}

TEST_P(Test_ONNX_nets, ZFNet)
{
    testModel("zfnet", "zfnet512");
}

TEST_P(Test_ONNX_nets, ResNet18v1)
{
    // output range: [-16; 22]
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD)
    {
        default_l1 = 0.022;
        default_lInf = 0.12;
    }
    testModel("resnet", "resnet18v1");
}

TEST_P(Test_ONNX_nets, ResNet50v1)
{
    // output range: [-67; 75]
    default_l1 = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 0.6 : 1.25e-5;
    default_lInf = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 0.51 : 1.2e-4;
    testModel("resnet", "resnet50v1");
}

TEST_P(Test_ONNX_nets, ResNet101_DUC_HDC)
{
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_OPENCL
                || target == DNN_TARGET_MYRIAD) {
        throw SkipTestException("");
    }
    testModel("resnet", "resnet101_duc_hdc");
}

TEST_P(Test_ONNX_nets, TinyYolov2)
{
    if (cvtest::skipUnstableTests ||
        backend == DNN_BACKEND_INFERENCE_ENGINE && (target == DNN_TARGET_OPENCL || target == DNN_TARGET_OPENCL_FP16)) {
        throw SkipTestException("");
    }
    // output range: [-11; 8]
    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD)
    {
        default_l1 = 0.017;
        default_lInf = 0.14;
    }
    testModel("yolo", "tiny_yolo2");
}

TEST_P(Test_ONNX_nets, CNN_MNIST)
{
    // output range: [-1952; 6574]
    default_l1 = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 3.82 : 4.4e-4;
    default_lInf = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 13.5 : 2e-3;

    testModel("mnist", "cnn_mnist");
}

TEST_P(Test_ONNX_nets, MobileNet_v2)
{
    // output range: [-166; 317]
    default_l1 = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 0.38 : 7e-5;
    default_lInf = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 2.87 : 5e-4;
    testModel("mobilenet", "mobilenetv2");
}

TEST_P(Test_ONNX_nets, LResNet100E_IR)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE &&
         (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_OPENCL || target == DNN_TARGET_MYRIAD))
        throw SkipTestException("");

    // output range: [-3; 3]
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16) {
        default_l1 = 0.009;
        default_lInf = 0.035;
    }
    testModel("resnet", "LResNet100E_IR");
}

TEST_P(Test_ONNX_nets, Emotion_ferplus)
{
    testModel("emotion", "emotion_ferplus");
}

TEST_P(Test_ONNX_nets, Inception_v2)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE)
        throw SkipTestException("");

    testModel("inception", "inception_v2");
}

TEST_P(Test_ONNX_nets, DenseNet121)
{
    // output range: [-87; 138]
    default_l1 = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 0.12 : 2.2e-5;
    default_lInf = (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD) ? 0.74 : 1.23e-4;
    testModel("densenet", "densenet121");
}

TEST_P(Test_ONNX_nets, Inception_v1)
{
    testModel("inception", "inception_v1");
}

INSTANTIATE_TEST_CASE_P(/**/, Test_ONNX_nets, dnnBackendsAndTargets());

}} // namespace
