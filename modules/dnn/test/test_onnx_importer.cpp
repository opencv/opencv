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
    void testLayerUsingONNXModels(const String& basename)
    {
        String onnxmodel = _tf("models/" + basename + ".onnx");

        String inpfile = _tf("data/input_" + basename + ".npy");
        String outfile = _tf("data/output_" + basename + ".npy");

        Mat inp = blobFromNPY(inpfile);
        Mat ref = blobFromNPY(outfile);
        checkBackend(&inp, &ref);

        Net net = readNetFromONNX(onnxmodel);
        ASSERT_FALSE(net.empty());

        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);

        net.setInput(inp);
        Mat out = net.forward();
        normAssert(ref, out, "", default_l1,  default_lInf);
    }
};

TEST_P(Test_ONNX_layers, MaxPooling)
{
    testLayerUsingONNXModels("maxpooling");
    testLayerUsingONNXModels("maxpooling_stride");
    testLayerUsingONNXModels("two_maxpooling");
}

TEST_P(Test_ONNX_layers, Convolution)
{
    testLayerUsingONNXModels("convolution");
    testLayerUsingONNXModels("convolution_pad");
    testLayerUsingONNXModels("convolution_stride");
    testLayerUsingONNXModels("two_convolution");
}

TEST_P(Test_ONNX_layers, Dropout)
{
    testLayerUsingONNXModels("dropout");
}

TEST_P(Test_ONNX_layers, Linear)
{
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        throw SkipTestException("");
    testLayerUsingONNXModels("linear");
}

TEST_P(Test_ONNX_layers, ReLU)
{
    testLayerUsingONNXModels("ReLU");
}

TEST_P(Test_ONNX_layers, MaxPooling_Sigmoid)
{
    testLayerUsingONNXModels("maxpooling_sigmoid");
}

TEST_P(Test_ONNX_layers, Concatenation)
{
    testLayerUsingONNXModels("concatenation");
}

TEST_P(Test_ONNX_layers, AveragePooling)
{
    testLayerUsingONNXModels("average_pooling");
}

TEST_P(Test_ONNX_layers, BatchNormalization)
{
    testLayerUsingONNXModels("batch_norm");
}


INSTANTIATE_TEST_CASE_P(/*nothing*/, Test_ONNX_layers, dnnBackendsAndTargets());

class Test_ONNX_nets : public DNNTestLayer {};
TEST_P(Test_ONNX_nets, Alexnet)
{
    const String model =  _tf("models/bvlc_alexnet.onnx");

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

    normAssert(ref, out, "", default_l1,  default_lInf);
}

TEST_P(Test_ONNX_nets, Squeezenet)
{
    const String model =  _tf("models/squeezenet.onnx");

    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat image = imread(_tf("../googlenet_0.png"));
    Mat inp = blobFromImage(image, 1.0f, Size(227,227), Scalar(), false);
    Mat ref = blobFromNPY(_tf("../squeezenet_v1.1_prob.npy"));
    checkBackend(&inp, &ref);

    net.setInput(inp);
    ASSERT_FALSE(net.empty());
    Mat out = net.forward();
    out = out.reshape(1, 1);

    normAssert(ref, out, "", default_l1,  default_lInf);
}

TEST_P(Test_ONNX_nets, Googlenet)
{
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
    const String model =  _tf("models/caffenet.onnx");

    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat inp = readTensorFromONNX(_tf("data/input_caffenet.pb"));
    Mat ref = readTensorFromONNX(_tf("data/output_caffenet.pb"));
    checkBackend(&inp, &ref);

    net.setInput(inp);
    ASSERT_FALSE(net.empty());
    Mat out = net.forward();

    normAssert(ref, out, "", default_l1,  default_lInf);
}

TEST_P(Test_ONNX_nets, RCNN_ILSVRC13)
{
    const String model = _tf("models/rcnn_ilsvrc13.onnx");

    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat inp = readTensorFromONNX(_tf("data/input_rcnn_ilsvrc13.pb"));
    Mat ref = readTensorFromONNX(_tf("data/output_rcnn_ilsvrc13.pb"));
    checkBackend(&inp, &ref);

    net.setInput(inp);
    ASSERT_FALSE(net.empty());
    Mat out = net.forward();

    normAssert(ref, out, "", default_l1,  default_lInf);
}

TEST_P(Test_ONNX_nets, VGG16)
{
    const String model = _tf("models/vgg16.onnx");

    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat inp = readTensorFromONNX(_tf("data/input_vgg16.pb"));
    Mat ref = readTensorFromONNX(_tf("data/output_vgg16.pb"));
    checkBackend(&inp, &ref);

    net.setInput(inp);
    ASSERT_FALSE(net.empty());
    Mat out = net.forward();

    normAssert(out, ref, "", default_l1,  default_lInf);
}

TEST_P(Test_ONNX_nets, VGG16_bn)
{
    const String model = _tf("models/vgg16-bn.onnx");

    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat inp = readTensorFromONNX(_tf("data/input_vgg16-bn.pb"));
    Mat ref = readTensorFromONNX(_tf("data/output_vgg16-bn.pb"));
    checkBackend(&inp, &ref);

    net.setInput(inp);
    ASSERT_FALSE(net.empty());
    Mat out = net.forward();

    normAssert(out, ref, "", default_l1,  default_lInf);
}

TEST_P(Test_ONNX_nets, ZFNet)
{
    const String model = _tf("models/zfnet512.onnx");

    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat inp = readTensorFromONNX(_tf("data/input_zfnet512.pb"));
    Mat ref = readTensorFromONNX(_tf("data/output_zfnet512.pb"));
    checkBackend(&inp, &ref);

    net.setInput(inp);
    ASSERT_FALSE(net.empty());
    Mat out = net.forward();

    normAssert(out, ref, "", default_l1,  default_lInf);
}

TEST_P(Test_ONNX_nets, ResNet18v1)
{
    const String model = _tf("models/resnet18v1.onnx");

    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat inp = readTensorFromONNX(_tf("data/input_resnet18v1.pb"));
    Mat ref = readTensorFromONNX(_tf("data/output_resnet18v1.pb"));
    checkBackend(&inp, &ref);

    net.setInput(inp);
    ASSERT_FALSE(net.empty());
    Mat out = net.forward();


    normAssert(out, ref, "", default_l1,  default_lInf);
}

TEST_P(Test_ONNX_nets, ResNet50v1)
{
    const String model = _tf("models/resnet50v1.onnx");

    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat inp = readTensorFromONNX(_tf("data/input_resnet50v1.pb"));
    Mat ref = readTensorFromONNX(_tf("data/output_resnet50v1.pb"));
    checkBackend(&inp, &ref);

    net.setInput(inp);
    ASSERT_FALSE(net.empty());
    Mat out = net.forward();

    normAssert(out, ref, "", default_l1,  default_lInf);
}

INSTANTIATE_TEST_CASE_P(/**/, Test_ONNX_nets, dnnBackendsAndTargets());

}} // namespace
