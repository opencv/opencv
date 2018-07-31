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
    testLayerUsingONNXModels("two_maxpooling");
}

TEST_P(Test_ONNX_layers, Convolution)
{
    testLayerUsingONNXModels("convolution");
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

INSTANTIATE_TEST_CASE_P(/**/, Test_ONNX_nets, dnnBackendsAndTargets());

}} // namespace
