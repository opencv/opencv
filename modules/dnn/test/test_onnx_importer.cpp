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
        String modelsFolder = "models/";
        String onnxmodel = _tf(modelsFolder + basename + ".onnx");

        String dataFolder = "data/";
        String inpfile = _tf(dataFolder + "input_" + basename + ".npy");
        String outfile = _tf(dataFolder + "output_" + basename + ".npy");

        Mat inp = blobFromNPY(inpfile);
        Mat ref = blobFromNPY(outfile);
        checkBackend(&inp, &ref);

        Net net = readNetFromONNX(onnxmodel);
        ASSERT_FALSE(net.empty());

        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);

        net.setInput(inp);
        Mat out = net.forward();
std::cout << out.size << '\n';
        normAssert(ref, out, "", default_l1,  default_lInf);
    }
};

TEST_P(Test_ONNX_layers, MaxPooling)
{
    testLayerUsingONNXModels("maxpooling");
}

TEST_P(Test_ONNX_layers, Convolution)
{
    testLayerUsingONNXModels("convolution");
}

TEST_P(Test_ONNX_layers, Dropout)
{
    testLayerUsingONNXModels("dropout");
}

TEST_P(Test_ONNX_layers, Linear)
{
    testLayerUsingONNXModels("linear");
}

TEST_P(Test_ONNX_layers, ReLU)
{
    testLayerUsingONNXModels("ReLU");
}

TEST_P(Test_ONNX_layers, Two_MaxPooling)
{
    testLayerUsingONNXModels("maxpooling2");
}

TEST_P(Test_ONNX_layers, Two_Convolution)
{
    testLayerUsingONNXModels("convolution2");
}

TEST_P(Test_ONNX_layers, MaxPooling_Sigmoid)
{
    testLayerUsingONNXModels("maxpooling_sigmoid");
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Test_ONNX_layers, dnnBackendsAndTargets());

class Test_ONNX_nets : public DNNTestLayer {};
TEST_P(Test_ONNX_nets, Alexnet)
{
    Net net;
    const String model =  _tf("models/bvlc_alexnet.onnx");

    net = readNetFromONNX(model);

    ASSERT_FALSE(net.empty());

    int targetId = get<1>(GetParam());

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(targetId);

    Mat sample = imread(_tf("../grace_hopper_227.png"));
    ASSERT_TRUE(!sample.empty());

    net.setInput(blobFromImage(sample, 1.0f, Size(227, 227), Scalar(), false));
    Mat out = net.forward();
    Mat ref = blobFromNPY(_tf("../caffe_alexnet_prob.npy"));
    normAssert(ref, out, "", default_l1,  default_lInf);
}

INSTANTIATE_TEST_CASE_P(/**/, Test_ONNX_nets, dnnBackendsAndTargets());

}} // namespace
