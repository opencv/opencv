// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2024, OpenCV Team, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/dnn/layer.details.hpp>  // CV_DNN_REGISTER_LAYER_CLASS

namespace opencv_test { namespace {

class Layer_Test_01D: public testing::TestWithParam<tuple<int>>
{
public:
    int dims;
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    float inp_value;
    Mat input;
    LayerParams lp;

    void SetUp()
    {
        dims = get<0>(GetParam());
        input_shape = {dims};
        output_shape = {dims};

        // generate random positeve value from 1 to 10
        RNG& rng = TS::ptr()->get_rng();
        inp_value = rng.uniform(1.0, 10.0); // random uniform value
        input = Mat(dims, input_shape.data(), CV_32F, inp_value);
    }

    void TestLayer(Ptr<Layer> layer, std::vector<Mat> &inputs, const Mat& output_ref){
        std::vector<Mat> outputs;
        runLayer(layer, inputs, outputs);
        ASSERT_EQ(shape(output_ref), shape(outputs[0]));
        normAssert(output_ref, outputs[0]);
    }

};

TEST_P(Layer_Test_01D, Scale)
{

    lp.type = "Scale";
    lp.name = "scaleLayer";
    lp.set("axis", 0);
    lp.set("mode", "scale");
    lp.set("bias_term", false);
    Ptr<ScaleLayer> layer = ScaleLayer::create(lp);

    Mat weight = Mat(dims, output_shape.data(), CV_32F, 2.0);
    std::vector<Mat> inputs{input, weight};
    Mat output_ref = input.mul(weight);

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, ReLU6)
{

    lp.type = "ReLU6";
    lp.name = "ReLU6Layer";
    lp.set("min_value", 0.0);
    lp.set("max_value", 1.0);
    Ptr<ReLU6Layer> layer = ReLU6Layer::create(lp);

    Mat output_ref(dims, output_shape.data(), CV_32F, 1.0);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Clip)
{

    lp.type = "Clip";
    lp.name = "clipLayer";
    lp.set("min_value", 0.0);
    lp.set("max_value", 1.0);
    Ptr<ReLU6Layer> layer = ReLU6Layer::create(lp);

    Mat output_ref(dims, output_shape.data(), CV_32F, 1.0);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, ReLU)
{

    lp.type = "ReLU";
    lp.name = "reluLayer";
    lp.set("negative_slope", 0.0);
    Ptr<ReLULayer> layer = ReLULayer::create(lp);

    Mat output_ref(dims, output_shape.data(), CV_32F, inp_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Gelu)
{

    lp.type = "Gelu";
    lp.name = "geluLayer";
    Ptr<GeluLayer> layer = GeluLayer::create(lp);

    float value = inp_value * 0.5 * (std::erf(inp_value * 1 / std::sqrt(2.0)) + 1.0);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, GeluApprox)
{

    lp.type = "GeluApprox";
    lp.name = "geluApproxLayer";
    Ptr<GeluApproximationLayer> layer = GeluApproximationLayer::create(lp);

    float value = inp_value * 0.5 * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (inp_value + 0.044715 * std::pow(inp_value, 3))));
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Sigmoid)
{

    lp.type = "Sigmoid";
    lp.name = "sigmoidLayer";
    Ptr<SigmoidLayer> layer = SigmoidLayer::create(lp);

    float value = 1.0 / (1.0 + std::exp(-inp_value));
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Tanh)
{

    lp.type = "TanH";
    lp.name = "TanHLayer";
    Ptr<Layer> layer = TanHLayer::create(lp);


    float value = std::tanh(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Swish)
{

    lp.type = "Swish";
    lp.name = "SwishLayer";
    Ptr<Layer> layer = SwishLayer::create(lp);

    float value = inp_value / (1 + std::exp(-inp_value));
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Mish)
{

    lp.type = "Mish";
    lp.name = "MishLayer";
    Ptr<Layer> layer = MishLayer::create(lp);

    float value = inp_value * std::tanh(std::log(1 + std::exp(inp_value)));
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, ELU)
{

    lp.type = "ELU";
    lp.name = "eluLayer";
    lp.set("alpha", 1.0);
    Ptr<Layer> layer = ELULayer::create(lp);

    float value = inp_value > 0 ? inp_value : std::exp(inp_value) - 1;
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Abs)
{

    lp.type = "Abs";
    lp.name = "absLayer";
    Ptr<Layer> layer = AbsLayer::create(lp);

    float value = std::abs(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, BNLL)
{

    lp.type = "BNLL";
    lp.name = "bnllLayer";
    Ptr<Layer> layer = BNLLLayer::create(lp);

    float value = std::log(1 + std::exp(inp_value));
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Ceil)
{

    lp.type = "Ceil";
    lp.name = "ceilLayer";
    Ptr<Layer> layer = CeilLayer::create(lp);

    float value = std::ceil(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Floor)
{

    lp.type = "Floor";
    lp.name = "floorLayer";
    Ptr<Layer> layer = FloorLayer::create(lp);

    float value = std::floor(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Log)
{

    lp.type = "Log";
    lp.name = "logLayer";
    Ptr<Layer> layer = LogLayer::create(lp);

    float value = std::log(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Round)
{

    lp.type = "Round";
    lp.name = "roundLayer";
    Ptr<Layer> layer = RoundLayer::create(lp);

    float value = std::round(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Sqrt)
{

    lp.type = "Sqrt";
    lp.name = "sqrtLayer";
    Ptr<Layer> layer = SqrtLayer::create(lp);

    float value = std::sqrt(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Acos)
{

    lp.type = "Acos";
    lp.name = "acosLayer";
    Ptr<Layer> layer = AcosLayer::create(lp);

    inp_value = 0.5 + static_cast <float> (inp_value) / (static_cast <float> (RAND_MAX/(1-0.5)));
    input = Mat(dims, input_shape.data(), CV_32F, inp_value);

    float value = std::acos(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Acosh)
{

    lp.type = "Acosh";
    lp.name = "acoshLayer";
    Ptr<Layer> layer = AcoshLayer::create(lp);

    float value = std::acosh(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Asin)
{

    lp.type = "Asin";
    lp.name = "asinLayer";
    Ptr<Layer> layer = AsinLayer::create(lp);

    inp_value = 0.5 + static_cast <float> (inp_value) / (static_cast <float> (RAND_MAX/(1-0.5)));
    input = Mat(dims, input_shape.data(), CV_32F, inp_value);

    float value = std::asin(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Asinh)
{

    lp.type = "Asinh";
    lp.name = "asinhLayer";
    Ptr<Layer> layer = AsinhLayer::create(lp);

    float value = std::asinh(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Atan)
{

    lp.type = "Atan";
    lp.name = "atanLayer";
    Ptr<Layer> layer = AtanLayer::create(lp);

    float value = std::atan(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Cos)
{

    lp.type = "Cos";
    lp.name = "cosLayer";
    Ptr<Layer> layer = CosLayer::create(lp);

    float value = std::cos(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Cosh)
{

    lp.type = "Cosh";
    lp.name = "coshLayer";
    Ptr<Layer> layer = CoshLayer::create(lp);

    float value = std::cosh(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Sin)
{

    lp.type = "Sin";
    lp.name = "sinLayer";
    Ptr<Layer> layer = SinLayer::create(lp);

    float value = std::sin(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Sinh)
{

    lp.type = "Sinh";
    lp.name = "sinhLayer";
    Ptr<Layer> layer = SinhLayer::create(lp);

    float value = std::sinh(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Tan)
{

    lp.type = "Tan";
    lp.name = "tanLayer";
    Ptr<Layer> layer = TanLayer::create(lp);

    float value = std::tan(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Erf)
{

    lp.type = "Erf";
    lp.name = "erfLayer";
    Ptr<Layer> layer = ErfLayer::create(lp);

    float out_value = std::erf(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, out_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Reciprocal)
{

    lp.type = "Reciprocal";
    lp.name = "reciprocalLayer";
    Ptr<Layer> layer = ReciprocalLayer::create(lp);

    float out_value = 1/inp_value;
    Mat output_ref(dims, output_shape.data(), CV_32F, out_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, HardSwish)
{

    lp.type = "HardSwish";
    lp.name = "hardSwishLayer";
    Ptr<Layer> layer = HardSwishLayer::create(lp);

    float out_value = inp_value * std::max(0.0f, std::min(6.0f, inp_value + 3.0f)) / 6.0f;
    Mat output_ref(dims, output_shape.data(), CV_32F, out_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Softplus)
{

    lp.type = "Softplus";
    lp.name = "softplusLayer";
    Ptr<Layer> layer = SoftplusLayer::create(lp);

    float out_value = std::log(1 + std::exp(inp_value));
    Mat output_ref(dims, output_shape.data(), CV_32F, out_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, SoftSign)
{

    lp.type = "Softsign";
    lp.name = "softsignLayer";
    Ptr<Layer> layer = SoftsignLayer::create(lp);

    float out_value = inp_value / (1 + std::abs(inp_value));
    Mat output_ref(dims, output_shape.data(), CV_32F, out_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, CELU)
{

    lp.type = "CELU";
    lp.name = "celuLayer";
    lp.set("alpha", 1.0);
    Ptr<Layer> layer = CeluLayer::create(lp);

    float out_value = inp_value < 0 ? std::exp(inp_value) - 1 : inp_value;
    Mat output_ref(dims, output_shape.data(), CV_32F, out_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, HardSigmoid)
{

    lp.type = "HardSigmoid";
    lp.name = "hardSigmoidLayer";
    Ptr<Layer> layer = HardSigmoidLayer::create(lp);

    float out_value = std::max(0.0f, std::min(1.0f, 0.2f * inp_value + 0.5f));
    Mat output_ref(dims, output_shape.data(), CV_32F, out_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, SELU)
{

    lp.type = "SELU";
    lp.name = "seluLayer";
    lp.set("alpha", 1.6732631921768188);
    lp.set("gamma", 1.0507009873554805);
    Ptr<Layer> layer = SeluLayer::create(lp);


    double inp_value_double = static_cast<double>(inp_value); // Ensure the input is treated as double for the computation

    double value_double = 1.0507009873554805 * (inp_value_double > 0 ? inp_value_double : 1.6732631921768188 * (std::exp(inp_value_double / 1.0) - 1));

    float value = static_cast<float>(value_double);

    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, ThresholdedReLU)
{

    lp.type = "ThresholdedReLU";
    lp.name = "thresholdedReluLayer";
    lp.set("alpha", 1.0);
    Ptr<Layer> layer = ThresholdedReluLayer::create(lp);

    float value = inp_value > 1.0 ? inp_value : 0.0;
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Power)
{

    lp.type = "Power";
    lp.name = "powerLayer";
    lp.set("power", 2.0);
    lp.set("scale", 1.0);
    lp.set("shift", 0.0);
    Ptr<Layer> layer = PowerLayer::create(lp);

    float value = std::pow(inp_value, 2.0);
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Exp)
{

    lp.type = "Exp";
    lp.name = "expLayer";
    Ptr<Layer> layer = ExpLayer::create(lp);

    float out_value = std::exp(inp_value);
    Mat output_ref(dims, output_shape.data(), CV_32F, out_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Sign)
{

    lp.type = "Sign";
    lp.name = "signLayer";
    Ptr<Layer> layer = SignLayer::create(lp);

    float value = inp_value > 0 ? 1.0 : 0.0;
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Shrink)
{

    lp.type = "Shrink";
    lp.name = "shrinkLayer";
    lp.set("lambda", 0.5);
    lp.set("bias", 0.5);
    Ptr<Layer> layer = ShrinkLayer::create(lp);

    float value = inp_value > 0.5 ? inp_value - 0.5 : 0.0;
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, ChannelsPReLU)
{

    lp.type = "ChannelsPReLU";
    lp.name = "ChannelsPReLULayer";
    Mat alpha = Mat(1, 3, CV_32F, 0.5);
    lp.blobs.push_back(alpha);
    Ptr<Layer> layer = ChannelsPReLULayer::create(lp);

    float value = inp_value > 0 ? inp_value : 0.5 * inp_value;
    Mat output_ref(dims, output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Test_01D, Values(0, 1));

typedef testing::TestWithParam<tuple<int, int>> Layer_Gather_Test;
TEST_P(Layer_Gather_Test, Accuracy_01D) {

    int dims = get<0>(GetParam());
    int axis = get<1>(GetParam());

    LayerParams lp;
    lp.type = "Gather";
    lp.name = "GatherLayer";
    lp.set("axis", axis);
    lp.set("real_ndims", 1);

    Ptr<GatherLayer> layer = GatherLayer::create(lp);

    std::vector<int> input_shape = {dims};
    std::vector<int> indices_shape = {1};
    std::vector<int> output_shape = {dims};

    Mat input(dims, input_shape.data(), CV_32F, 1.0);
    cv::randu(input, 0.0, 1.0);

    Mat indices(indices_shape, CV_32SC1, 0.0);
    Mat output_ref(dims, output_shape.data(), CV_32F, input.at<float>(0, 0));

    std::vector<Mat> inputs{input, indices};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(output_ref.size, outputs[0].size);
    normAssert(output_ref, outputs[0]);
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Gather_Test, Combine(
/*input blob shape*/    Values(0, 1),
/*operation*/           Values(0)
));

typedef testing::TestWithParam<tuple<int, std::string>> Layer_Arg_Test;
TEST_P(Layer_Arg_Test, Accuracy_01D) {

    int dims = get<0>(GetParam());
    std::string operation = get<1>(GetParam());

    LayerParams lp;
    lp.type = "Arg";
    lp.name = "arg" + operation + "_Layer";
    lp.set("op", operation);
    lp.set("axis", 0);
    lp.set("keepdims", 0);
    lp.set("select_last_index", 0);

    Ptr<ArgLayer> layer = ArgLayer::create(lp);
    std::vector<int> input_shape = {dims};
    std::vector<int> output_shape = {1};

    Mat input(dims, input_shape.data(), CV_32F, 1);
    Mat output_ref(dims, output_shape.data(),  CV_32F, 0);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);
    ASSERT_EQ(output_ref.size , outputs[0].size);
    normAssert(output_ref, outputs[0]);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Arg_Test, Combine(
/*input blob shape*/    Values(0, 1),
/*operation*/           Values( "max", "min")
));

typedef testing::TestWithParam<tuple<int, std::string>> Layer_NaryElemwise_Test;
TEST_P(Layer_NaryElemwise_Test, Accuracy_01D) {

    int dims = get<0>(GetParam());
    std::string operation = get<1>(GetParam());

    LayerParams lp;
    lp.type = "Eltwise";
    lp.name = operation + "_Layer";
    lp.set("operation", operation);
    Ptr<NaryEltwiseLayer> layer = NaryEltwiseLayer::create(lp);

    std::vector<int> input_shape = {dims};
    Mat input1(dims, input_shape.data(), CV_32F, 0.0);
    Mat input2(dims, input_shape.data(), CV_32F, 0.0);
    cv::randu(input1, 0.0, 1.0);
    cv::randu(input2, 0.0, 1.0);

    Mat output_ref;
    if (operation == "sum") {
        output_ref = input1 + input2;
    } else if (operation == "mul") {
        output_ref = input1.mul(input2);
    } else if (operation == "div") {
        output_ref = input1 / input2;
    } else if (operation == "sub") {
        output_ref = input1 - input2;
    } else {
        output_ref = Mat();
    }
    std::vector<Mat> inputs{input1, input2};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    if (!output_ref.empty()) {
        ASSERT_EQ(shape(output_ref), shape(outputs[0]));
        normAssert(output_ref, outputs[0]);
    } else {
        CV_Error(Error::StsAssert, "Provided operation: " + operation + " is not supported. Please check the test instantiation.");
    }
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_NaryElemwise_Test, Combine(
/*input blob shape*/    Values(0, 1),
/*operation*/           Values("div", "mul", "sum", "sub")
));

typedef testing::TestWithParam<tuple<int, std::string>> Layer_Elemwise_Test;
TEST_P(Layer_Elemwise_Test, Accuracy_01D) {

    int dims = get<0>(GetParam());
    std::string operation = get<1>(GetParam());

    LayerParams lp;
    lp.type = "Eltwise";
    lp.name = operation + "_Layer";
    lp.set("operation", operation);
    Ptr<EltwiseLayer> layer = EltwiseLayer::create(lp);

    std::vector<int> input_shape = {dims};
    Mat input1(dims, input_shape.data(), CV_32F);
    Mat input2(dims, input_shape.data(), CV_32F);
    cv::randu(input1, 0.0, 1.0);
    cv::randu(input2, 0.0, 1.0);

    // Dynamically select the operation
    Mat output_ref;
    if (operation == "sum") {
        output_ref = input1 + input2;
    } else if (operation == "max") {
        output_ref = cv::max(input1, input2);
    } else if (operation == "min") {
        output_ref = cv::min(input1, input2);
    } else if (operation == "prod") {
        output_ref = input1.mul(input2);
    } else if (operation == "div") {
        output_ref = input1 / input2;
    } else {
        output_ref = Mat();
    }

    std::vector<Mat> inputs{input1, input2};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);

    if (!output_ref.empty()) {
        ASSERT_EQ(shape(output_ref), shape(outputs[0]));
        normAssert(output_ref, outputs[0]);
    } else {
        CV_Error(Error::StsAssert, "Provided operation: " + operation + " is not supported. Please check the test instantiation.");
    }
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Elemwise_Test, Combine(
/*input blob shape*/    Values(0, 1),
/*operation*/           Values("div", "prod", "max", "min", "sum")
));

TEST(Layer_Reshape_Test, Accuracy)
{
    LayerParams lp;
    lp.type = "Reshape";
    lp.name = "ReshapeLayer";
    lp.set("axis", 0); // Set axis to 0 to start reshaping from the first dimension
    lp.set("num_axes", -1); // Set num_axes to -1 to indicate all following axes are included in the reshape
    int newShape[] = {1};
    lp.set("dim", DictValue::arrayInt(newShape, 1));

    Ptr<ReshapeLayer> layer = ReshapeLayer::create(lp);

    std::vector<int> input_shape = {0};

    Mat input(0, input_shape.data(), CV_32F);
    randn(input, 0.0, 1.0);
    Mat output_ref(1, newShape, CV_32F, input.data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

typedef testing::TestWithParam<tuple<std::vector<int>>> Layer_Split_Test;
TEST_P(Layer_Split_Test, Accuracy_01D)
{
    LayerParams lp;
    lp.type = "Split";
    lp.name = "SplitLayer";
    int top_count = 2; // 2 is for simplicity
    lp.set("top_count", top_count);
    Ptr<SplitLayer> layer = SplitLayer::create(lp);

    std::vector<int> input_shape = std::get<0>(GetParam());

    Mat input(input_shape.size(), input_shape.data(), CV_32F);
    cv::randn(input, 0.0, 1.0);

    Mat output_ref = Mat(input_shape.size(), input_shape.data(), CV_32F, input.data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);
    for (int i = 0; i < top_count; i++)
    {
        ASSERT_EQ(shape(output_ref), shape(outputs[i]));
        normAssert(output_ref, outputs[i]);
    }
}
INSTANTIATE_TEST_CASE_P(/*nothting*/, Layer_Split_Test,
                        testing::Values(
                            std::vector<int>({}),
                            std::vector<int>({1}),
                            std::vector<int>({1, 4}),
                            std::vector<int>({1, 5}),
                            std::vector<int>({4, 1}),
                            std::vector<int>({4, 5})
));

}}
