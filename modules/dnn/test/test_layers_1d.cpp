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

class Layer_Test_01D: public testing::TestWithParam<tuple<std::vector<int>>>
{
public:
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    float inp_value;
    Mat input;
    LayerParams lp;

    void SetUp()
    {
        input_shape = get<0>(GetParam());
        output_shape = input_shape;

        // generate random positeve value from 1 to 10
        RNG& rng = TS::ptr()->get_rng();
        inp_value = rng.uniform(1.0, 10.0); // random uniform value
        input = Mat(input_shape.size(), input_shape.data(), CV_32F, inp_value);
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
    lp.name = "ScaleLayer";
    lp.set("axis", 0);
    lp.set("mode", "scale");
    lp.set("bias_term", false);
    Ptr<ScaleLayer> layer = ScaleLayer::create(lp);

    Mat weight = Mat(output_shape.size(), output_shape.data(), CV_32F, 2.0);
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

    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, 1.0);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Clip)
{

    lp.type = "Clip";
    lp.name = "ClipLayer";
    lp.set("min_value", 0.0);
    lp.set("max_value", 1.0);
    Ptr<ReLU6Layer> layer = ReLU6Layer::create(lp);

    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, 1.0);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, ReLU)
{

    lp.type = "ReLU";
    lp.name = "ReluLayer";
    lp.set("negative_slope", 0.0);
    Ptr<ReLULayer> layer = ReLULayer::create(lp);

    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, inp_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Gelu)
{

    lp.type = "Gelu";
    lp.name = "GeluLayer";
    Ptr<GeluLayer> layer = GeluLayer::create(lp);

    float value = inp_value * 0.5 * (std::erf(inp_value * 1 / std::sqrt(2.0)) + 1.0);
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, GeluApprox)
{

    lp.type = "GeluApprox";
    lp.name = "GeluApproxLayer";
    Ptr<GeluApproximationLayer> layer = GeluApproximationLayer::create(lp);

    float value = inp_value * 0.5 * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (inp_value + 0.044715 * std::pow(inp_value, 3))));
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Sigmoid)
{

    lp.type = "Sigmoid";
    lp.name = "SigmoidLayer";
    Ptr<SigmoidLayer> layer = SigmoidLayer::create(lp);

    float value = 1.0 / (1.0 + std::exp(-inp_value));
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Tanh)
{

    lp.type = "TanH";
    lp.name = "TanHLayer";
    Ptr<Layer> layer = TanHLayer::create(lp);


    float value = std::tanh(inp_value);
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Swish)
{

    lp.type = "Swish";
    lp.name = "SwishLayer";
    Ptr<Layer> layer = SwishLayer::create(lp);

    float value = inp_value / (1 + std::exp(-inp_value));
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Mish)
{

    lp.type = "Mish";
    lp.name = "MishLayer";
    Ptr<Layer> layer = MishLayer::create(lp);

    float value = inp_value * std::tanh(std::log(1 + std::exp(inp_value)));
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, ELU)
{

    lp.type = "ELU";
    lp.name = "EluLayer";
    lp.set("alpha", 1.0);
    Ptr<Layer> layer = ELULayer::create(lp);

    float value = inp_value > 0 ? inp_value : std::exp(inp_value) - 1;
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Abs)
{

    lp.type = "Abs";
    lp.name = "AbsLayer";
    Ptr<Layer> layer = AbsLayer::create(lp);

    float value = std::abs(inp_value);
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, BNLL)
{

    lp.type = "BNLL";
    lp.name = "BNLLLayer";
    Ptr<Layer> layer = BNLLLayer::create(lp);

    float value = std::log(1 + std::exp(inp_value));
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Ceil)
{

    lp.type = "Ceil";
    lp.name = "CeilLayer";
    Ptr<Layer> layer = CeilLayer::create(lp);

    float value = std::ceil(inp_value);
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Floor)
{

    lp.type = "Floor";
    lp.name = "FloorLayer";
    Ptr<Layer> layer = FloorLayer::create(lp);

    float value = std::floor(inp_value);
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Log)
{

    lp.type = "Log";
    lp.name = "LogLayer";
    Ptr<Layer> layer = LogLayer::create(lp);

    float value = std::log(inp_value);
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Round)
{

    lp.type = "Round";
    lp.name = "RoundLayer";
    Ptr<Layer> layer = RoundLayer::create(lp);

    float value = std::round(inp_value);
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Sqrt)
{

    lp.type = "Sqrt";
    lp.name = "SqrtLayer";
    Ptr<Layer> layer = SqrtLayer::create(lp);

    float value = std::sqrt(inp_value);
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Acos)
{

    lp.type = "Acos";
    lp.name = "AcosLayer";
    Ptr<Layer> layer = AcosLayer::create(lp);

    inp_value = 0.5 + static_cast <float> (inp_value) / (static_cast <float> (RAND_MAX/(1-0.5)));
    input = Mat(input_shape.size(), input_shape.data(), CV_32F, inp_value);

    float value = std::acos(inp_value);
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Acosh)
{

    lp.type = "Acosh";
    lp.name = "AcoshLayer";
    Ptr<Layer> layer = AcoshLayer::create(lp);

    float value = std::acosh(inp_value);
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Asin)
{

    lp.type = "Asin";
    lp.name = "AsinLayer";
    Ptr<Layer> layer = AsinLayer::create(lp);

    inp_value = 0.5 + static_cast <float> (inp_value) / (static_cast <float> (RAND_MAX/(1-0.5)));
    input = Mat(input_shape.size(), input_shape.data(), CV_32F, inp_value);

    float value = std::asin(inp_value);
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Asinh)
{

    lp.type = "Asinh";
    lp.name = "AsinhLayer";
    Ptr<Layer> layer = AsinhLayer::create(lp);

    float value = std::asinh(inp_value);
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Atan)
{

    lp.type = "Atan";
    lp.name = "AtanLayer";
    Ptr<Layer> layer = AtanLayer::create(lp);

    float value = std::atan(inp_value);
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Cos)
{

    lp.type = "Cos";
    lp.name = "CosLayer";
    Ptr<Layer> layer = CosLayer::create(lp);

    float value = std::cos(inp_value);
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Cosh)
{

    lp.type = "Cosh";
    lp.name = "CoshLayer";
    Ptr<Layer> layer = CoshLayer::create(lp);

    float value = std::cosh(inp_value);
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Sin)
{

    lp.type = "Sin";
    lp.name = "SinLayer";
    Ptr<Layer> layer = SinLayer::create(lp);

    float value = std::sin(inp_value);
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Sinh)
{

    lp.type = "Sinh";
    lp.name = "SinhLayer";
    Ptr<Layer> layer = SinhLayer::create(lp);

    float value = std::sinh(inp_value);
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Tan)
{

    lp.type = "Tan";
    lp.name = "TanLayer";
    Ptr<Layer> layer = TanLayer::create(lp);

    float value = std::tan(inp_value);
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Erf)
{

    lp.type = "Erf";
    lp.name = "ErfLayer";
    Ptr<Layer> layer = ErfLayer::create(lp);

    float out_value = std::erf(inp_value);
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, out_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Reciprocal)
{

    lp.type = "Reciprocal";
    lp.name = "ReciprocalLayer";
    Ptr<Layer> layer = ReciprocalLayer::create(lp);

    float out_value = 1/inp_value;
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, out_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, HardSwish)
{

    lp.type = "HardSwish";
    lp.name = "HardSwishLayer";
    Ptr<Layer> layer = HardSwishLayer::create(lp);

    float out_value = inp_value * std::max(0.0f, std::min(6.0f, inp_value + 3.0f)) / 6.0f;
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, out_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Softplus)
{

    lp.type = "Softplus";
    lp.name = "SoftplusLayer";
    Ptr<Layer> layer = SoftplusLayer::create(lp);

    float out_value = std::log(1 + std::exp(inp_value));
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, out_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, SoftSign)
{

    lp.type = "Softsign";
    lp.name = "SoftsignLayer";
    Ptr<Layer> layer = SoftsignLayer::create(lp);

    float out_value = inp_value / (1 + std::abs(inp_value));
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, out_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, CELU)
{

    lp.type = "CELU";
    lp.name = "CeluLayer";
    lp.set("alpha", 1.0);
    Ptr<Layer> layer = CeluLayer::create(lp);

    float out_value = inp_value < 0 ? std::exp(inp_value) - 1 : inp_value;
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, out_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, HardSigmoid)
{

    lp.type = "HardSigmoid";
    lp.name = "HardSigmoidLayer";
    Ptr<Layer> layer = HardSigmoidLayer::create(lp);

    float out_value = std::max(0.0f, std::min(1.0f, 0.2f * inp_value + 0.5f));
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, out_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, SELU)
{

    lp.type = "SELU";
    lp.name = "SeluLayer";
    lp.set("alpha", 1.6732631921768188);
    lp.set("gamma", 1.0507009873554805);
    Ptr<Layer> layer = SeluLayer::create(lp);


    double inp_value_double = static_cast<double>(inp_value); // Ensure the input is treated as double for the computation

    double value_double = 1.0507009873554805 * (inp_value_double > 0 ? inp_value_double : 1.6732631921768188 * (std::exp(inp_value_double / 1.0) - 1));

    float value = static_cast<float>(value_double);

    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, ThresholdedReLU)
{

    lp.type = "ThresholdedRelu";
    lp.name = "ThresholdedReluLayer";
    lp.set("alpha", 1.0);
    Ptr<Layer> layer = ThresholdedReluLayer::create(lp);

    float value = inp_value > 1.0 ? inp_value : 0.0;
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Power)
{

    lp.type = "Power";
    lp.name = "PowerLayer";
    lp.set("power", 2.0);
    lp.set("scale", 1.0);
    lp.set("shift", 0.0);
    Ptr<Layer> layer = PowerLayer::create(lp);

    float value = std::pow(inp_value, 2.0);
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Exp)
{

    lp.type = "Exp";
    lp.name = "ExpLayer";
    Ptr<Layer> layer = ExpLayer::create(lp);

    float out_value = std::exp(inp_value);
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, out_value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Sign)
{

    lp.type = "Sign";
    lp.name = "SignLayer";
    Ptr<Layer> layer = SignLayer::create(lp);

    float value = inp_value > 0 ? 1.0 : 0.0;
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}

TEST_P(Layer_Test_01D, Shrink)
{

    lp.type = "Shrink";
    lp.name = "ShrinkLayer";
    lp.set("lambda", 0.5);
    lp.set("bias", 0.5);
    Ptr<Layer> layer = ShrinkLayer::create(lp);

    float value = inp_value > 0.5 ? inp_value - 0.5 : 0.0;
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
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
    Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, value);
    std::vector<Mat> inputs{input};

    TestLayer(layer, inputs, output_ref);
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Test_01D,
        testing::Values(
            std::vector<int>{},
            std::vector<int>{1}
        ));

typedef testing::TestWithParam<tuple<std::vector<int>, int>> Layer_Gather_Test;
TEST_P(Layer_Gather_Test, Accuracy_01D) {

    std::vector<int> input_shape = get<0>(GetParam());
    int axis = get<1>(GetParam());

    // skip case when axis > input shape
    if (axis > input_shape.size())
        return;

    LayerParams lp;
    lp.type = "Gather";
    lp.name = "GatherLayer";
    lp.set("axis", axis);
    lp.set("real_ndims", 1);
    Ptr<GatherLayer> layer = GatherLayer::create(lp);

    cv::Mat input(input_shape.size(), input_shape.data(), CV_32F);
    cv::randu(input, 0.0, 1.0);

    std::vector<int> indices_shape = {1};
    cv::Mat indices = cv::Mat(indices_shape.size(), indices_shape.data(), CV_32S, 0.0);

    cv::Mat output_ref;
    if (input_shape.size() == 0 || input_shape.size() == 1){
        output_ref = input;
    } else if (axis == 0){
        output_ref = input.row(0);
    } else if (axis == 1){
        output_ref = input.col(0);
    }

    std::vector<Mat> inputs{input, indices};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(1, outputs.size());
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Gather_Test, Combine(
/*input blob shape*/    testing::Values(
                                std::vector<int>({}),
                                std::vector<int>({1}),
                                std::vector<int>({1, 4}),
                                std::vector<int>({4, 4})
                                ),
/*axis*/           testing::Values(0, 1)
));

template <typename T>
int arg_op(const std::vector<T>& vec, const std::string& operation) {
    CV_Assert(!vec.empty());
    if (operation == "max") {
        return static_cast<int>(std::distance(vec.begin(), std::max_element(vec.begin(), vec.end())));
    } else if (operation == "min") {
        return static_cast<int>(std::distance(vec.begin(), std::min_element(vec.begin(), vec.end())));
    } else {
        CV_Error(Error::StsAssert, "Provided operation: " + operation + " is not supported. Please check the test instantiation.");
    }
}
// Test for ArgLayer is disabled because there problem in runLayer function related to type assignment
typedef testing::TestWithParam<tuple<std::vector<int>, std::string>> Layer_Arg_Test;
TEST_P(Layer_Arg_Test, Accuracy_01D) {
    std::vector<int> input_shape = get<0>(GetParam());
    std::string operation = get<1>(GetParam());

    LayerParams lp;
    lp.type = "Arg";
    lp.name = "Arg" + operation + "_Layer";
    int axis = (input_shape.size() == 0 || input_shape.size() == 1 ) ? 0 : 1;
    lp.set("op", operation);
    lp.set("axis", axis);
    lp.set("keepdims", 1);
    lp.set("select_last_index", 0);

    Ptr<ArgLayer> layer = ArgLayer::create(lp);

    cv::Mat input = cv::Mat(input_shape.size(), input_shape.data(), CV_32F);
    for (int i = 0; i < input.total(); i++){
        input.at<float>(i) = i;
    }

    // create reference output with required shape and values
    int index;
    cv::Mat output_ref;
    std::vector<int> ref_output;
    if (input_shape.size() == 2 ){
        int rows = input_shape[0];
        int cols = input_shape[1];
        ref_output.resize(rows);
        for (int i = 0; i < rows; i++) {
            std::vector<float> row_vec(cols);
            for (int j = 0; j < cols; j++) {
                row_vec[j] = input.at<float>(i, j);
            }
            ref_output[i] = (int) arg_op(row_vec, operation);
        }
        output_ref = cv::Mat(rows, (axis == 1) ? 1 : cols, CV_32S, ref_output.data());
    } else if (input_shape.size() <= 1) {
        index = arg_op(std::vector<float>(input.begin<float>(), input.end<float>()), operation);
        output_ref = cv::Mat(input_shape.size(), input_shape.data(), CV_32FC1, &index);
    }

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(1, outputs.size());
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    // convert output_ref to float to match the output type
    output_ref.convertTo(output_ref, CV_64SC1);
    normAssert(output_ref, outputs[0]);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Arg_Test, Combine(
/*input blob shape*/    testing::Values(
                                std::vector<int>({}),
                                std::vector<int>({1}),
                                std::vector<int>({1, 4}),
                                std::vector<int>({4, 4})
                                ),
/*operation*/           Values( "max", "min")
));

typedef testing::TestWithParam<tuple<std::vector<int>, std::string>> Layer_NaryElemwise_1d_Test;
TEST_P(Layer_NaryElemwise_1d_Test, Accuracy) {

    std::vector<int> input_shape = get<0>(GetParam());
    std::string operation = get<1>(GetParam());

    LayerParams lp;
    lp.type = "NaryEltwise";
    lp.name = operation + "_Layer";
    lp.set("operation", operation);
    Ptr<NaryEltwiseLayer> layer = NaryEltwiseLayer::create(lp);

    cv::Mat input1 = cv::Mat(input_shape.size(), input_shape.data(), CV_32F);
    cv::Mat input2 = cv::Mat(input_shape.size(), input_shape.data(), CV_32F);
    cv::randu(input1, 0.0, 1.0);
    cv::randu(input2, 0.0, 1.0);

    cv::Mat output_ref;
    if (operation == "sum") {
        output_ref = input1 + input2;
    } else if (operation == "mul") {
        output_ref = input1.mul(input2);
    } else if (operation == "div") {
        output_ref = input1 / input2;
    } else if (operation == "sub") {
        output_ref = input1 - input2;
    } else {
        output_ref = cv::Mat();
    }
    std::vector<Mat> inputs{input1, input2};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    if (!output_ref.empty()) {
        ASSERT_EQ(1, outputs.size());
        ASSERT_EQ(shape(output_ref), shape(outputs[0]));
        normAssert(output_ref, outputs[0]);
    } else {
        CV_Error(Error::StsAssert, "Provided operation: " + operation + " is not supported. Please check the test instantiation.");
    }
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_NaryElemwise_1d_Test, Combine(
/*input blob shape*/    testing::Values(
                            std::vector<int>({}),
                            std::vector<int>({1}),
                            std::vector<int>({1, 4}),
                            std::vector<int>({4, 1})),
/*operation*/           testing::Values("div", "mul", "sum", "sub")
));

typedef testing::TestWithParam<tuple<std::vector<int>, std::string>> Layer_Elemwise_1d_Test;
TEST_P(Layer_Elemwise_1d_Test, Accuracy_01D) {

    std::vector<int> input_shape = get<0>(GetParam());
    std::string operation = get<1>(GetParam());

    LayerParams lp;
    lp.type = "Eltwise";
    lp.name = operation + "_Layer";
    lp.set("operation", operation);
    Ptr<EltwiseLayer> layer = EltwiseLayer::create(lp);

    cv::Mat input1(input_shape.size(), input_shape.data(), CV_32F);
    cv::Mat input2(input_shape.size(), input_shape.data(), CV_32F);
    cv::randu(input1, 0.0, 1.0);
    cv::randu(input2, 0.0, 1.0);

    // Dynamically select the operation
    cv::Mat output_ref;
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
        output_ref = cv::Mat();
    }

    std::vector<Mat> inputs{input1, input2};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    if (!output_ref.empty()) {
        ASSERT_EQ(1, outputs.size());
        ASSERT_EQ(shape(output_ref), shape(outputs[0]));
        normAssert(output_ref, outputs[0]);
    } else {
        CV_Error(Error::StsAssert, "Provided operation: " + operation + " is not supported. Please check the test instantiation.");
    }
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Elemwise_1d_Test, Combine(
/*input blob shape*/    testing::Values(
                                    std::vector<int>({}),
                                    std::vector<int>({1}),
                                    std::vector<int>({4}),
                                    std::vector<int>({1, 4}),
                                    std::vector<int>({4, 1})),
/*operation*/           testing::Values("div", "prod", "max", "min", "sum")
));

TEST(Layer_Reshape_Test, Accuracy_1D)
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
    ASSERT_EQ(1, outputs.size());
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
    ASSERT_EQ(outputs.size(), top_count);
    for (int i = 0; i < top_count; i++)
    {
        ASSERT_EQ(shape(outputs[i]), shape(output_ref));
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

typedef testing::TestWithParam<tuple<std::vector<int>, std::vector<int>>> Layer_Expand_Test;
TEST_P(Layer_Expand_Test, Accuracy_ND) {

    std::vector<int> input_shape = get<0>(GetParam());
    std::vector<int> target_shape = get<1>(GetParam());
    if (input_shape.size() >= target_shape.size()) // Skip if input shape is already larger than target shape
        return;

    LayerParams lp;
    lp.type = "Expand";
    lp.name = "ExpandLayer";
    lp.set("shape", DictValue::arrayInt(&target_shape[0], target_shape.size()));

    Ptr<ExpandLayer> layer = ExpandLayer::create(lp);
    Mat input(input_shape.size(), input_shape.data(), CV_32F);
    cv::randn(input, 0.0, 1.0);

    cv::Mat output_ref(target_shape, CV_32F, input.data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(1, outputs.size());
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Expand_Test, Combine(
/*input blob shape*/ testing::Values(
        std::vector<int>({}),
        std::vector<int>({1}),
        std::vector<int>({1, 1}),
        std::vector<int>({1, 1, 1})
    ),
/*output blob shape*/ testing::Values(
        std::vector<int>({1}),
        std::vector<int>({1, 1}),
        std::vector<int>({1, 1, 1}),
        std::vector<int>({1, 1, 1, 1})
    )
));

typedef testing::TestWithParam<tuple<std::vector<int>>> Layer_Concat_Test;
TEST_P(Layer_Concat_Test, Accuracy_01D)
{
    LayerParams lp;
    lp.type = "Concat";
    lp.name = "ConcatLayer";
    lp.set("axis", 0);

    Ptr<ConcatLayer> layer = ConcatLayer::create(lp);

    std::vector<int> input_shape = get<0>(GetParam());
    std::vector<int> output_shape = {3};

    Mat input1(input_shape.size(), input_shape.data(), CV_32F, 1.0);
    Mat input2(input_shape.size(), input_shape.data(), CV_32F, 2.0);
    Mat input3(input_shape.size(), input_shape.data(), CV_32F, 3.0);

    float data[] = {1.0, 2.0, 3.0};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input1, input2, input3};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(1, outputs.size());
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Concat_Test,
/*input blob shape*/    testing::Values(
    // ONNX Concat produces output tensor of the same dimensionality as inputs.
    // Therefore 0-dimensional tensors cannot be concatenated.
    // They first need to be converted to 1D tensors, e.g. using Unsqueeze.
    //std::vector<int>({}),
    std::vector<int>({1})
));

typedef testing::TestWithParam<tuple<std::vector<int>, int>> Layer_Softmax_Test;
TEST_P(Layer_Softmax_Test, Accuracy_01D) {

    int axis = get<1>(GetParam());
    std::vector<int> input_shape = get<0>(GetParam());
    if ((input_shape.size() == 0 && axis == 1) ||
        (!input_shape.empty() && input_shape.size() == 2 && input_shape[0] > 1 && axis == 1) ||
        (!input_shape.empty() && input_shape[0] > 1 && axis == 0)) // skip since not valid case
        return;

    LayerParams lp;
    lp.type = "Softmax";
    lp.name = "softmaxLayer";
    lp.set("axis", axis);
    Ptr<SoftmaxLayer> layer = SoftmaxLayer::create(lp);

    Mat input = Mat(input_shape.size(), input_shape.data(), CV_32F);
    cv::randn(input, 0.0, 1.0);

    Mat output_ref;
    cv::exp(input, output_ref);
    if (axis == 1){
        cv::divide(output_ref, cv::sum(output_ref), output_ref);
    } else {
        cv::divide(output_ref, output_ref, output_ref);
    }

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);
    ASSERT_EQ(1, outputs.size());
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Softmax_Test, Combine(
    /*input blob shape*/
    testing::Values(
        std::vector<int>({}),
        std::vector<int>({1}),
        std::vector<int>({4}),
        std::vector<int>({1, 4}),
        std::vector<int>({4, 1})
        ),
    /*Axis */
    testing::Values(0, 1)
));

typedef testing::TestWithParam<std::tuple<std::tuple<int, std::vector<int>>, std::string>> Layer_Scatter_Test;
TEST_P(Layer_Scatter_Test, Accuracy1D) {
    auto tup = get<0>(GetParam());
    int axis = get<0>(tup);
    std::vector<int> input_shape = get<1>(tup);
    std::string opr = get<1>(GetParam());

    LayerParams lp;
    lp.type = "Scatter";
    lp.name = "ScatterLayer";
    lp.set("axis", axis);
    lp.set("reduction", opr);
    Ptr<ScatterLayer> layer = ScatterLayer::create(lp);

    cv::Mat input = cv::Mat(input_shape.size(), input_shape.data(), CV_32F);
    cv::randn(input, 0.0, 1.0);

    int indices[] = {3, 2, 1, 0};
    cv::Mat indices_mat(input_shape.size(), input_shape.data(), CV_32S, indices);
    cv::Mat output(input_shape.size(), input_shape.data(), CV_32F, 0.0);

    // create reference output
    cv::Mat output_ref(input_shape, CV_32F, 0.0);
    for (int i = 0; i < ((input_shape.size() == 1) ? input_shape[0] : input_shape[1]); i++){
        output_ref.at<float>(indices[i]) = input.at<float>(i);
    }

    if (opr == "add"){
        output_ref += output;
    } else if (opr == "mul"){
        output_ref = output.mul(output_ref);
    } else if (opr == "max"){
        cv::max(output_ref, output, output_ref);
    } else if (opr == "min"){
        cv::min(output_ref, output, output_ref);
    }

    std::vector<Mat> inputs{output, indices_mat, input};
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);
    ASSERT_EQ(1, outputs.size());
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Scatter_Test, Combine(
/*input blob shape*/    testing::Values(std::make_tuple(0, std::vector<int>{4}),
                                        std::make_tuple(1, std::vector<int>{1, 4})),
/*reduce*/              testing::Values("none", "add", "mul", "max", "min")
));



typedef testing::TestWithParam<tuple<std::vector<int>, std::string, int>> Layer_Reduce_Test;
TEST_P(Layer_Reduce_Test, Accuracy_01D)
{
    auto reduceOperation = [](const cv::Mat& input, const std::string& operation, int axis) -> cv::Mat {
        // Initialize result matrix
        cv::Mat result;
        MatShape inpshape = input.shape();
        if (inpshape.dims == 0) {
            result = cv::Mat(0, nullptr, CV_32F);
        } else if (inpshape.dims == 1) {
            result = cv::Mat({1}, CV_32F);
        } else {
            if (axis == 0) {
                result = cv::Mat::zeros(1, input.cols, CV_32F);
            } else {
                result = cv::Mat::zeros(input.rows, 1, CV_32F);
            }
        }

        auto process_value = [&](float& res, float value, bool is_first) {
            if (operation == "max") {
                res = is_first ? value : std::max(res, value);
            } else if (operation == "min") {
                res = is_first ? value : std::min(res, value);
            } else {
                if (is_first) {
                    if (operation == "sum" || operation == "l1" || operation == "l2"
                        || operation == "sum_square" || operation == "mean" || operation == "log_sum"
                        || operation == "log_sum_exp") res = 0;
                    else if (operation == "prod") res = 1;
                }

                if (operation == "sum" || operation == "mean") res += value;
                else if (operation == "sum_square") {
                        res += value * value;
                } else if (operation == "l1") res += std::abs(value);
                else if (operation == "l2") res += value * value;
                else if (operation == "prod") res *= value;
                else if (operation == "log_sum") res += value;
                else if (operation == "log_sum_exp") res += std::exp(value);
            }
        };

        for (int r = 0; r < input.rows; ++r) {
            for (int c = 0; c < input.cols; ++c) {
                float value = input.at<float>(r, c);
                if (shape(input).size() == 1 && shape(input)[0] != 1 && axis == 0){
                        process_value(result.at<float>(0, 0), value, c == 0);
                } else {
                    if (axis == 0) {
                        process_value(result.at<float>(0, c), value, r == 0);
                    } else {
                        process_value(result.at<float>(r, 0), value, c == 0);
                    }
                }
            }
        }

        if (operation == "mean") {
            if (shape(input).size() == 1 && shape(input)[0] != 1 && axis == 0){
                result.at<float>(0, 0) /= input.cols;
            } else {
            if (axis == 0) {
                    result /= input.rows;
                } else {
                    result /= input.cols;
                }
            }
        } else if (operation == "l2") {
            cv::sqrt(result, result);
        } else if (operation == "log_sum_exp" || operation == "log_sum") {
            cv::log(result, result);
        }

        return result;
    };

    std::vector<int> input_shape = get<0>(GetParam());
    std::string reduce_operation = get<1>(GetParam());
    int axis = get<2>(GetParam());

    if ((input_shape.size() == 2 && reduce_operation == "log_sum") ||
        (axis > input_shape.size())) // both output and reference are nans
        return;

    LayerParams lp;
    lp.type = "Reduce";
    lp.name = "reduceLayer";
    lp.set("reduce", reduce_operation);

    // for scalar tensors we cannot specify reduction axis,
    // because it will be out-of-range anyway
    if (!input_shape.empty())
        lp.set("axes", axis);

    lp.set("keepdims", true);
    Ptr<ReduceLayer> layer = ReduceLayer::create(lp);

    cv::Mat input((int)input_shape.size(), input_shape.data(), CV_32F, 1.0);
    cv::randu(input, 0.0, 1.0);

    cv::Mat output_ref = reduceOperation(input, reduce_operation, axis);
    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);

    MatShape ref_shape = output_ref.shape();
    MatShape out_shape = outputs[0].shape();
    if (ref_shape != out_shape) {
        printf("ref shape: %s\n", ref_shape.str().c_str());
        printf("out shape: %s\n", out_shape.str().c_str());
    }
    ASSERT_EQ(ref_shape, out_shape);
    normAssert(output_ref, outputs[0]);
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Reduce_Test, Combine(
/*input blob shape*/    Values(
    std::vector<int>({}),
    std::vector<int>({1}),
    std::vector<int>({4}),
    std::vector<int>({1, 4}),
    std::vector<int>({4, 1}),
    std::vector<int>({4, 4})
    ),
/*reduce operation type*/
    Values("max", "min", "mean", "sum", "sum_square", "l1", "l2", "prod", "log_sum", "log_sum_exp"),
    Values(0, 1))
);


typedef testing::TestWithParam<tuple<std::vector<int>>> Layer_Permute_Test;
TEST_P(Layer_Permute_Test, Accuracy_01D)
{
    LayerParams lp;
    lp.type = "Permute";
    lp.name = "PermuteLayer";

    int order[] = {0}; // Since it's a 0D tensor, the order remains [0]
    lp.set("order", DictValue::arrayInt(order, 1));
    Ptr<PermuteLayer> layer = PermuteLayer::create(lp);

    std::vector<int> input_shape = get<0>(GetParam());

    Mat input = Mat(input_shape.size(), input_shape.data(), CV_32F);
    cv::randn(input, 0.0, 1.0);
    Mat output_ref = input.clone();

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(1, outputs.size());
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}
INSTANTIATE_TEST_CASE_P(/*nothing*/,  Layer_Permute_Test,
/*input blob shape*/ testing::Values(
            std::vector<int>{},
            std::vector<int>{1},
            std::vector<int>{1, 4},
            std::vector<int>{4, 1}
));

typedef testing::TestWithParam<tuple<std::vector<int>>> Layer_Slice_Test;
TEST_P(Layer_Slice_Test, Accuracy_1D){

    LayerParams lp;
    lp.type = "Slice";
    lp.name = "SliceLayer";

    std::vector<int> input_shape = get<0>(GetParam());

    int splits = 2;
    int axis = (input_shape.size() > 1 ) ? 1 : 0;

    lp.set("axis", axis);
    lp.set("num_split", splits);

    Ptr<SliceLayer> layer = SliceLayer::create(lp);
    std::vector<int> output_shape;
    if (input_shape.size() > 1)
        output_shape = {1, input_shape[1] / splits};
    else
        output_shape = {input_shape[0] / splits};

    cv::Mat input = cv::Mat(input_shape, CV_32F);
    cv::randu(input, 0.0, 1.0);

    std::vector<cv::Mat> output_refs;
    for (int i = 0; i < splits; ++i){
        output_refs.push_back(cv::Mat(output_shape, CV_32F));
        if (input_shape.size() > 1 ) {
            for (int j = 0; j < output_shape[1]; ++j){
                output_refs[i].at<float>(j) = input.at<float>(i * output_shape[1] + j);
            }
        } else {
            for (int j = 0; j < output_shape[0]; ++j){
                output_refs[i].at<float>(j) = input.at<float>(i * output_shape[0] + j);
            }
        }
    }

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);
    ASSERT_EQ(outputs.size(), splits);
    for (int i = 0; i < splits; ++i){
        ASSERT_EQ(shape(outputs[i]), shape(output_refs[i]));
        normAssert(output_refs[i], outputs[i]);
    }
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Slice_Test,
/*input blob shape*/    testing::Values(
                std::vector<int>({4}),
                std::vector<int>({1, 4})
));

typedef testing::TestWithParam<tuple<std::vector<int>>> Layer_Padding_Test;
TEST_P(Layer_Padding_Test, Accuracy_01D){

    std::vector<int> input_shape = get<0>(GetParam());
    float pad_value = 10;

    LayerParams lp;
    lp.type = "Padding";
    lp.name = "PaddingLayer";
    std::vector<int> paddings = {5, 3}; // Pad before and pad after for one dimension
    lp.set("paddings", DictValue::arrayInt(paddings.data(), paddings.size()));
    lp.set("value", pad_value);
    lp.set("input_dims", (input_shape.size() == 1) ? -1 : 0);
    Ptr<PaddingLayer> layer = PaddingLayer::create(lp);

    cv::Mat input(input_shape.size(), input_shape.data(), CV_32F);
    cv::randn(input, 0.0, 1.0);


    // Fill in the padding values manually
    // Create output ref shape depending on the input shape and input_dims
    std::vector<int> output_shape;
    if (input_shape.size() == 0){
        output_shape = {1 + paddings[0] + paddings[1]};
    } else if (input_shape.size() == 1){
        output_shape = {input_shape[0] + paddings[0] + paddings[1]};
    } else {
        output_shape = {input_shape[0], input_shape[1] + paddings[0] + paddings[1]};
    }

    cv::Mat output_ref(output_shape.size(), output_shape.data(), CV_32F, pad_value);

    if (input_shape.size() == 0){
        output_ref.at<float>(paddings[0]) = input.at<float>(0);
    } else if (input_shape.size() == 1){
        for (int i = 0; i < input_shape[0]; ++i){
            output_ref.at<float>(i + paddings[0]) = input.at<float>(i);
        }
    } else {
        for (int i = 0; i < input_shape[0]; ++i){
            for (int j = 0; j < input_shape[1]; ++j){
                output_ref.at<float>(i, j + paddings[0]) = input.at<float>(i, j);
            }
        }
    }

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);
    ASSERT_EQ(1, outputs.size());
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}
INSTANTIATE_TEST_CASE_P(/*nothing*/,  Layer_Padding_Test,
/*input blob shape*/ testing::Values(
            std::vector<int>{},
            std::vector<int>{1},
            std::vector<int>{1, 4},
            std::vector<int>{4, 1}
));

typedef testing::TestWithParam<tuple<std::vector<int>>> Layer_FullyConnected_Test;
TEST_P(Layer_FullyConnected_Test, Accuracy_01D)
{
    LayerParams lp;
    lp.type = "InnerProduct";
    lp.name = "InnerProductLayer";
    lp.set("num_output", 1);
    lp.set("bias_term", false);
    lp.set("axis", 0);

    MatShape input_shape(get<0>(GetParam()));

    RNG& rng = TS::ptr()->get_rng();
    float inp_value = rng.uniform(0.0, 10.0);
    Mat weights({(int)input_shape.total(), 1}, CV_32F, inp_value);
    lp.blobs.push_back(weights);

    Ptr<Layer> layer = LayerFactory::createLayerInstance("InnerProduct", lp);

    Mat input(input_shape, CV_32F);
    randn(input, 0, 1);
    Mat output_ref = input.reshape(1, 1) * weights;
    output_ref.dims = input_shape.dims;

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);
    ASSERT_EQ(1, outputs.size());
    ASSERT_EQ(output_ref.shape(), outputs[0].shape());
    normAssert(output_ref, outputs[0]);
}
INSTANTIATE_TEST_CASE_P(/*nothting*/, Layer_FullyConnected_Test,
                        testing::Values(
                            std::vector<int>({}),
                            std::vector<int>({1}),
                            std::vector<int>({4})
));

typedef testing::TestWithParam<std::vector<int>> Layer_BatchNorm_Test;
TEST_P(Layer_BatchNorm_Test, Accuracy_01D)
{
    std::vector<int> input_shape = GetParam();

    // Layer parameters
    LayerParams lp;
    lp.type = "BatchNorm";
    lp.name = "BatchNormLayer";
    lp.set("has_weight", false);
    lp.set("has_bias", false);

    RNG& rng = TS::ptr()->get_rng();
    float inp_value = rng.uniform(0.0, 10.0);

    Mat meanMat(input_shape.size(), input_shape.data(), CV_32F, inp_value);
    Mat varMat(input_shape.size(), input_shape.data(), CV_32F, inp_value);
    vector<Mat> blobs = {meanMat, varMat};
    lp.blobs = blobs;

    // Create the layer
    Ptr<Layer> layer = BatchNormLayer::create(lp);

    Mat input(input_shape.size(), input_shape.data(), CV_32F, 1.0);
    cv::randn(input, 0, 1);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);

    //create output_ref to compare with outputs
    Mat output_ref = input.clone();
    cv::sqrt(varMat + 1e-5, varMat);
    output_ref = (output_ref - meanMat) / varMat;

    ASSERT_EQ(1, outputs.size());
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);

}
INSTANTIATE_TEST_CASE_P(/*nothting*/, Layer_BatchNorm_Test,
                        testing::Values(
                            std::vector<int>({}),
                            std::vector<int>({4}),
                            std::vector<int>({1, 4}),
                            std::vector<int>({4, 1})
));


typedef testing::TestWithParam<tuple<std::vector<int>>> Layer_Const_Test;
TEST_P(Layer_Const_Test, Accuracy_01D)
{
    std::vector<int> input_shape = get<0>(GetParam());

    LayerParams lp;
    lp.type = "Const";
    lp.name = "ConstLayer";

    Mat constBlob = Mat(input_shape.size(), input_shape.data(), CV_32F);
    cv::randn(constBlob, 0.0, 1.0);
    Mat output_ref = constBlob.clone();

    lp.blobs.push_back(constBlob);
    Ptr<Layer> layer = ConstLayer::create(lp);

    std::vector<Mat> inputs; // No inputs are needed for a ConstLayer
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);
    ASSERT_EQ(1, outputs.size());
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Const_Test, testing::Values(
    std::vector<int>({}),
    std::vector<int>({1}),
    std::vector<int>({1, 4}),
    std::vector<int>({4, 1})
    ));

typedef testing::TestWithParam<std::vector<int>> Layer_Tile_Test;
TEST_P(Layer_Tile_Test, Accuracy_01D){

    std::vector<int> input_shape = GetParam();
    std::vector<int> repeats = {2, 2};

    LayerParams lp;
    lp.type = "Tile";
    lp.name = "TileLayer";
    lp.set("repeats", DictValue::arrayInt(repeats.data(), repeats.size()));
    Ptr<TileLayer> layer = TileLayer::create(lp);

    cv::Mat input = cv::Mat(input_shape.size(), input_shape.data(), CV_32F);
    cv::randn(input, 0, 1);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);

    // Manually create the expected output for verification
    cv::Mat output_ref = input.clone();
    for (int i = 0; i < repeats.size(); ++i) {
        cv::Mat tmp;
        cv::repeat(output_ref, (i == 0 ? repeats[i] : 1), (i == 1 ? repeats[i] : 1), tmp);
        output_ref = tmp;
    }

    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(shape(outputs[0]), shape(output_ref));
    normAssert(output_ref, outputs[0]);

}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Tile_Test,
/*input blob shape*/    testing::Values(
        std::vector<int>({}),
        std::vector<int>({2}),
        std::vector<int>({2, 1}),
        std::vector<int>({1, 2}),
        std::vector<int>({2, 2})
        ));

typedef testing::TestWithParam<tuple<std::vector<int>, std::vector<int>, std::string>> Layer_Einsum_Test;
TEST_P(Layer_Einsum_Test, Accuracy_01D)
{
    auto tup = GetParam();
    std::vector<int> input_shape1 = std::get<0>(tup);
    std::vector<int> input_shape2 = std::get<1>(tup);
    std::string equation = std::get<2>(tup);

    LayerParams lp;
    lp.type = "Einsum";
    lp.name = "EinsumLayer";
    lp.set("equation", equation);
    lp.set("inputSize", 2);
    lp.set("outputSize", 1);
    lp.set("inputShapes0", DictValue::arrayInt(&input_shape1[0], input_shape1.size()));
    lp.set("inputShapes1", DictValue::arrayInt(&input_shape2[0], input_shape2.size()));

    Ptr<Layer> layer = EinsumLayer::create(lp);

    cv::Mat input1(input_shape1.size(), input_shape1.data(), CV_32F);
    cv::Mat input2(input_shape2.size(), input_shape2.data(), CV_32F);
    cv::randn(input1, 0.0, 1.0); cv::randn(input2, 0.0, 1.0);

    std::vector<Mat> inputs = {input1, input2};
    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);
    ASSERT_EQ(1, outputs.size());

    // create output_ref to compare with outputs
    cv::Mat output_ref;
    int size[] = {1};
    if(equation == ",->" || equation == "i,->i" || equation == ",i->i" || equation == "ij,->ij"){
        output_ref = input1.mul(input2);
        if (equation == ",i->i")
            output_ref = output_ref.reshape(1, 1, size);
    } else if (equation == "i,i->i"){
        output_ref = input1.mul(input2);
    } else if (equation == "i,i->"){
        output_ref = input1.mul(input2);
        cv::Scalar sum = cv::sum(output_ref);
        output_ref = cv::Mat(0, nullptr, CV_32F, sum[0]);
    } else if (equation == "ij,ij->ij"){
        output_ref = input1.mul(input2);
    } else if (equation == "ij,ij->i"){
        output_ref = input1.mul(input2);
        if (input_shape1[0] == 1){
            cv::Scalar sum = cv::sum(output_ref);
            output_ref = cv::Mat(1, size, CV_32F, sum[0]);
        } else if (input_shape1[1] == 1){
            size[0] = input_shape1[0];
            output_ref = output_ref.reshape(1, 1, size);
        } else {
            cv::reduce(output_ref, output_ref, 1, cv::REDUCE_SUM, CV_32F);
            size[0] = input_shape1[0];
            output_ref = output_ref.reshape(1, 1, size);
        }
    } else {
        output_ref = cv::Mat();
    }

    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Einsum_Test, testing::Values(
    std::make_tuple(std::vector<int>({}), std::vector<int>({}), ",->"),
    std::make_tuple(std::vector<int>({1}), std::vector<int>({}), "i,->i"),
    std::make_tuple(std::vector<int>({}), std::vector<int>({1}), ",i->i"),
    std::make_tuple(std::vector<int>({4, 1}), std::vector<int>({}), "ij,->ij"),
    // std::make_tuple(std::vector<int>({}), std::vector<int>({4, 1}), ",ij->ij")), // mul function of arithm_op can not handle cases with different number of channels
    std::make_tuple(std::vector<int>({1}), std::vector<int>({1}), "i,i->i"),
    std::make_tuple(std::vector<int>({1}), std::vector<int>({1}), "i,i->"),
    std::make_tuple(std::vector<int>({4}), std::vector<int>({4}), "i,i->i"),
    std::make_tuple(std::vector<int>({4}), std::vector<int>({4}), "i,i->"),
    std::make_tuple(std::vector<int>({1, 4}), std::vector<int>({1, 4}), "ij,ij->ij"),
    std::make_tuple(std::vector<int>({4, 1}), std::vector<int>({4, 1}), "ij,ij->ij"),
    std::make_tuple(std::vector<int>({1, 4}), std::vector<int>({1, 4}), "ij,ij->i"),
    std::make_tuple(std::vector<int>({4, 1}), std::vector<int>({4, 1}), "ij,ij->i"),
    std::make_tuple(std::vector<int>({4, 4}), std::vector<int>({4, 4}), "ij,ij->i")
    ));


}}
