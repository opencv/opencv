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

class Layer_1d_Test: public testing::TestWithParam<tuple<int>>
{
public:
    int batch_size;
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    float inp_value;
    cv::Mat input;

    void SetUp()
    {
        batch_size = get<0>(GetParam());
        input_shape = {batch_size, 3};
        output_shape = {batch_size, 3};
        if (batch_size == 0)
        {
            input_shape.erase(input_shape.begin());
            output_shape.erase(output_shape.begin());
        }

        // generate random positeve value from 1 to 10
        inp_value = 1 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(10-1)));
        input = cv::Mat(input_shape, CV_32F, inp_value);
    }
};

TEST_P(Layer_1d_Test, Scale)
{
    LayerParams lp;
    lp.type = "Scale";
    lp.name = "scaleLayer";
    lp.set("axis", 0);
    lp.set("mode", "scale");
    lp.set("bias_term", false);
    Ptr<ScaleLayer> layer = ScaleLayer::create(lp);

    input = cv::Mat(input_shape, CV_32F);
    cv::randn(input, 0.0, 1.0);
    cv::Mat weight = cv::Mat(output_shape, CV_32F, 2.0);

    std::vector<Mat> inputs{input, weight};
    std::vector<Mat> outputs;

    cv::Mat output_ref = input.mul(weight);
    runLayer(layer, inputs, outputs);

    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, ReLU6)
{
    LayerParams lp;

    lp.type = "ReLU6";
    lp.name = "ReLU6Layer";
    lp.set("min_value", 0.0);
    lp.set("max_value", 1.0);
    Ptr<ReLU6Layer> layer = ReLU6Layer::create(lp);

    cv::Mat output_ref(output_shape, CV_32F, 1.0);
    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Clip)
{
    LayerParams lp;

    lp.type = "Clip";
    lp.name = "clipLayer";
    lp.set("min_value", 0.0);
    lp.set("max_value", 1.0);
    Ptr<ReLU6Layer> layer = ReLU6Layer::create(lp);

    cv::Mat output_ref(output_shape, CV_32F, 1.0);
    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, ReLU)
{
    LayerParams lp;

    lp.type = "ReLU";
    lp.name = "reluLayer";
    lp.set("negative_slope", 0.0);
    Ptr<ReLULayer> layer = ReLULayer::create(lp);

    cv::Mat output_ref(output_shape, CV_32F, inp_value);
    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Gelu)
{
    LayerParams lp;

    lp.type = "Gelu";
    lp.name = "geluLayer";
    Ptr<GeluLayer> layer = GeluLayer::create(lp);

    float geluValue = inp_value * 0.5 * (std::erf(inp_value * 1 / std::sqrt(2.0)) + 1.0);

    float data[3] = {geluValue, geluValue, geluValue};
    cv::Mat output_ref(output_shape, CV_32F, data);
    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, GeluApprox)
{
    LayerParams lp;

    lp.type = "GeluApprox";
    lp.name = "geluApproxLayer";
    Ptr<GeluApproximationLayer> layer = GeluApproximationLayer::create(lp);

    float geluValue = inp_value * 0.5 * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (inp_value + 0.044715 * std::pow(inp_value, 3))));
    float data[3] = {geluValue, geluValue, geluValue};
    cv::Mat output_ref(output_shape, CV_32F, data);
    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Sigmoid)
{
    LayerParams lp;

    lp.type = "Sigmoid";
    lp.name = "sigmoidLayer";
    Ptr<SigmoidLayer> layer = SigmoidLayer::create(lp);

    float sigmoidValue = 1.0 / (1.0 + std::exp(-inp_value));
    float data[3] = {sigmoidValue, sigmoidValue, sigmoidValue};
    cv::Mat output_ref(output_shape, CV_32F, data);
    std::vector<Mat> inputs{input};

    std::vector<Mat> outputs;
    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Tanh)
{
    LayerParams lp;

    lp.type = "TanH";
    lp.name = "TanHLayer";
    Ptr<Layer> layer = TanHLayer::create(lp);

    std::vector<int> input_shape = {1, 3};
    std::vector<int> output_shape = {1, 3};
    if (batch_size == 0) {
        input_shape.erase(input_shape.begin());
        output_shape.erase(output_shape.begin());
    }

    float value = std::tanh(inp_value);
    float data[3] = {value, value, value};
    Mat output_ref = Mat(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Swish)
{
    LayerParams lp;

    lp.type = "Swish";
    lp.name = "SwishLayer";
    Ptr<Layer> layer = SwishLayer::create(lp);

    float value = inp_value / (1 + std::exp(-inp_value));
    float data[3] = {value, value, value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Mish)
{
    LayerParams lp;

    lp.type = "Mish";
    lp.name = "MishLayer";
    Ptr<Layer> layer = MishLayer::create(lp);

    float value = inp_value * std::tanh(std::log(1 + std::exp(inp_value)));
    float data[3] = {value, value, value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, ELU)
{
    LayerParams lp;

    lp.type = "ELU";
    lp.name = "eluLayer";
    lp.set("alpha", 1.0);
    Ptr<Layer> layer = ELULayer::create(lp);
    float value = inp_value > 0 ? inp_value : std::exp(inp_value) - 1;
    float data[3] = {value, value, value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Abs)
{
    LayerParams lp;

    lp.type = "Abs";
    lp.name = "absLayer";
    Ptr<Layer> layer = AbsLayer::create(lp);

    float value = std::abs(inp_value);
    float data[3] = {value, value, value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, BNLL)
{
    LayerParams lp;

    lp.type = "BNLL";
    lp.name = "bnllLayer";
    Ptr<Layer> layer = BNLLLayer::create(lp);

    float value = std::log(1 + std::exp(inp_value));
    float data[3] = {value, value, value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Ceil)
{
    LayerParams lp;

    lp.type = "Ceil";
    lp.name = "ceilLayer";
    Ptr<Layer> layer = CeilLayer::create(lp);

    float value = std::ceil(inp_value);
    float data[3] = {value, value, value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Floor)
{
    LayerParams lp;

    lp.type = "Floor";
    lp.name = "floorLayer";
    Ptr<Layer> layer = FloorLayer::create(lp);

    float value = std::floor(inp_value);
    float data[3] = {value, value, value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Log)
{
    LayerParams lp;

    lp.type = "Log";
    lp.name = "logLayer";
    Ptr<Layer> layer = LogLayer::create(lp);

    float value = std::log(inp_value);
    float data[3] = {value, value, value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Round)
{
    LayerParams lp;

    lp.type = "Round";
    lp.name = "roundLayer";
    Ptr<Layer> layer = RoundLayer::create(lp);

    float value = std::round(inp_value);
    float data[3] = {value, value, value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Sqrt)
{
    LayerParams lp;

    lp.type = "Sqrt";
    lp.name = "sqrtLayer";
    Ptr<Layer> layer = SqrtLayer::create(lp);

    float value = std::sqrt(inp_value);
    float data[3] = {value, value, value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Acos)
{
    LayerParams lp;

    lp.type = "Acos";
    lp.name = "acosLayer";
    Ptr<Layer> layer = AcosLayer::create(lp);

    inp_value = 0.5 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(1-0.5)));
    input = Mat(input_shape, CV_32F, inp_value);

    float value = std::acos(inp_value);
    float data[3] = {value, value, value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Acosh)
{
    LayerParams lp;

    lp.type = "Acosh";
    lp.name = "acoshLayer";
    Ptr<Layer> layer = AcoshLayer::create(lp);

    inp_value = 1.5 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(10-1)));
    input = Mat(input_shape, CV_32F, inp_value);

    float value = std::acosh(inp_value);
    float data[3] = {value, value, value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Asin)
{
    LayerParams lp;

    lp.type = "Asin";
    lp.name = "asinLayer";
    Ptr<Layer> layer = AsinLayer::create(lp);

    inp_value = 0.5 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(1-0.5)));
    input = Mat(input_shape, CV_32F, inp_value);

    float value = std::asin(inp_value);
    float data[3] = {value, value, value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Asinh)
{
    LayerParams lp;

    lp.type = "Asinh";
    lp.name = "asinhLayer";
    Ptr<Layer> layer = AsinhLayer::create(lp);

    inp_value = 0.5 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(1-0.5)));
    input = Mat(input_shape, CV_32F, inp_value);

    float value = std::asinh(inp_value);
    float data[3] = {value, value, value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Atan)
{
    LayerParams lp;

    lp.type = "Atan";
    lp.name = "atanLayer";
    Ptr<Layer> layer = AtanLayer::create(lp);

    inp_value = 0.5 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(1-0.5)));
    input = Mat(input_shape, CV_32F, inp_value);

    float value = std::atan(inp_value);
    float data[3] = {value, value, value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Cos)
{
    LayerParams lp;

    lp.type = "Cos";
    lp.name = "cosLayer";
    Ptr<Layer> layer = CosLayer::create(lp);

    inp_value = M_PI / 4 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(M_PI/2-M_PI/4)));
    input = Mat(input_shape, CV_32F, inp_value);

    float value = std::cos(inp_value);
    float data[3] = {value, value, value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Cosh)
{
    LayerParams lp;

    lp.type = "Cosh";
    lp.name = "coshLayer";
    Ptr<Layer> layer = CoshLayer::create(lp);

    inp_value = M_PI / 4 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(M_PI/2-M_PI/4)));
    input = Mat(input_shape, CV_32F, inp_value);

    float value = std::cosh(inp_value);
    float data[3] = {value, value, value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Sin)
{
    LayerParams lp;

    lp.type = "Sin";
    lp.name = "sinLayer";
    Ptr<Layer> layer = SinLayer::create(lp);

    inp_value = M_PI / 4 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(M_PI/2-M_PI/4)));
    input = Mat(input_shape, CV_32F, inp_value);

    float value = std::sin(inp_value);
    float data[3] = {value, value, value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Sinh)
{
    LayerParams lp;

    lp.type = "Sinh";
    lp.name = "sinhLayer";
    Ptr<Layer> layer = SinhLayer::create(lp);

    inp_value = M_PI / 4 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(M_PI/2-M_PI/4)));
    input = Mat(input_shape, CV_32F, inp_value);

    float value = std::sinh(inp_value);
    float data[3] = {value, value, value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Tan)
{
    LayerParams lp;

    lp.type = "Tan";
    lp.name = "tanLayer";
    Ptr<Layer> layer = TanLayer::create(lp);

    inp_value = M_PI / 4 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(M_PI/2-M_PI/4)));
    input = Mat(input_shape, CV_32F, inp_value);

    float value = std::tan(inp_value);
    float data[3] = {value, value, value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Erf)
{
    LayerParams lp;

    lp.type = "Erf";
    lp.name = "erfLayer";
    Ptr<Layer> layer = ErfLayer::create(lp);

    float value = 1 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(10-1)));
    input = Mat(input_shape, CV_32F, value);

    float data[3] = {std::erf(value), std::erf(value), std::erf(value)};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Reciprocal)
{
    LayerParams lp;

    lp.type = "Reciprocal";
    lp.name = "reciprocalLayer";
    Ptr<Layer> layer = ReciprocalLayer::create(lp);

    float value = 1 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(10-1)));
    input = Mat(input_shape, CV_32F, value);

    float data[3] = {1/value, 1/value, 1/value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, HardSwish)
{
    LayerParams lp;

    lp.type = "HardSwish";
    lp.name = "hardSwishLayer";
    Ptr<Layer> layer = HardSwishLayer::create(lp);

    float value = 1 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(10-1)));
    input = Mat(input_shape, CV_32F, value);

    float data[3] = {value * std::max(0.0f, std::min(6.0f, value + 3.0f)) / 6.0f, value * std::max(0.0f, std::min(6.0f, value + 3.0f)) / 6.0f, value * std::max(0.0f, std::min(6.0f, value + 3.0f)) / 6.0f};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Softplus)
{
    LayerParams lp;

    lp.type = "Softplus";
    lp.name = "softplusLayer";
    Ptr<Layer> layer = SoftplusLayer::create(lp);

    float value = 1 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(10-1)));
    input = Mat(input_shape, CV_32F, value);

    float data[3] = {std::log(1 + std::exp(value)), std::log(1 + std::exp(value)), std::log(1 + std::exp(value))};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, SoftSign)
{
    LayerParams lp;

    lp.type = "Softsign";
    lp.name = "softsignLayer";
    Ptr<Layer> layer = SoftsignLayer::create(lp);

    float value = 1 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(10-1)));
    input = Mat(input_shape, CV_32F, value);

    float data[3] = {value / (1 + std::abs(value)), value / (1 + std::abs(value)), value / (1 + std::abs(value))};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, CELU)
{
    LayerParams lp;

    lp.type = "CELU";
    lp.name = "celuLayer";
    lp.set("alpha", 1.0);
    Ptr<Layer> layer = CeluLayer::create(lp);

    float value = 1 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(10-1)));
    input = Mat(input_shape, CV_32F, value);

    float data[3] = {value < 0 ? std::exp(value) - 1 : value, value < 0 ? std::exp(value) - 1 : value, value < 0 ? std::exp(value) - 1 : value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, HardSigmoid)
{
    LayerParams lp;

    lp.type = "HardSigmoid";
    lp.name = "hardSigmoidLayer";
    Ptr<Layer> layer = HardSigmoidLayer::create(lp);

    float value = 1 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(10-1)));
    input = Mat(input_shape, CV_32F, value);

    float data[3] = {std::max(0.0f, std::min(1.0f, 0.2f * value + 0.5f)), std::max(0.0f, std::min(1.0f, 0.2f * value + 0.5f)), std::max(0.0f, std::min(1.0f, 0.2f * value + 0.5f))};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, SELU)
{
    LayerParams lp;

    lp.type = "SELU";
    lp.name = "seluLayer";
    lp.set("alpha", 1.6732631921768188);
    lp.set("gamma", 1.0507009873554805);
    Ptr<Layer> layer = SeluLayer::create(lp);


    input = Mat(input_shape, CV_32F, inp_value);

    // float data[3] = {value < 0 ? 1.0507 * (std::exp(value) - 1) : 1.0507 * value, value < 0 ? 1.0507 * (std::exp(value) - 1) : 1.0507 * value, value < 0 ? 1.0507 * (std::exp(value) - 1) : 1.0507 * value};
    float value = 1.0507009873554805 * (inp_value > 0 ? inp_value : 1.6732631921768188 * (std::exp(inp_value / 1.0) - 1));
    float data[] = {value, value, value};

    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, ThresholdedReLU)
{
    LayerParams lp;

    lp.type = "ThresholdedReLU";
    lp.name = "thresholdedReluLayer";
    lp.set("alpha", 1.0);
    Ptr<Layer> layer = ThresholdedReluLayer::create(lp);

    float value = inp_value > 1.0 ? inp_value : 0.0;
    float data[] = {value, value, value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Power)
{
    LayerParams lp;

    lp.type = "Power";
    lp.name = "powerLayer";
    lp.set("power", 2.0);
    lp.set("scale", 1.0);
    lp.set("shift", 0.0);
    Ptr<Layer> layer = PowerLayer::create(lp);

    float value = std::pow(inp_value, 2.0);
    float data[] = {value, value, value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Exp)
{
    LayerParams lp;

    lp.type = "Exp";
    lp.name = "expLayer";
    Ptr<Layer> layer = ExpLayer::create(lp);

    float value = 1 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(10-1)));
    input = Mat(input_shape, CV_32F, value);

    float data[3] = {std::exp(value), std::exp(value), std::exp(value)};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Sign)
{
    LayerParams lp;

    lp.type = "Sign";
    lp.name = "signLayer";
    Ptr<Layer> layer = SignLayer::create(lp);


    input = Mat(input_shape, CV_32F, inp_value);

    float value = inp_value > 0 ? 1.0 : 0.0;
    float data[] = {value, value, value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, Shrink)
{
    LayerParams lp;

    lp.type = "Shrink";
    lp.name = "shrinkLayer";
    lp.set("lambda", 0.5);
    lp.set("bias", 0.5);
    Ptr<Layer> layer = ShrinkLayer::create(lp);


    input = Mat(input_shape, CV_32F, inp_value);

    float value = inp_value > 0.5 ? inp_value - 0.5 : 0.0;
    float data[] = {value, value, value};
    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

TEST_P(Layer_1d_Test, ChannelsPReLU)
{
    LayerParams lp;

    lp.type = "ChannelsPReLU";
    lp.name = "channelsPReLULayer";
    Mat alpha = Mat(1, 3, CV_32F, 0.5);
    lp.blobs.push_back(alpha);
    Ptr<Layer> layer = ChannelsPReLULayer::create(lp);

    float value = inp_value > 0 ? inp_value : 0.5 * inp_value;
    float data[] = {value, value, value};

    Mat output_ref(output_shape, CV_32F, data);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

INSTANTIATE_TEST_CASE_P(, Layer_1d_Test, Values(0, 1));

typedef testing::TestWithParam<tuple<int, int>> Layer_Gather_1d_Test;
TEST_P(Layer_Gather_1d_Test, Accuracy) {

    int batch_size = get<0>(GetParam());
    int axis = get<1>(GetParam());

    LayerParams lp;
    lp.type = "Gather";
    lp.name = "gatherLayer";
    lp.set("axis", axis);
    lp.set("real_ndims", 1);

    Ptr<GatherLayer> layer = GatherLayer::create(lp);

    std::vector<int> input_shape = {batch_size, 1};
    std::vector<int> indices_shape = {1, 1};
    std::vector<int> output_shape = {batch_size, 1};

    if (batch_size == 0){
        input_shape.erase(input_shape.begin());
        indices_shape.erase(indices_shape.begin());
        output_shape.erase(output_shape.begin());
    } else if (axis == 0) {
        output_shape[0] = 1;
    }

    cv::Mat input = cv::Mat(input_shape, CV_32F, 1.0);
    cv::randu(input, 0.0, 1.0);
    cv::Mat indices = cv::Mat(indices_shape, CV_32F, 0.0);
    cv::Mat output_ref(output_shape, CV_32F, input(cv::Range::all(), cv::Range(0, 1)).data);

    std::vector<Mat> inputs{input, indices};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Gather_1d_Test, Combine(
/*input blob shape*/    Values(0, 1, 2, 3),
/*operation*/           Values(0, 1)
));

typedef testing::TestWithParam<tuple<int, int, std::string>> Layer_Arg_1d_Test;
TEST_P(Layer_Arg_1d_Test, Accuracy) {

    int batch_size = get<0>(GetParam());
    int axis = get<1>(GetParam());
    std::string operation = get<2>(GetParam());

    LayerParams lp;
    lp.type = "Arg";
    lp.name = "arg" + operation + "_Layer";
    lp.set("op", operation);
    lp.set("axis", axis);
    lp.set("keepdims", 1);
    lp.set("select_last_index", 0);

    Ptr<ArgLayer> layer = ArgLayer::create(lp);

    std::vector<int> input_shape = {batch_size, 1};
    std::vector<int> output_shape = {1, 1};

    if (batch_size == 0){
        input_shape.erase(input_shape.begin());
        output_shape.erase(output_shape.begin());
    }

    if (axis != 0 && batch_size != 0){
        output_shape[0] = batch_size;
    }

    cv::Mat input = cv::Mat(input_shape, CV_32F, 1);
    cv::Mat output_ref(output_shape,  CV_32F, 0);

    for (int i = 0; i < batch_size; ++i)
        input.at<float>(i, 0) = static_cast<float>(i + 1);

    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Arg_1d_Test, Combine(
/*input blob shape*/    Values(0, 1, 2, 3),
/*operation*/           Values(0, 1),
/*operation*/           Values( "max", "min")
));

typedef testing::TestWithParam<tuple<int, std::string>> Layer_NaryElemwise_1d_Test;
TEST_P(Layer_NaryElemwise_1d_Test, Accuracy) {

    int batch_size = get<0>(GetParam());
    std::string operation = get<1>(GetParam());

    LayerParams lp;
    lp.type = "Eltwise";
    lp.name = operation + "_Layer";
    lp.set("operation", operation);
    Ptr<NaryEltwiseLayer> layer = NaryEltwiseLayer::create(lp);

    std::vector<int> input_shape = {batch_size, 1};
    if (batch_size == 0)
        input_shape.erase(input_shape.begin());

    cv::Mat input1 = cv::Mat(input_shape, CV_32F, 0.0);
    cv::Mat input2 = cv::Mat(input_shape, CV_32F, 0.0);
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
        ASSERT_EQ(shape(output_ref), shape(outputs[0]));
        normAssert(output_ref, outputs[0]);
    } else {
        CV_Error(Error::StsAssert, "Provided operation: " + operation + " is not supported. Please check the test instantiation.");
    }
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_NaryElemwise_1d_Test, Combine(
/*input blob shape*/    Values(0, 1),
/*operation*/           Values("div", "mul", "sum", "sub")
));

typedef testing::TestWithParam<tuple<int, std::string>> Layer_Elemwise_1d_Test;
TEST_P(Layer_Elemwise_1d_Test, Accuracy) {

    int batch_size = get<0>(GetParam());
    std::string operation = get<1>(GetParam());

    LayerParams lp;
    lp.type = "Eltwise";
    lp.name = operation + "_Layer";
    lp.set("operation", operation);
    Ptr<EltwiseLayer> layer = EltwiseLayer::create(lp);

    std::vector<int> input_shape = {batch_size, 1};
    if (batch_size == 0)
        input_shape.erase(input_shape.begin());

    cv::Mat input1 = cv::Mat(input_shape, CV_32F, 1.0);
    cv::Mat input2 = cv::Mat(input_shape, CV_32F, 1.0);
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
        ASSERT_EQ(shape(output_ref), shape(outputs[0]));
        normAssert(output_ref, outputs[0]);
    } else {
        CV_Error(Error::StsAssert, "Provided operation: " + operation + " is not supported. Please check the test instantiation.");
    }
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Elemwise_1d_Test, Combine(
/*input blob shape*/    Values(0, 1, 2, 3),
/*operation*/           Values("div", "prod", "max", "min", "sum")
));

}}
