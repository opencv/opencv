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

typedef testing::TestWithParam<tuple<int>> Layer_1d_Test;
TEST_P(Layer_1d_Test, Scale)
{
    int batch_size = get<0>(GetParam());

    LayerParams lp;
    lp.type = "Scale";
    lp.name = "scaleLayer";
    lp.set("axis", 0);
    lp.set("mode", "scale");
    lp.set("bias_term", false);
    Ptr<ScaleLayer> layer = ScaleLayer::create(lp);

    std::vector<int> input_shape = {batch_size, 3};
    std::vector<int> output_shape = {batch_size, 3};

    if (batch_size == 0){
        input_shape.erase(input_shape.begin());
        output_shape.erase(output_shape.begin());
    }

    cv::Mat input = cv::Mat(input_shape, CV_32F, 1.0);
    cv::randn(input, 0.0, 1.0);
    cv::Mat weight = cv::Mat(output_shape, CV_32F, 2.0);

    std::vector<Mat> inputs{input, weight};
    std::vector<Mat> outputs;

    cv::Mat output_ref = input.mul(weight);
    runLayer(layer, inputs, outputs);

    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}

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
    cv::Mat indices = cv::Mat(indices_shape, CV_32S, 0.0);
    cv::Mat output_ref = cv::Mat(output_shape, CV_32F, input(cv::Range::all(), cv::Range(0, 1)).data);

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
    cv::Mat output_ref = cv::Mat(output_shape,  CV_32F, 0);

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

typedef testing::TestWithParam<tuple<std::tuple<int, std::vector<int>>, std::string>> Layer_Reduce_Test;
TEST_P(Layer_Reduce_Test, Accuracy)
{
    auto reduceOperation = [](const cv::Mat& input, const std::string& operation) -> float {
        float result = operation == "max" ? std::numeric_limits<float>::lowest() : 0;
        float sum = 0;
        for (int i = 0; i < input.total(); ++i) {
            float value = input.at<float>(i);
            if (operation == "max") {
                result = std::max(result, value);
            } else if (operation == "min") {
                if (i == 0 || result > value) result = value;
            } else if (operation == "mean") {
                sum += value;
            } else if (operation == "sum") {
                result += value;
            } else if (operation == "sum_square") {
                result += value * value;
            } else if (operation == "l1") {
                result += std::abs(value);
            } else if (operation == "l2") {
                result += value * value;
            } else if (operation == "prod") {
                result = (i == 0) ? value : result * value;
            }
        }
        if (operation == "mean") {
            result = sum / input.total();
        } else if (operation == "l2") {
            result = std::sqrt(result);
        } else if (operation == "log_sum" || operation == "log_sum_exp") {
            for (int i = 0; i < input.total(); ++i) {
                result += (operation == "log_sum") ? std::log(input.at<float>(i)) : std::exp(input.at<float>(i));
            }
            if (operation == "log_sum_exp") {
                result = std::log(result);
            }
        }
        return result;
    };

    auto tup = get<0>(GetParam());
    int dims = get<0>(tup);
    std::vector<int> input_shape = get<1>(tup);
    std::string reduce_operation = get<1>(GetParam());

    if (dims == 2 && reduce_operation == "log_sum") // both output and reference are nans
        return;

    LayerParams lp;
    lp.type = "Reduce";
    lp.name = "reduceLayer";
    lp.set("reduce", reduce_operation);
    lp.set("axis", 0);
    lp.set("keepdims", true);
    Ptr<ReduceLayer> layer = ReduceLayer::create(lp);

    cv::Mat input(dims, input_shape.data(), CV_32F, 1.0);
    cv::randn(input, 0.0, 1.0);

    float out_value = reduceOperation(input, reduce_operation);

    cv::Mat output_ref = cv::Mat(dims, (dims <= 1) ? input_shape.data() : std::vector<int>({1, 1}).data(), CV_32F, out_value);
    std::vector<Mat> inputs{input};
    std::vector<Mat> outputs;

    runLayer(layer, inputs, outputs);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Reduce_Test, Combine(
/*input blob shape*/    Values(
    std::make_tuple(0, std::vector<int>{0}),
    std::make_tuple(1, std::vector<int>{1}),
    std::make_tuple(2, std::vector<int>{1, 4})
    ),
/*input blob shape*/    Values("max", "min", "mean", "sum", "sum_square", "l1", "l2", "prod", "log_sum", "log_sum_exp"))
);


}}
