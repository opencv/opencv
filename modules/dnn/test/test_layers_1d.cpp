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
    ASSERT_EQ(outputs.size(), 1);
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
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
    normAssert(output_ref, outputs[0]);
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Concat_Test,
/*input blob shape*/    testing::Values(
    std::vector<int>({}),
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
    ASSERT_EQ(outputs.size(), 1);
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

typedef testing::TestWithParam<tuple<std::vector<int>, std::string>> Layer_Scatter_Test;
TEST_P(Layer_Scatter_Test, Accuracy1D) {

    std::vector<int> input_shape = get<0>(GetParam());
    std::string opr = get<1>(GetParam());

    LayerParams lp;
    lp.type = "Scatter";
    lp.name = "addLayer";
    lp.set("axis", 0);
    lp.set("reduction", opr);
    Ptr<ScatterLayer> layer = ScatterLayer::create(lp);

    float data[] = {1, 2, 3, 4};
    cv::Mat input = cv::Mat(input_shape.size(), input_shape.data(), CV_32F, data);
    // cv::randn(input, 0.0, 1.0);
    std::cout << "input: " << input << std::endl;

    int indices[] = {3, 2, 1, 0};
    cv::Mat indices_mat(input_shape.size(), input_shape.data(), CV_32S, indices);
    cv::Mat output(input_shape.size(), input_shape.data(), CV_32F, 0.0);

    // create reference output
    cv::Mat output_ref(input_shape, CV_32F, 0.0);
    for (int i = 0; i < input_shape[0]; i++){
        output_ref.at<float>(indices[i]) = input.at<float>(i);
    }
    std::cout << "output_ref: " << output_ref << std::endl;

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

    std::cout << "input: " << input << std::endl;
    std::cout << "indices: " << indices_mat << std::endl;
    std::cout << "output[0]: " << outputs[0] << std::endl;
    std::cout << "output_ref: " << output_ref << std::endl;

    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(shape(output_ref), shape(outputs[0]));
}
INSTANTIATE_TEST_CASE_P(/*nothing*/, Layer_Scatter_Test, Combine(
/*input blob shape*/    testing::Values(std::vector<int>{4}),
/*reduce*/              Values("none", "add", "mul", "max", "min")
// /*operation*/           testing::Values(0, 1),
));



}}
