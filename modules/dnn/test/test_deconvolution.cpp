// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/dnn/all_layers.hpp>

namespace opencv_test { namespace {

typedef TestWithParam<tuple<int, int, bool, int> > DeconvolutionCoordinates;

TEST_P(DeconvolutionCoordinates, Accuracy)
{
    const int dims = get<0>(GetParam());
    const int mode = get<1>(GetParam());
    const bool dynamicWeights = get<2>(GetParam());
    const int previousThreads = cv::getNumThreads();
    cv::setNumThreads(get<3>(GetParam()));
    struct RestoreThreads
    {
        int count;
        ~RestoreThreads() { cv::setNumThreads(count); }
    } restoreThreads = {previousThreads};

    const int inChannels = 4, outChannels = 6, groups = 2;
    const bool hasBias = mode % 2 == 0;
    std::vector<int> kernel(dims), stride(dims), dilation(dims, 1);
    std::vector<int> pads(2 * dims, 0), adjust(dims, 0);
    int kernelSize = 1;
    for (int d = 0; d < dims; ++d)
    {
        kernel[d] = mode == 0 ? 1 : 2 + d % 2;
        stride[d] = mode == 1 ? kernel[d] : mode == 3 ? 2 : 1;
        dilation[d] = mode == 3 && d == 0 ? 2 : 1;
        pads[d] = mode >= 2 ? 1 : 0;
        pads[d + dims] = mode >= 2 ? d % 2 : 0;
        adjust[d] = mode == 3 ? 1 : 0;
        kernelSize *= kernel[d];
    }
    LayerParams lp;
    lp.set("kernel_size", DictValue::arrayInt(kernel.data(), dims));
    lp.set("stride", DictValue::arrayInt(stride.data(), dims));
    lp.set("dilation", DictValue::arrayInt(dilation.data(), dims));
    lp.set("pad", DictValue::arrayInt(pads.data(), 2 * dims));
    lp.set("adj", DictValue::arrayInt(adjust.data(), dims));
    lp.set("num_output", outChannels);
    lp.set("group", groups);
    lp.set("bias_term", hasBias);

    std::vector<int> weightShape = {inChannels, outChannels / groups};
    weightShape.insert(weightShape.end(), kernel.begin(), kernel.end());
    Mat weights(weightShape, CV_32F), bias(1, outChannels, CV_32F);
    for (size_t i = 0; i < weights.total(); ++i)
        weights.ptr<float>()[i] = (int(i * 7 % 11) - 5) * 0.125f;
    for (int i = 0; i < outChannels; ++i)
        bias.ptr<float>()[i] = hasBias ? (i - 3) * 0.25f : 0.0f;
    if (!dynamicWeights)
    {
        lp.blobs.push_back(weights);
        if (hasBias)
            lp.blobs.push_back(bias);
    }
    Ptr<Layer> layer = DeconvolutionLayer::create(lp);

    for (int run = 0; run < 2; ++run)
    {
        std::vector<int> inputShape = {2 - run, inChannels};
        std::vector<int> outputShape = {2 - run, outChannels};
        int inputSize = 1, outputSize = 1;
        for (int d = 0; d < dims; ++d)
        {
            const int size = mode == 0 && run ? 1 : 3 + 2 * d + run;
            const int outSize = (size - 1) * stride[d] + dilation[d] * (kernel[d] - 1)
                                + 1 - pads[d] - pads[d + dims] + adjust[d];
            inputShape.push_back(size);
            outputShape.push_back(outSize);
            inputSize *= size;
            outputSize *= outSize;
        }
        Mat input(inputShape, CV_32F), expected(outputShape, CV_32F, Scalar::all(0));
        for (size_t i = 0; i < input.total(); ++i)
            input.ptr<float>()[i] = (int((i * 13 + run) % 17) - 8) * 0.125f;

        // Scatter input values to the output. All values are exact binary fractions.
        for (int n = 0; n < inputShape[0]; ++n)
            for (int c = 0; c < inChannels; ++c)
                for (int pos = 0; pos < inputSize; ++pos)
                    for (int k = 0; k < kernelSize; ++k)
                    {
                        int inputPos = pos, kernelPos = k, outputPos = 0, pitch = 1;
                        bool valid = true;
                        for (int d = dims - 1; d >= 0; --d)
                        {
                            const int coordinate = (inputPos % inputShape[d + 2]) * stride[d]
                                - pads[d] + (kernelPos % kernel[d]) * dilation[d];
                            valid = valid && coordinate >= 0 && coordinate < outputShape[d + 2];
                            outputPos += coordinate * pitch;
                            pitch *= outputShape[d + 2];
                            inputPos /= inputShape[d + 2];
                            kernelPos /= kernel[d];
                        }
                        if (!valid)
                            continue;
                        for (int q = 0; q < outChannels / groups; ++q)
                        {
                            const int outputChannel = c / (inChannels / groups) * (outChannels / groups) + q;
                            expected.ptr<float>()[(n * outChannels + outputChannel) * outputSize + outputPos] +=
                                input.ptr<float>()[(n * inChannels + c) * inputSize + pos] *
                                weights.ptr<float>()[(c * (outChannels / groups) + q) * kernelSize + k];
                        }
                    }
        for (int n = 0; n < inputShape[0]; ++n)
            for (int c = 0; c < outChannels; ++c)
                for (int pos = 0; pos < outputSize; ++pos)
                    expected.ptr<float>()[(n * outChannels + c) * outputSize + pos] += bias.ptr<float>()[c];

        std::vector<Mat> inputs(1, input), outputs;
        if (dynamicWeights)
        {
            inputs.push_back(weights);
            if (hasBias)
                inputs.push_back(bias);
        }
        runLayer(layer, inputs, outputs);
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs[0].shape(), expected.shape());
        EXPECT_TRUE(cv::checkRange(outputs[0]));
        EXPECT_EQ(cv::norm(outputs[0], expected, NORM_INF), 0.0);
    }
}

INSTANTIATE_TEST_CASE_P(Layer_Test, DeconvolutionCoordinates, testing::Combine(
    testing::Values(2, 3), testing::Values(0, 1, 2, 3), testing::Bool(), testing::Values(1, 4)));

}} // namespace
