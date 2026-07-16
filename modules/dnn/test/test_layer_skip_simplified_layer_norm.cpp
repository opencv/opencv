// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "npy_blob.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <cmath>

namespace opencv_test { namespace {

static bool skipIfClassicEngineForced()
{
    auto engine_forced = static_cast<cv::dnn::EngineType>(
        cv::utils::getConfigurationParameterSizeT("OPENCV_FORCE_DNN_ENGINE", cv::dnn::ENGINE_AUTO));
    if (engine_forced == cv::dnn::ENGINE_CLASSIC)
    {
        applyTestTag(CV_TEST_TAG_DNN_SKIP_PARSER);
        return true;
    }
    return false;
}

TEST(SkipSimplifiedLayerNormalizationLayer, ONNXModel_NoBiasMultiRow)
{
    if (skipIfClassicEngineForced()) return;

    const std::string basename = "skip_simplified_layer_norm";
    Net net = readNetFromONNX(findDataFile("dnn/onnx/models/" + basename + ".onnx", true), cv::dnn::ENGINE_NEW);
    ASSERT_FALSE(net.empty());

    net.setInput(blobFromNPY(findDataFile("dnn/onnx/data/input_" + basename + "_0.npy")), "input");
    net.setInput(blobFromNPY(findDataFile("dnn/onnx/data/input_" + basename + "_1.npy")), "skip");
    net.setInput(blobFromNPY(findDataFile("dnn/onnx/data/input_" + basename + "_2.npy")), "gamma");

    std::vector<Mat> outs;
    net.forward(outs, std::vector<String>{"output", "input_skip_bias_sum"});
    ASSERT_EQ(outs.size(), (size_t)2);

    Mat refOutput = blobFromNPY(findDataFile("dnn/onnx/data/output_" + basename + "_0.npy"));
    Mat refSum = blobFromNPY(findDataFile("dnn/onnx/data/output_" + basename + "_1.npy"));
    normAssert(refOutput, outs[0], "output", 1e-4, 1e-3);
    normAssert(refSum, outs[1], "input_skip_bias_sum", 1e-4, 1e-3);
}

TEST(SkipSimplifiedLayerNormalizationLayer, ONNXModel_BiasNonUniformGammaFourOutputs)
{
    if (skipIfClassicEngineForced()) return;

    const float epsilon = 1e-5f;
    const std::string basename = "skip_simplified_layer_norm_with_bias";
    Net net = readNetFromONNX(findDataFile("dnn/onnx/models/" + basename + ".onnx", true), cv::dnn::ENGINE_NEW);
    ASSERT_FALSE(net.empty());

    net.setInput(blobFromNPY(findDataFile("dnn/onnx/data/input_" + basename + "_0.npy")), "input");
    net.setInput(blobFromNPY(findDataFile("dnn/onnx/data/input_" + basename + "_1.npy")), "skip");
    net.setInput(blobFromNPY(findDataFile("dnn/onnx/data/input_" + basename + "_2.npy")), "gamma");
    net.setInput(blobFromNPY(findDataFile("dnn/onnx/data/input_" + basename + "_3.npy")), "bias");

    std::vector<Mat> outs;
    net.forward(outs, std::vector<String>{"output", "mean", "inv_std_var", "input_skip_bias_sum"});
    ASSERT_EQ(outs.size(), (size_t)4);

    Mat refOutput = blobFromNPY(findDataFile("dnn/onnx/data/output_" + basename + "_0.npy"));
    Mat refSum = blobFromNPY(findDataFile("dnn/onnx/data/output_" + basename + "_3.npy"));
    normAssert(refOutput, outs[0], "output", 1e-4, 1e-3);
    normAssert(refSum, outs[3], "input_skip_bias_sum", 1e-4, 1e-3);

    const int hidden = refSum.size[refSum.dims - 1];
    const int rows = (int)(refSum.total() / hidden);
    std::vector<int> reducedShape(outs[1].size.p, outs[1].size.p + outs[1].dims);
    Mat expectedMean = Mat::zeros((int)reducedShape.size(), reducedShape.data(), CV_32F);
    Mat expectedInvStd((int)reducedShape.size(), reducedShape.data(), CV_32F);
    const float* s = refSum.ptr<float>();
    float* iv = expectedInvStd.ptr<float>();
    for (int r = 0; r < rows; ++r)
    {
        double sumSq = 0.0;
        for (int c = 0; c < hidden; ++c)
        {
            float v = s[(size_t)r * hidden + c];
            sumSq += (double)v * v;
        }
        iv[r] = (float)(1.0 / std::sqrt(sumSq / hidden + epsilon));
    }
    normAssert(expectedMean, outs[1], "mean", 1e-4, 1e-3);
    normAssert(expectedInvStd, outs[2], "inv_std_var", 1e-4, 1e-3);
}

TEST(SkipSimplifiedLayerNormalizationLayer, RequiresAtLeastThreeInputs)
{
    LayerParams lp;
    lp.type = "SkipSimplifiedLayerNormalization";
    lp.name = "test_skip_norm";
    lp.set("epsilon", 1e-5f);
    Ptr<Layer> layer = LayerFactory::createLayerInstance("SkipSimplifiedLayerNormalization", lp);
    CV_Assert(layer);

    std::vector<MatShape> inputs = { MatShape({1, 1, 4}), MatShape({1, 1, 4}) };
    std::vector<MatShape> outputs, internals;
    EXPECT_ANY_THROW(layer->getMemoryShapes(inputs, 2, outputs, internals));
}

}} // namespace opencv_test::(anonymous)
