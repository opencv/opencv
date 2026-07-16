// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "npy_blob.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/dnn/all_layers.hpp>

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

static void runGQAModel(const std::string& basename, const std::vector<std::string>& inputNames,
                         std::vector<Mat>& outs)
{
    Net net = readNetFromONNX(findDataFile("dnn/onnx/models/" + basename + ".onnx", true), cv::dnn::ENGINE_NEW);
    ASSERT_FALSE(net.empty());

    for (size_t i = 0; i < inputNames.size(); ++i)
        net.setInput(blobFromNPY(findDataFile(format("dnn/onnx/data/input_%s_%d.npy", basename.c_str(), (int)i))),
                     inputNames[i]);

    net.forward(outs, std::vector<String>{"output", "present_key", "present_value"});
}

static void checkGQAOutputs(const std::string& basename, const std::vector<Mat>& outs)
{
    ASSERT_EQ(outs.size(), (size_t)3);
    Mat refOutput = blobFromNPY(findDataFile("dnn/onnx/data/output_" + basename + "_0.npy"));
    Mat refPresentKey = blobFromNPY(findDataFile("dnn/onnx/data/output_" + basename + "_1.npy"));
    Mat refPresentValue = blobFromNPY(findDataFile("dnn/onnx/data/output_" + basename + "_2.npy"));

    normAssert(refOutput, outs[0], "output", 1e-4, 1e-3);
    normAssert(refPresentKey, outs[1], "present_key", 1e-4, 1e-3);
    normAssert(refPresentValue, outs[2], "present_value", 1e-4, 1e-3);
}

static const std::vector<std::string> GQA_INPUT_NAMES = {
    "query", "key", "value", "past_key", "past_value",
    "seqlens_k", "total_sequence_length", "cos_cache", "sin_cache"
};

TEST(GroupQueryAttentionLayer, ONNXModel_CausalSelfAttentionNoCache)
{
    if (skipIfClassicEngineForced()) return;
    std::vector<Mat> outs;
    runGQAModel("group_query_attention_causal", GQA_INPUT_NAMES, outs);
    checkGQAOutputs("group_query_attention_causal", outs);
}

TEST(GroupQueryAttentionLayer, ONNXModel_GroupedHeadsMapToCorrectKVHead)
{
    if (skipIfClassicEngineForced()) return;
    std::vector<Mat> outs;
    runGQAModel("group_query_attention_grouped_heads", GQA_INPUT_NAMES, outs);
    checkGQAOutputs("group_query_attention_grouped_heads", outs);
}

TEST(GroupQueryAttentionLayer, ONNXModel_PresentKVConcatenatesPastAndNew)
{
    if (skipIfClassicEngineForced()) return;
    std::vector<Mat> outs;
    runGQAModel("group_query_attention_past_kv", GQA_INPUT_NAMES, outs);
    checkGQAOutputs("group_query_attention_past_kv", outs);
}

TEST(GroupQueryAttentionLayer, ONNXModel_LocalWindowRestrictsAttentionRange)
{
    if (skipIfClassicEngineForced()) return;
    std::vector<Mat> outs;
    runGQAModel("group_query_attention_local_window", GQA_INPUT_NAMES, outs);
    checkGQAOutputs("group_query_attention_local_window", outs);
}

TEST(GroupQueryAttentionLayer, ONNXModel_SoftcapClampsScores)
{
    if (skipIfClassicEngineForced()) return;
    std::vector<Mat> outs;
    runGQAModel("group_query_attention_softcap", GQA_INPUT_NAMES, outs);
    checkGQAOutputs("group_query_attention_softcap", outs);
}

TEST(GroupQueryAttentionLayer, ONNXModel_RotaryAppliesOnlyToNewToken)
{
    if (skipIfClassicEngineForced()) return;
    std::vector<Mat> outs;
    runGQAModel("group_query_attention_rotary", GQA_INPUT_NAMES, outs);
    checkGQAOutputs("group_query_attention_rotary", outs);
}

}} // namespace opencv_test::(anonymous)
