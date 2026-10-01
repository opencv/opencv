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

static void runGQAModel(const std::string& basename, const std::vector<std::string>& inputNames,
                         std::vector<Mat>& outs)
{
    Net net = readNetFromONNX(findDataFile("dnn/onnx/models/" + basename + ".onnx", true));
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

// cos_cache / sin_cache exist only on a node that sets do_rotary, as in real exports,
// so the non-rotary models stop at total_sequence_length.
static const std::vector<std::string> GQA_INPUT_NAMES = {
    "query", "key", "value", "past_key", "past_value",
    "seqlens_k", "total_sequence_length"
};

static const std::vector<std::string> GQA_ROTARY_INPUT_NAMES = {
    "query", "key", "value", "past_key", "past_value",
    "seqlens_k", "total_sequence_length", "cos_cache", "sin_cache"
};

// (model basename, whether the node sets do_rotary and so carries cos_cache/sin_cache)
typedef std::tuple<std::string, bool> GQAModelParam;
typedef testing::TestWithParam<GQAModelParam> Test_GQA_Model;

TEST_P(Test_GQA_Model, Accuracy)
{
    const std::string basename = std::get<0>(GetParam());
    const bool rotary = std::get<1>(GetParam());
    SCOPED_TRACE(basename);

    std::vector<Mat> outs;
    runGQAModel(basename, rotary ? GQA_ROTARY_INPUT_NAMES : GQA_INPUT_NAMES, outs);
    checkGQAOutputs(basename, outs);
}

// A GroupQueryAttention node must reach exactly one AttentionOnnxAi layer and no layer of
// the old standalone type. Naming that type here is deliberate: if the forked lowering ever
// comes back, this fails instead of passing quietly.
TEST_P(Test_GQA_Model, LowersToOneAttentionLayer)
{
    const std::string basename = std::get<0>(GetParam());
    SCOPED_TRACE(basename);

    Net net = readNetFromONNX(findDataFile("dnn/onnx/models/" + basename + ".onnx", true));
    ASSERT_FALSE(net.empty());

    int unified = 0, forked = 0;
    const std::vector<String> names = net.getLayerNames();
    for (size_t i = 0; i < names.size(); ++i)
    {
        const String type = net.getLayer(net.getLayerId(names[i]))->type;
        if (type == "AttentionOnnxAi")
            unified++;
        else if (type == "GroupQueryAttention")
            forked++;
    }
    EXPECT_EQ(unified, 1);
    EXPECT_EQ(forked, 0);
}

INSTANTIATE_TEST_CASE_P(/**/, Test_GQA_Model, testing::Values(
    GQAModelParam("group_query_attention_causal",        false),  // causal self-attention, no cache
    GQAModelParam("group_query_attention_grouped_heads", false),  // grouped heads map to the right KV head
    GQAModelParam("group_query_attention_past_kv",       false),  // present = past concatenated with new
    GQAModelParam("group_query_attention_local_window",  false),  // sliding window restricts the range
    GQAModelParam("group_query_attention_softcap",       false),  // softcap clamps the scores
    GQAModelParam("group_query_attention_rotary",        true)    // rotary applies to the new token only
));

// ---- guards on the GroupQueryAttention path ----
// Built through LayerParams with the flags the importer sets, so these are the same code
// paths a real export takes without needing a malformed model on disk.

static const int GQA_NUM_HEADS = 4, GQA_KV_NUM_HEADS = 2, GQA_HEAD_SIZE = 8;

static Net makeGQANet()
{
    LayerParams lp;
    lp.type = "AttentionOnnxAi";
    lp.name = "gqa";
    lp.set("q_num_heads", GQA_NUM_HEADS);
    lp.set("kv_num_heads", GQA_KV_NUM_HEADS);
    lp.set("is_causal", true);
    lp.set("has_attn_mask", 0);
    lp.set("has_past", 1);
    lp.set("has_seqlens_k", 1);
    lp.set("has_rotary_cache", 0);

    Net net;
    int id = net.addLayerToPrev(lp.name, lp.type, lp);
    for (int i = 0; i < 6; ++i)
        net.connect(0, i, id, i);
    net.setInputsNames(std::vector<String>{"query", "key", "value",
                                           "past_key", "past_value", "seqlens_k"});
    return net;
}

// The fed blob shapes, not any declared shape, are what the layer infers from, so a
// malformed case has to be malformed here.
static void feedGQANet(Net& net, int S, int keyS, int pastLen, int pastHeads, int pastD,
                       int seqlens)
{
    const int B = 1, D = GQA_HEAD_SIZE;
    net.setInput(Mat(std::vector<int>{B, S, GQA_NUM_HEADS * D}, CV_32F, Scalar(0.1f)), "query");
    net.setInput(Mat(std::vector<int>{B, keyS, GQA_KV_NUM_HEADS * D}, CV_32F, Scalar(0.1f)), "key");
    net.setInput(Mat(std::vector<int>{B, keyS, GQA_KV_NUM_HEADS * D}, CV_32F, Scalar(0.1f)), "value");
    net.setInput(Mat(std::vector<int>{B, pastHeads, pastLen, pastD}, CV_32F, Scalar(0.1f)), "past_key");
    net.setInput(Mat(std::vector<int>{B, pastHeads, pastLen, pastD}, CV_32F, Scalar(0.1f)), "past_value");
    net.setInput(Mat(std::vector<int>{B}, CV_32S, Scalar(seqlens)), "seqlens_k");
}

// Control: an over-strict guard would fail here rather than in the rejection tests below.
TEST(GroupQueryAttentionLayer, Guard_AcceptsWellFormedNode)
{
    Net net = makeGQANet();
    feedGQANet(net, /*S*/4, /*keyS*/4, /*pastLen*/2, /*pastHeads*/GQA_KV_NUM_HEADS,
               /*pastD*/GQA_HEAD_SIZE, /*seqlens*/5);
    EXPECT_NO_THROW(net.forward());
}

// A shorter key is an out-of-bounds read, not just a wrong answer: the shared-buffer write
// and the rotary pass both step through key/value a query length at a time.
TEST(GroupQueryAttentionLayer, Guard_RejectsKeyShorterThanQuery)
{
    Net net = makeGQANet();
    feedGQANet(net, /*S*/4, /*keyS*/2, /*pastLen*/2, /*pastHeads*/GQA_KV_NUM_HEADS,
               /*pastD*/GQA_HEAD_SIZE, /*seqlens*/5);
    EXPECT_ANY_THROW(net.forward());
}

TEST(GroupQueryAttentionLayer, Guard_RejectsWrongPastHeadCount)
{
    Net net = makeGQANet();
    feedGQANet(net, /*S*/4, /*keyS*/4, /*pastLen*/2, /*pastHeads*/GQA_KV_NUM_HEADS + 1,
               /*pastD*/GQA_HEAD_SIZE, /*seqlens*/5);
    EXPECT_ANY_THROW(net.forward());
}

TEST(GroupQueryAttentionLayer, Guard_RejectsWrongPastHeadSize)
{
    Net net = makeGQANet();
    feedGQANet(net, /*S*/4, /*keyS*/4, /*pastLen*/2, /*pastHeads*/GQA_KV_NUM_HEADS,
               /*pastD*/GQA_HEAD_SIZE * 2, /*seqlens*/5);
    EXPECT_ANY_THROW(net.forward());
}

TEST(GroupQueryAttentionLayer, Guard_RejectsSeqlensSmallerThanQuery)
{
    Net net = makeGQANet();
    feedGQANet(net, /*S*/4, /*keyS*/4, /*pastLen*/2, /*pastHeads*/GQA_KV_NUM_HEADS,
               /*pastD*/GQA_HEAD_SIZE, /*seqlens*/1);
    EXPECT_ANY_THROW(net.forward());
}

}} // namespace opencv_test::(anonymous)
