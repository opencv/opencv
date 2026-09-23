// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "vlm_generation.hpp"

#include <algorithm>
#include <cstring>

namespace cv { namespace vlm {

using namespace cv::dnn;

int argmaxLastToken(const Mat& logits)
{
    // Shape and type come from a downloaded ONNX export, so they are input, not an
    // invariant: check before indexing size[] and reinterpreting the data as float.
    CV_CheckEQ(logits.dims, 3, "vlm: decoder logits must be 1xSxV");
    CV_CheckTypeEQ(logits.type(), CV_32F, "vlm: decoder logits must be CV_32F");
    CV_CheckGT(logits.size[1], 0, "vlm: decoder logits have an empty sequence dimension");
    CV_CheckGT(logits.size[2], 0, "vlm: decoder logits have an empty vocabulary dimension");

    int seqLen = logits.size[1];
    int vocabSize = logits.size[2];
    const float* row = logits.ptr<float>(0, seqLen - 1);
    return (int)(std::max_element(row, row + vocabSize) - row);
}

void scatterImageFeatures(Mat& inputsEmbeds, const std::vector<int>& tokens,
                           int imageTokenId, const Mat& imageFeatures)
{
    // The write below is indexed by token position, so a token list longer than the
    // embedding sequence would run off the end of the buffer. Both shapes come from the
    // model export and the tokenizer config, which can disagree with each other.
    CV_CheckEQ(inputsEmbeds.dims, 3, "vlm: inputs_embeds must be 1xSxH");
    CV_CheckTypeEQ(inputsEmbeds.type(), CV_32F, "vlm: inputs_embeds must be CV_32F");
    CV_CheckTypeEQ(imageFeatures.type(), CV_32F, "vlm: vision features must be CV_32F");
    // Both buffers are addressed flat below, which a padded Mat would break.
    CV_Assert(inputsEmbeds.isContinuous() && imageFeatures.isContinuous());
    CV_CheckEQ((int)tokens.size(), inputsEmbeds.size[1],
               "vlm: token count and embedding sequence length disagree "
               "-- model export and tokenizer config are inconsistent");

    int hiddenDim = inputsEmbeds.size[2];
    CV_CheckGT(hiddenDim, 0, "vlm: inputs_embeds has an empty hidden dimension");
    CV_CheckEQ((int)(imageFeatures.total() % (size_t)hiddenDim), 0,
               "vlm: vision feature count is not a multiple of the embedding hidden size");

    int numFeatures = (int)(imageFeatures.total() / hiddenDim);
    float* embedsData = inputsEmbeds.ptr<float>();
    const float* featData = imageFeatures.ptr<float>();
    int featIdx = 0;
    for (size_t i = 0; i < tokens.size(); i++)
        if (tokens[i] == imageTokenId)
        {
            CV_CheckLT(featIdx, numFeatures,
                       "vlm: more <image> tokens than vision-encoder output features "
                       "-- model export and preprocessor config are inconsistent");
            memcpy(embedsData + i * hiddenDim, featData + (size_t)(featIdx++) * hiddenDim,
                   hiddenDim * sizeof(float));
        }
}

std::vector<int> generateWithKVCache(Net& embedNet, Net& decoderNet,
                                      const Mat& promptInputsEmbeds, int promptLen,
                                      int maxNewTokens, int eosTokenId)
{
    decoderNet.enableKVCache();

    std::vector<int64_t> maskData(promptLen, 1);
    int attnShape[] = {1, promptLen};
    Mat attentionMask(2, attnShape, CV_64S, maskData.data());

    decoderNet.setInput(promptInputsEmbeds, "inputs_embeds");
    decoderNet.setInput(attentionMask, "attention_mask");
    Mat logits = decoderNet.forward();
    int newId = argmaxLastToken(logits);
    std::vector<int> generated = {newId};

    for (int step = 0; step < maxNewTokens - 1; step++)
    {
        if (newId == eosTokenId)
            break;

        int64_t idData[1] = {newId};
        int idShape[] = {1, 1};
        Mat newIdMat(2, idShape, CV_64S, idData);
        embedNet.setInput(newIdMat, "input_ids");
        Mat newEmbed = embedNet.forward();

        maskData.push_back(1);
        int newAttnShape[] = {1, (int)maskData.size()};
        Mat newAttentionMask(2, newAttnShape, CV_64S, maskData.data());

        decoderNet.setInput(newEmbed, "inputs_embeds");
        decoderNet.setInput(newAttentionMask, "attention_mask");
        logits = decoderNet.forward();
        newId = argmaxLastToken(logits);
        generated.push_back(newId);
    }

    if (!generated.empty() && generated.back() == eosTokenId)
        generated.pop_back();

    return generated;
}

}} // namespace cv::vlm
