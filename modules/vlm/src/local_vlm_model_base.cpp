// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "local_vlm_model_base.hpp"
#include "vlm_generation.hpp"

namespace cv { namespace vlm {

String LocalVLMModelBase::infer(InputArray image, const String& prompt, int max_new_tokens)
{
    CV_Assert(visionNet_ && embedNet_ && decoderNet_);

    Mat imageBgr = image.getMat();
    CV_CheckFalse(imageBgr.empty(), "vlm: input image is empty");
    String actualPrompt = prompt.empty() ? defaultPrompt() : prompt;

    Vec2i dims;
    Mat imageEmbeds = runVisionEncoder(imageBgr, dims);
    String fullPrompt = buildPrompt(dims, actualPrompt);

    std::vector<int> tokens = tokenizer_.encode(fullPrompt);
    int promptLen = (int)tokens.size();
    std::vector<int64_t> inputIdsData(tokens.begin(), tokens.end());
    int idsShape[] = {1, promptLen};
    Mat inputIds(2, idsShape, CV_64S, inputIdsData.data());

    embedNet_->setInput(inputIds, "input_ids");
    Mat inputsEmbeds = embedNet_->forward();

    scatterImageFeatures(inputsEmbeds, tokens, imageTokenId_, imageEmbeds);

    std::vector<int> generated = generateWithKVCache(*embedNet_, *decoderNet_, inputsEmbeds,
                                                      promptLen, max_new_tokens, eosTokenId_);
    setLastTokensUsed(promptLen + (int)generated.size());
    return tokenizer_.decode(generated);
}

void LocalVLMModelBase::registerNets(dnn::Net& visionNet, dnn::Net& embedNet, dnn::Net& decoderNet)
{
    visionNet_ = &visionNet;
    embedNet_ = &embedNet;
    decoderNet_ = &decoderNet;
}

void LocalVLMModelBase::reset()
{
    CV_Assert(decoderNet_);
    decoderNet_->resetKVCache();
}

int LocalVLMModelBase::lastTokensUsed() const
{
    return lastTokensUsed_;
}

void LocalVLMModelBase::setLastTokensUsed(int tokens)
{
    lastTokensUsed_ = tokens;
}

void LocalVLMModelBase::setPreferableBackend(dnn::Backend backendId)
{
    CV_Assert(visionNet_ && embedNet_ && decoderNet_);
    visionNet_->setPreferableBackend(backendId);
    embedNet_->setPreferableBackend(backendId);
    decoderNet_->setPreferableBackend(backendId);
}

void LocalVLMModelBase::setPreferableTarget(dnn::Target targetId)
{
    CV_Assert(visionNet_ && embedNet_ && decoderNet_);
    visionNet_->setPreferableTarget(targetId);
    embedNet_->setPreferableTarget(targetId);
    decoderNet_->setPreferableTarget(targetId);
}

}} // namespace cv::vlm
