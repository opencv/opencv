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

void LocalVLMModelBase::setPreferableDevice(const String& device)
{
    CV_Assert(visionNet_ && embedNet_ && decoderNet_);

    int backendId, targetId;
    if (device == "cpu")
    {
        backendId = dnn::DNN_BACKEND_DEFAULT;
        targetId = dnn::DNN_TARGET_CPU;
    }
    else if (device == "cuda")
    {
        backendId = dnn::DNN_BACKEND_CUDA;
        targetId = dnn::DNN_TARGET_CUDA;
    }
    else
    {
        CV_Error(Error::StsBadArg,
                 "vlm: unknown device '" + device + "' (expected 'cpu' or 'cuda')");
    }

    dnn::Net* nets[] = {visionNet_, embedNet_, decoderNet_};
    for (dnn::Net* net : nets)
    {
        net->setPreferableBackend(backendId);
        net->setPreferableTarget(targetId);
    }
}

}} // namespace cv::vlm
