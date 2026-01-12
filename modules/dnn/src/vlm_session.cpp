// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "precomp.hpp"
#include "net_impl.hpp"
#include "kv_cache_manager.hpp"
namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN
vLMSession::vLMSession(const String &languageModelPath)
{
    net = readNetFromONNX2(languageModelPath);
    net.setProfilingMode(DNN_PROFILE_DETAILED);
    Net::Impl* netImpl = net.getImpl();
    netImpl->useKVCache = true;
    setKVCacheManager(netImpl);
}

vLMSession::~vLMSession(){}

void vLMSession::setPrompt(Mat inputEmbedding_)
{
    inputEmbedding = inputEmbedding_;
    int T = inputEmbedding.size[1];
    int B = inputEmbedding.size[0];

    cv::Mat position_ids(B, T, CV_32U); // 1 row, N columns, 32-bit signed integers

    for (int i = 0; i < B; i++)
        for (int j = 0; j < T; j++)
            position_ids.at<int>(i, j) = j;

    net.setInput(inputEmbedding, "input_embedding");
    net.setInput(position_ids, "input_pos_idx");

    std::vector<Mat> outputs;
    std::vector<std::string> outputNames = {"next_token_id", "next_token_embedding"};

    net.forward(outputs, outputNames);

    generatedTokens.push_back(outputs[0].at<int>(0,0));
    generatedTokenEmbeddings.push_back(outputs[1]);
    nGeneratedTokens += (T + 1);
}

void vLMSession::generate()
{
    Mat position_ids(1, 1, CV_32U); // 1 row, 1 column, 32-bit signed integers
    int newTokenId = -1;
    for (int i = 0; i < maxTokens && (isDummy || newTokenId != finalTokenId); i++)
    {
        int curPosIdx = nGeneratedTokens - 1;

        position_ids.at<int>(0, 0) = curPosIdx;

        net.setInput(generatedTokenEmbeddings.back(), "input_embedding");
        net.setInput(position_ids, "input_pos_idx");

        std::vector<Mat> outputs;
        std::vector<std::string> outputNames = {"next_token_id", "next_token_embedding"};

        net.forward(outputs, outputNames);

        newTokenId = outputs[0].at<int>(0,0);
        generatedTokens.push_back(newTokenId);
        generatedTokenEmbeddings.push_back(outputs[1]);

        nGeneratedTokens += 1;
    }
}

std::vector<int> vLMSession::getGeneratedTokens() const
{
    return generatedTokens;
}

std::vector<Mat> vLMSession::getGeneratedTokenEmbeddings() const
{
    return generatedTokenEmbeddings;
}



CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn