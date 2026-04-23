// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026,BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include <iostream>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;

const string about =
    "Text generation using the LLM class with multiple tokenizer backends.\n\n"
    "Supports three modes via --tokenizer_type flag:\n"
    "  ort_genai  : ORT-GenAI tokenizer + inference (requires WITH_ONNXRUNTIME_GENAI=ON)\n"
    "  gpt2       : OpenCV BPE tokenizer (GPT-2) + ONNX inference\n"
    "  qwen       : OpenCV BPE tokenizer (Qwen2.5) + ONNX inference\n\n"
    "Examples:\n"
    "  # ORT-GenAI (Llama, Phi, etc.)\n"
    "  ./example_dnn_text_generation --tokenizer_type=ort_genai --model=/path/to/ort_genai_model_dir\n\n"
    "  # GPT-2 with OpenCV tokenizer\n"
    "  ./example_dnn_text_generation --tokenizer_type=gpt2 --model=/path/to/gpt2.onnx --tokenizer=/path/to/gpt2/config.json\n\n"
    "  # Qwen2.5 with OpenCV tokenizer\n"
    "  ./example_dnn_text_generation --tokenizer_type=qwen --model=/path/to/qwen2.5.onnx --tokenizer=/path/to/qwen2.5/config.json\n";

const string param_keys =
    "{ help           h  |                  | Print help message. }"
    "{ tokenizer_type    | ort_genai        | Tokenizer type: ort_genai, gpt2, or qwen. }"
    "{ model          m  |                  | Path to ONNX model file or ORT-GenAI model directory (required). }"
    "{ tokenizer      t  |                  | Path to tokenizer config.json (required for gpt2/qwen backends). }"
    "{ prompt         p  | What is OpenCV?  | User prompt text. }"
    "{ max_new_tokens    | 100              | Maximum number of new tokens to generate. }";

static Mat greedyDecode(Net& net, const Mat& inputTokens, int maxNewTokens,
                         const string& inputName = "", const vector<int>& stopTokens = {})
{
    Mat tokens = inputTokens.clone();

    for (int i = 0; i < maxNewTokens; i++)
    {
        if (inputName.empty())
            net.setInput(tokens);
        else
            net.setInput(tokens, inputName);

        Mat logits = net.forward();
        logits = logits.reshape(1, logits.size[1]);  // (seq_len, vocab_size)

        // Take last token logits
        Mat lastLogits = logits.row(logits.rows - 1);
        Point maxLoc;
        minMaxLoc(lastLogits, nullptr, nullptr, nullptr, &maxLoc);
        int newId = maxLoc.x;

        // Check stop tokens
        for (int sid : stopTokens)
        {
            if (newId == sid)
                return tokens;
        }

        // Append new token
        Mat newToken(1, 1, CV_32S, Scalar(newId));
        hconcat(tokens, newToken, tokens);
    }
    return tokens;
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, param_keys);
    parser.about(about);

    if (parser.has("help") || !parser.has("model"))
    {
        parser.printMessage();
        return 0;
    }

    const string tokenizerType =parser.get<String>("tokenizer_type");
    const string modelPath    = parser.get<String>("model");
    const string tokenizerCfg = parser.get<String>("tokenizer");
    const string userPrompt   = parser.get<String>("prompt");
    const int    maxNewTokens = parser.get<int>("max_new_tokens");

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    if (tokenizerType =="ort_genai")
    {
        // ---- ORT-GenAI path ----
        LLM llm = LLM::create(modelPath, TOKENIZER_ORT_GENAI);

        cout << "Model type  : " << llm.getModelType()  << endl;
        cout << "Device type : " << llm.getDeviceType() << endl;

        const string messages = "[{\"role\": \"user\", \"content\": \"" + userPrompt + "\"}]";
        const string prompt   = llm.applyChatTemplate(messages);

        Mat tokens = llm.tokenize(prompt);
        llm.setSearchOption("max_length", static_cast<double>(tokens.cols + maxNewTokens));
        llm.setSearchOptionBool("do_sample", false);

        Net net = llm.getNet();
        net.setInput(tokens);
        Mat out = net.forward();

        cout << llm.detokenize(out) << endl;
    }
    else if (tokenizerType =="gpt2")
    {
        // ---- GPT-2 path ----
        if (tokenizerCfg.empty())
        {
            cerr << "Error: --tokenizer is required for gpt2 backend." << endl;
            return 1;
        }

        LLM llm = LLM::create(modelPath, TOKENIZER_OPENCV_BPE, tokenizerCfg);

        Mat tokens = llm.tokenize(userPrompt);
        Net net = llm.getNet();

        vector<int> stopTokens = {50256};  // <|endoftext|>
        Mat out = greedyDecode(net, tokens, maxNewTokens, "idx", stopTokens);

        cout << llm.detokenize(out) << endl;
    }
    else if (tokenizerType =="qwen")
    {
        // ---- Qwen2.5 path ----
        if (tokenizerCfg.empty())
        {
            cerr << "Error: --tokenizer is required for qwen backend." << endl;
            return 1;
        }

        LLM llm = LLM::create(modelPath, TOKENIZER_OPENCV_BPE, tokenizerCfg);

        // ChatML format
        const string chatmlPrompt = "<|im_start|>user\n" + userPrompt + "<|im_end|>\n<|im_start|>assistant\n";
        vector<int> ids = llm.encode(chatmlPrompt);
        Mat tokens(1, (int)ids.size(), CV_64F);
        for (int i = 0; i < (int)ids.size(); i++)
            tokens.at<double>(0, i) = static_cast<double>(ids[i]);
        tokens.convertTo(tokens, CV_64S);

        Net net = llm.getNet();

        vector<int> stopTokens = {151645, 151643};  // <|im_end|>, <|endoftext|>

        for (int i = 0; i < maxNewTokens; i++)
        {
            int seqLen = tokens.cols;
            Mat attentionMask(1, seqLen, CV_64S, Scalar(1));
            Mat positionIds(1, seqLen, CV_64S);
            for (int j = 0; j < seqLen; j++)
                positionIds.at<int64_t>(0, j) = j;

            net.setInput(tokens, "input_ids");
            net.setInput(attentionMask, "attention_mask");
            net.setInput(positionIds, "position_ids");
            Mat logits = net.forward();

            // Take last token logits
            Mat lastLogits;
            if (logits.dims == 3)
            {
                // (1, seq_len, vocab_size) -> last row
                int vocabSize = logits.size[2];
                lastLogits = Mat(1, vocabSize, CV_32F, logits.ptr<float>() + (seqLen - 1) * vocabSize);
            }
            else
            {
                lastLogits = logits.row(logits.rows - 1);
            }

            Point maxLoc;
            minMaxLoc(lastLogits, nullptr, nullptr, nullptr, &maxLoc);
            int newId = maxLoc.x;

            bool stop = false;
            for (int sid : stopTokens)
            {
                if (newId == sid) { stop = true; break; }
            }
            if (stop) break;

            Mat newToken(1, 1, CV_64S, Scalar(static_cast<int64_t>(newId)));
            hconcat(tokens, newToken, tokens);
        }

        // Decode output tokens
        Mat tokens32s;
        tokens.convertTo(tokens32s, CV_32S);
        cout << llm.detokenize(tokens32s) << endl;
    }
    else
    {
        cerr << "Error: Unknown tokenizer_type '" << tokenizerType << "'. Use ort_genai, gpt2, or qwen." << endl;
        return 1;
    }

    return 0;
}
