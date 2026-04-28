// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
LLM inference using the LLM class with multiple tokenizer backends.

Supports three modes via --tokenizer_type flag:
  ort_genai  : ORT-GenAI tokenizer (Llama, Phi, DeepSeek, etc.)
  gpt2       : OpenCV BPE tokenizer (GPT-2)
  qwen       : OpenCV BPE tokenizer (Qwen2.5)

=== ORT-GenAI (Llama, Phi, DeepSeek, etc.) ===

Download and convert a model using ORT-GenAI model builder:

    pip install onnxruntime-genai
    python3 -m onnxruntime_genai.models.builder \
        -m <hf_model_id> -o <output_dir> -p <precision> -e cpu

    where <precision> is fp32, fp16, or int4.

Build OpenCV with:
    -DWITH_ONNXRUNTIME=ON -DWITH_ONNXRUNTIME_GENAI=ON

To run:
    ./example_dnn_llm_inference --tokenizer_type=ort_genai --model=/path/to/ort_genai_model_dir \
        --prompt="What is OpenCV?" --max_new_tokens=100

=== GPT-2 with OpenCV BPE tokenizer ===

Exporting GPT-2 model to ONNX:

1. Clone fork of Andrej Karpathy's GPT-2 repository:

    git clone -b fix-dynamic-axis-export https://github.com/nklskyoy/build-nanogpt

2. Install the required dependencies:

    pip install -r requirements.txt

3. Export the model to ONNX:

    python export2onnx.py --promt=<Any-promt-you-want>

To run:
    ./example_dnn_llm_inference --tokenizer_type=gpt2 --model=/path/to/gpt2.onnx \
        --tokenizer=/path/to/gpt2/config.json \
        --prompt=<use-promt-of-the-same-length-used-while-exporting>

=== Qwen2.5 with OpenCV BPE tokenizer ===

Model: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

Exporting Qwen2.5 model to ONNX:

1. Install the required dependencies:

    pip install optimum[exporters] torch transformers

2. Export the model to ONNX:

    optimum-cli export onnx --model Qwen/Qwen2.5-0.5B-Instruct --task causal-lm qwen2.5_instruct_onnx/

To run:
    ./example_dnn_llm_inference --tokenizer_type=qwen --model=/path/to/qwen2.5.onnx \
        --tokenizer=/path/to/qwen2.5/config.json --prompt="What is OpenCV?" --max_new_tokens=100
*/

#include <iostream>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;

//
// ORT-GenAI path: uses OGA tokenizer + single run() call for full generation.
//
static void runOrtGenAI(const string& modelPath, const string& userPrompt, int maxNewTokens)
{
    // 1. Create LLM with ORT-GenAI tokenizer
    LLM llm = LLM::create(modelPath, TOKENIZER_ORT_GENAI);
    Tokenizer tokenizer = llm.getTokenizer();

    cout << "Model type  : " << llm.getModelType()  << endl;
    cout << "Device type : " << llm.getDeviceType() << endl;

    // 2. Apply chat template and encode prompt
    const string messages = "[{\"role\": \"user\", \"content\": \"" + userPrompt + "\"}]";
    const string prompt   = llm.applyChatTemplate(messages);
    vector<int> tokens    = tokenizer.encode(prompt);

    // 3. Configure generation parameters
    llm.setSearchOption("max_length", static_cast<double>(tokens.size() + maxNewTokens));
    llm.setSearchOptionBool("do_sample", false);

    // 4. Run inference and decode output
    Mat out = llm.run(tokens);
    vector<int> outIds(out.ptr<int>(), out.ptr<int>() + out.total());
    cout << tokenizer.decode(outIds) << endl;
}

//
// GPT-2 path: uses OpenCV BPE tokenizer + autoregressive greedy decoding loop.
//
static void runGpt2(const string& modelPath, const string& tokenizerCfg,
             const string& userPrompt, int maxSeqLen)
{
    // 1. Create LLM with OpenCV BPE tokenizer
    LLM llm = LLM::create(modelPath, TOKENIZER_OPENCV_BPE, tokenizerCfg);
    Tokenizer tokenizer = llm.getTokenizer();

    // 2. Encode prompt
    vector<int> tokens = tokenizer.encode(userPrompt);

    // 3. Autoregressive greedy decoding loop
    const int stopToken = 50256;  // <|endoftext|>
    int remaining = maxSeqLen;

    while (remaining > 0 && tokens.back() != stopToken)
    {
        Mat logits = llm.run(tokens, "idx");  // (1, seq_len, vocab_size)

        int seqLen    = logits.size[1];
        int vocabSize = logits.size[2];
        Mat lastLogits(1, vocabSize, CV_32F, logits.ptr<float>() + (seqLen - 1) * vocabSize);

        Point maxLoc;
        minMaxLoc(lastLogits, nullptr, nullptr, nullptr, &maxLoc);
        tokens.push_back(maxLoc.x);
        remaining--;
    }

    // 4. Decode and print
    cout << tokenizer.decode(tokens) << endl;
}

//
// Qwen2.5 path: uses OpenCV BPE tokenizer + autoregressive greedy decoding loop
// with multiple named inputs (input_ids, attention_mask, position_ids).
//
static void runQwen(const string& modelPath, const string& tokenizerCfg,
             const string& userPrompt, int maxNewTokens)
{
    // 1. Create LLM with OpenCV BPE tokenizer and ENGINE_NEW
    LLM llm = LLM::create(modelPath, TOKENIZER_OPENCV_BPE, tokenizerCfg, ENGINE_NEW);
    Tokenizer tokenizer = llm.getTokenizer();

    // 2. Encode prompt with ChatML format
    const string chatmlPrompt = "<|im_start|>user\n" + userPrompt + "<|im_end|>\n<|im_start|>assistant\n";
    vector<int> ids = tokenizer.encode(chatmlPrompt);
    Mat tokens(1, (int)ids.size(), CV_64F);
    for (int i = 0; i < (int)ids.size(); i++)
        tokens.at<double>(0, i) = static_cast<double>(ids[i]);
    tokens.convertTo(tokens, CV_64S);

    // 3. Autoregressive greedy decoding loop
    const vector<int> stopIds = {151645, 151643};  // <|im_end|>, <|endoftext|>

    for (int i = 0; i < maxNewTokens; i++)
    {
        int seqLen = tokens.cols;
        Mat attentionMask(1, seqLen, CV_64S, Scalar(1));
        Mat positionIds(1, seqLen, CV_64S);
        for (int j = 0; j < seqLen; j++)
            positionIds.at<int64_t>(0, j) = j;

        Mat logits = llm.run({tokens, attentionMask, positionIds},
                             {"input_ids", "attention_mask", "position_ids"});

        Mat lastLogits;
        if (logits.dims == 3)
        {
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
        for (int sid : stopIds)
        {
            if (newId == sid) { stop = true; break; }
        }
        if (stop) break;

        Mat newToken(1, 1, CV_64S, Scalar(static_cast<int64_t>(newId)));
        hconcat(tokens, newToken, tokens);
    }

    // 4. Decode and print
    Mat tokens32s;
    tokens.convertTo(tokens32s, CV_32S);
    vector<int> allIds(tokens32s.ptr<int>(), tokens32s.ptr<int>() + tokens32s.total());
    cout << tokenizer.decode(allIds) << endl;
}

const string param_keys =
    "{ help           h  |                  | Print help message. }"
    "{ tokenizer_type    | ort_genai        | Tokenizer type: ort_genai, gpt2, or qwen. }"
    "{ model          m  |                  | Path to ONNX model file or ORT-GenAI model directory (required). }"
    "{ tokenizer      t  |                  | Path to tokenizer config.json (required for gpt2/qwen). }"
    "{ prompt         p  | What is OpenCV?  | User prompt text. }"
    "{ max_new_tokens    | 100              | Maximum number of new tokens to generate (ort_genai/qwen). }"
    "{ max_seq_len       | 32               | Number of tokens to continue (gpt2 only). }";

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, param_keys);

    if (parser.has("help") || !parser.has("model"))
    {
        parser.printMessage();
        return 0;
    }

    const string tokenizerType = parser.get<String>("tokenizer_type");
    const string modelPath     = parser.get<String>("model");
    const string tokenizerCfg  = parser.get<String>("tokenizer");
    const string userPrompt    = parser.get<String>("prompt");
    const int    maxNewTokens  = parser.get<int>("max_new_tokens");
    const int    maxSeqLen     = parser.get<int>("max_seq_len");

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    if (tokenizerType == "ort_genai")
    {
        runOrtGenAI(modelPath, userPrompt, maxNewTokens);
    }
    else if (tokenizerType == "gpt2")
    {
        if (tokenizerCfg.empty())
        {
            cerr << "Error: --tokenizer is required for gpt2 tokenizer_type." << endl;
            return 1;
        }
        runGpt2(modelPath, tokenizerCfg, userPrompt, maxSeqLen);
    }
    else if (tokenizerType == "qwen")
    {
        if (tokenizerCfg.empty())
        {
            cerr << "Error: --tokenizer is required for qwen tokenizer_type." << endl;
            return 1;
        }
        runQwen(modelPath, tokenizerCfg, userPrompt, maxNewTokens);
    }
    else
    {
        cerr << "Error: Unknown tokenizer_type '" << tokenizerType << "'. Use ort_genai, gpt2, or qwen." << endl;
        return 1;
    }

    return 0;
}
