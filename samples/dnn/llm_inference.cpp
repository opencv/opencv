// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#include <iostream>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;

const string about =
    "Text generation using ORT-GenAI backend via OpenCV DNN.\n\n"
    "Download and convert a model using ORT-GenAI model builder:\n"
    "    pip install onnxruntime-genai\n"
    "    python3 -m onnxruntime_genai.models.builder \\\n"
    "        -m <hf_model_id> -o <output_dir> -p <precision> -e cpu\n\n"
    "    where <precision> is fp32, fp16, or int4.\n\n"
    "    Note for Llama and  GPT-OSS models: after conversion, open tokenizer_config.json and change\n"
    "        \"tokenizer_class\": \"TokenizersBackend\"\n"
    "    to\n"
    "        \"tokenizer_class\": \"PreTrainedTokenizer\"\n\n"
    "Build OpenCV with:\n"
    "     -DWITH_ONNXRUNTIME=ON  -DWITH_ONNXRUNTIME_GENAI=ON\n\n"
    "To run:\n"
    "    ./example_dnn_llm_inference --model=<model_dir> [--prompt=\"<text>\"] [--max_new_tokens=<n>]\n";

const string param_keys =
    "{ help           h  |                  | Print help message. }"
    "{ model          m  |                  | Path to ONNX GenAI model directory (required). }"
    "{ prompt         p  | What is OpenCV?  | User prompt text. }"
    "{ max_new_tokens    | 100              | Maximum number of new tokens to generate. }";

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, param_keys);
    parser.about(about);

    if (parser.has("help") || !parser.has("model"))
    {
        parser.printMessage();
        return 0;
    }

    const string modelDir     = parser.get<String>("model");
    const string userMsg      = parser.get<String>("prompt");
    const int    maxNewTokens = parser.get<int>("max_new_tokens");

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    Net net = readNetFromONNX(modelDir, ENGINE_ORT_GENAI);
    if (net.empty())
    {
        cerr << "Error: Failed to load model from: " << modelDir << endl;
        return 1;
    }

    cout << "Model type  : " << net.getModelType()  << endl;
    cout << "Device type : " << net.getDeviceType() << endl;

    const string messages = "[{\"role\": \"user\", \"content\": \"" + userMsg + "\"}]";
    const string prompt   = net.applyChatTemplate(messages);

    Mat tokens = net.tokenize(prompt);
    net.setSearchOption("max_length", static_cast<double>(tokens.cols + maxNewTokens));
    net.setSearchOptionBool("do_sample", false);  // greedy — deterministic output
    net.setInput(tokens);
    Mat out = net.forward();

    cout << net.detokenize(out) << endl;

    return 0;
}
