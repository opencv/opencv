// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
 * This is a sample program demonstrating cv::vlm::create(): a single API for running
 * vision-language OCR / document-understanding inference either locally (PaddleOCR-VL-1.5,
 * Granite-Docling-258M) or via a hosted cloud API (OpenAI, Anthropic, Gemini, Grok), given a
 * model type, a local ONNX export directory (or cloud model name), and an input image.
 *
 * Run the sample:
 *
 *      ./vlm_ocr --model_type=paddleocr-vl --model_dir=<dir> --input=<path-to-image>
 *      export OPENAI_API_KEY=...   # or ANTHROPIC_API_KEY / GEMINI_API_KEY / XAI_API_KEY
 *      ./vlm_ocr --model_type=openai --model_dir=<provider model name> --input=<path-to-image>
 *
 * Cloud model types read the key from the provider's usual environment variable. Prefer that
 * over --api_key, which puts the key in your shell history and in the process list.
 */

#include <iostream>
#include <map>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/vlm.hpp>

using namespace cv;
using namespace cv::vlm;
using namespace std;

int main(int argc, char** argv)
{
    const string keys =
        "{ help h         |      | Print help message }"
        "{ model_type     |      | Which VLM to run: paddleocr-vl, granite-docling, openai, anthropic, gemini or grok }"
        "{ model_dir      |      | Local model types: path to the ONNX export directory. "
        "Cloud model types: provider model name, taken from the provider's model list (required, no default) }"
        "{ api_key        |      | API key for cloud model types; ignored otherwise. Leave unset to read it from "
        "OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY / XAI_API_KEY instead }"
        "{ input i        |      | Path to the input image (.png/.jpg/.jpeg) }"
        "{ prompt         |      | Task prompt (defaults to the model's built-in prompt) }"
        "{ max_new_tokens | 512  | Maximum number of new tokens to generate }"
        "{ engine         | opencv | Local model types only: dnn engine used to load each ONNX sub-model: opencv }"
        "{ device         | cpu  | Local model types only: compute device: cpu or cuda }";

    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this sample to run vision-language OCR / document-understanding inference in OpenCV");
    if (parser.has("help") || !parser.has("model_type") || !parser.has("input"))
    {
        parser.printMessage();
        return 0;
    }

    static const map<string, VLMModelType> MODEL_TYPES = {
        {"paddleocr-vl",     VLM_MODEL_PADDLEOCR_VL},
        {"granite-docling",  VLM_MODEL_GRANITE_DOCLING},
        {"openai",           VLM_MODEL_OPENAI},
        {"anthropic",        VLM_MODEL_ANTHROPIC},
        {"gemini",           VLM_MODEL_GEMINI},
        {"grok",             VLM_MODEL_GROK},
    };

    string modelTypeArg = parser.get<String>("model_type");
    auto it = MODEL_TYPES.find(modelTypeArg);
    if (it == MODEL_TYPES.end())
    {
        cerr << "Unknown model_type: " << modelTypeArg << endl;
        return 1;
    }
    VLMModelType modelType = it->second;
    bool isCloud = (modelType == VLM_MODEL_OPENAI || modelType == VLM_MODEL_ANTHROPIC ||
                    modelType == VLM_MODEL_GEMINI || modelType == VLM_MODEL_GROK);

    if (!parser.has("model_dir"))
    {
        cerr << "model_dir is required: local model types need the ONNX export directory, "
                "cloud model types need the provider's model name" << endl;
        return 1;
    }

    string engine = parser.get<String>("engine");
    if (!isCloud && engine != "opencv")
    {
        cerr << "Unknown engine: " << engine << " (expected opencv)" << endl;
        return 1;
    }

    string device = "cloud";
    if (!isCloud)
    {
        device = parser.get<String>("device");
        if (device != "cpu" && device != "cuda")
        {
            cerr << "Unknown device: " << device << " (expected cpu or cuda)" << endl;
            return 1;
        }
    }

    string modelDir = parser.get<String>("model_dir");
    string apiKey = parser.get<String>("api_key");
    string inputPath = parser.get<String>("input");
    string prompt = parser.get<String>("prompt");
    int maxNewTokens = parser.get<int>("max_new_tokens");

    cout << "Preparing " << modelTypeArg << " model..." << endl;
    Ptr<VLMModel> model = create(modelType, modelDir, engine, device, apiKey);

    cout << "Running inference on " << inputPath << "..." << endl;
    vector<String> results = model->inferDocument(inputPath, prompt, maxNewTokens);
    for (size_t i = 0; i < results.size(); i++)
        cout << "Page " << (i + 1) << ":\n" << results[i] << endl;

    return 0;
}
