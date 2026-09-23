// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "../src/base64.hpp"
#include "../src/config_json.hpp"
#include "../src/json_parser.hpp"
#include "../src/vlm_generation.hpp"
#include "../src/local_vlm_model_base.hpp"
#include "../src/engines/granite_docling_preprocess.hpp"
#include "../src/engines/paddleocr_vl_preprocess.hpp"

// None of these helpers are exported from "opencv_vlm", so the only way to test them
// is to compile the source code into "opencv_test_vlm" (same approach as
// core's test_logtagmanager.cpp). This workaround may cause step debugger
// breakpoints to work unreliably.
#if 1
#include "../src/base64.cpp"
#include "../src/config_json.cpp"
#include "../src/json_parser.cpp"
#include "../src/vlm_generation.cpp"
#include "../src/vlm_model_base.cpp"
#include "../src/local_vlm_model_base.cpp"
#include "../src/engines/granite_docling_preprocess.cpp"
#include "../src/engines/paddleocr_vl_preprocess.cpp"
#include "../src/engines/cloud_engine.cpp"
#endif

#include <fstream>
#include <cstdio>

namespace opencv_test { namespace {

using namespace cv::vlm;

static std::string b64(const std::string& s)
{
    return base64Encode(reinterpret_cast<const unsigned char*>(s.data()), s.size());
}

TEST(Vlm_Base64, RFC4648TestVectors)
{
    EXPECT_EQ(b64(""), "");
    EXPECT_EQ(b64("f"), "Zg==");
    EXPECT_EQ(b64("fo"), "Zm8=");
    EXPECT_EQ(b64("foo"), "Zm9v");
    EXPECT_EQ(b64("foob"), "Zm9vYg==");
    EXPECT_EQ(b64("fooba"), "Zm9vYmE=");
    EXPECT_EQ(b64("foobar"), "Zm9vYmFy");
}

TEST(Vlm_Base64, FullByteRange)
{
    std::vector<unsigned char> data(256);
    for (int i = 0; i < 256; i++)
        data[i] = (unsigned char)i;

    std::string encoded = base64Encode(data.data(), data.size());
    EXPECT_EQ(encoded.size() % 4, 0u);
    EXPECT_EQ(encoded.size(), ((data.size() + 2) / 3) * 4);
}

TEST(Vlm_JsonParser, ScalarsAndNesting)
{
    const JsonValue root = jsonParse(
        "{\"s\": \"text\", \"i\": 42, \"neg\": -7, \"f\": 1.5, \"e\": 2e3,"
        " \"t\": true, \"f2\": false, \"n\": null, \"a\": [1, 2, 3],"
        " \"o\": {\"inner\": \"value\"}}");

    EXPECT_EQ(JsonValue::OBJECT, root.type());
    EXPECT_EQ("text", root["s"].asString());
    EXPECT_EQ(42, root["i"].asInt(-1));
    EXPECT_EQ(-7, root["neg"].asInt(0));
    EXPECT_EQ(2, root["f"].asInt(-1)); // 1.5 rounds to even, as cvRound does
    EXPECT_EQ(2000, root["e"].asInt(-1));
    EXPECT_EQ(1, root["t"].asInt(-1));
    EXPECT_EQ(0, root["f2"].asInt(-1));
    EXPECT_EQ((size_t)3, root["a"].size());
    EXPECT_EQ(3, root["a"][2].asInt(-1));
    EXPECT_EQ("value", root["o"]["inner"].asString());
}

TEST(Vlm_JsonParser, MissingAndNullBothFallBack)
{
    const JsonValue root = jsonParse("{\"usage\": null}");

    // A provider that sends "usage": null and one that omits it must behave the same way.
    EXPECT_TRUE(root["usage"].empty());
    EXPECT_TRUE(root["absent"].empty());
    EXPECT_EQ(-1, root["usage"]["total_tokens"].asInt(-1));
    EXPECT_EQ(-1, root["absent"]["total_tokens"].asInt(-1));
}

TEST(Vlm_JsonParser, LookupsOnWrongTypesNeverThrow)
{
    const JsonValue root = jsonParse("{\"s\": \"text\", \"a\": [1]}");

    EXPECT_TRUE(root["s"][0].empty());          // index into a string
    EXPECT_TRUE(root["s"]["nope"].empty());     // key into a string
    EXPECT_TRUE(root["a"][5].empty());          // out of range
    EXPECT_TRUE(root["a"]["nope"].empty());     // key into an array
    EXPECT_EQ((size_t)0, root["s"].size());
    EXPECT_EQ("", root["a"].asString());
}

TEST(Vlm_JsonParser, StringLongerThanFileStorageLimit)
{
    // The reason this parser exists: FileStorage's JSON reader rejects a single string
    // value at CV_FS_MAX_LEN (4096), and a page of recognized text is far longer.
    const std::string page(20000, 'x');
    const JsonValue root = jsonParse("{\"content\": \"" + page + "\"}");

    EXPECT_EQ((size_t)20000, root["content"].asString().size());
    EXPECT_EQ(page, root["content"].asString());
}

TEST(Vlm_JsonParser, Escapes)
{
    const JsonValue root = jsonParse(
        "{\"s\": \"a\\\"b\\\\c\\/d\\be\\ff\\ng\\rh\\ti\"}");

    EXPECT_EQ("a\"b\\c/d\be\ff\ng\rh\ti", root["s"].asString());
}

TEST(Vlm_JsonParser, UnicodeEscapeToUtf8)
{
    // U+00E1 a-acute and U+010C C-caron, 2 UTF-8 bytes each: the Czech test page needs both.
    const JsonValue root = jsonParse("{\"s\": \"P\\u00e1tek \\u010cervence\"}");

    // The literal is split because a C++ hex escape is greedy: "\xC4\x8Cervence" would read
    // \x8Ce as one escape, since 'e' is a hex digit. Do not join these back together.
    EXPECT_EQ("P\xC3\xA1tek \xC4\x8C" "ervence", root["s"].asString());
}

TEST(Vlm_JsonParser, SurrogatePairBecomesOneCodePoint)
{
    // U+1F600, which only reaches us as a surrogate pair, is 4 UTF-8 bytes.
    const JsonValue root = jsonParse("{\"s\": \"\\ud83d\\ude00\"}");

    EXPECT_EQ("\xF0\x9F\x98\x80", root["s"].asString());
}

TEST(Vlm_JsonParser, UnpairedSurrogateBecomesReplacementChar)
{
    // Never emit invalid UTF-8: a lone surrogate turns into U+FFFD.
    EXPECT_EQ("\xEF\xBF\xBD", jsonParse("{\"s\": \"\\ud83d\"}")["s"].asString());
    EXPECT_EQ("\xEF\xBF\xBD", jsonParse("{\"s\": \"\\ude00\"}")["s"].asString());
}

TEST(Vlm_JsonParser, WhitespaceAndEmptyContainers)
{
    const JsonValue root = jsonParse("  {\n\t\"a\" : [ ] ,\r \"o\" : { }\n}  ");

    EXPECT_EQ(JsonValue::ARRAY, root["a"].type());
    EXPECT_EQ((size_t)0, root["a"].size());
    EXPECT_EQ(JsonValue::OBJECT, root["o"].type());
    EXPECT_EQ((size_t)0, root["o"].size());
}

TEST(Vlm_JsonParser, MalformedInputThrows)
{
    EXPECT_THROW(jsonParse(""), cv::Exception);
    EXPECT_THROW(jsonParse("{"), cv::Exception);
    EXPECT_THROW(jsonParse("{\"a\""), cv::Exception);
    EXPECT_THROW(jsonParse("{\"a\": }"), cv::Exception);
    EXPECT_THROW(jsonParse("{\"a\" 1}"), cv::Exception);
    EXPECT_THROW(jsonParse("[1, 2"), cv::Exception);
    EXPECT_THROW(jsonParse("\"unterminated"), cv::Exception);
    EXPECT_THROW(jsonParse("{\"a\": \"bad \\q escape\"}"), cv::Exception);
    EXPECT_THROW(jsonParse("{\"a\": \"\\u12\"}"), cv::Exception);
    EXPECT_THROW(jsonParse("tru"), cv::Exception);
    EXPECT_THROW(jsonParse("{\"a\": 1} trailing"), cv::Exception);
}

TEST(Vlm_JsonParser, ExcessiveNestingThrowsInsteadOfOverflowingTheStack)
{
    const JsonValue shallow = jsonParse(std::string(60, '[') + std::string(60, ']'));
    EXPECT_EQ(JsonValue::ARRAY, shallow.type());

    const std::string tooDeep = std::string(500, '[') + std::string(500, ']');
    EXPECT_THROW(jsonParse(tooDeep), cv::Exception);
}

TEST(Vlm_JsonParser, OpenAIResponseShape)
{
    const JsonValue root = jsonParse(
        "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"page text\"}}],"
        " \"usage\":{\"prompt_tokens\":700,\"completion_tokens\":12,\"total_tokens\":712}}");

    EXPECT_EQ(712, root["usage"]["total_tokens"].asInt(-1));
    ASSERT_EQ((size_t)1, root["choices"].size());
    EXPECT_EQ("page text", root["choices"][0]["message"]["content"].asString());
}

TEST(Vlm_JsonParser, AnthropicResponseShape)
{
    const JsonValue root = jsonParse(
        "{\"content\":[{\"type\":\"text\",\"text\":\"page text\"}],"
        " \"usage\":{\"input_tokens\":700,\"output_tokens\":12}}");

    EXPECT_EQ(700, root["usage"]["input_tokens"].asInt(0));
    EXPECT_EQ(12, root["usage"]["output_tokens"].asInt(0));
    ASSERT_EQ((size_t)1, root["content"].size());
    EXPECT_EQ("page text", root["content"][0]["text"].asString());
}

TEST(Vlm_JsonParser, GeminiResponseShape)
{
    const JsonValue root = jsonParse(
        "{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"page text\"}],"
        "\"role\":\"model\"}}], \"usageMetadata\":{\"totalTokenCount\":712}}");

    EXPECT_EQ(712, root["usageMetadata"]["totalTokenCount"].asInt(-1));
    ASSERT_EQ((size_t)1, root["candidates"].size());
    const JsonValue& parts = root["candidates"][0]["content"]["parts"];
    ASSERT_EQ((size_t)1, parts.size());
    EXPECT_EQ("page text", parts[0]["text"].asString());
}

TEST(Vlm_CloudEngine, RedactsExactApiKeyFromErrorBody)
{
    const std::string key = "sk-proj-AbCdEf1234567890";
    const std::string body = "{\"error\":{\"message\":\"Incorrect API key provided: "
                             + key + ". Check your key.\"}}";

    const std::string out = redactSecrets(body, key);
    EXPECT_EQ(std::string::npos, out.find(key));
    EXPECT_NE(std::string::npos, out.find("<redacted>"));
    EXPECT_NE(std::string::npos, out.find("Incorrect API key provided"));
}

TEST(Vlm_CloudEngine, RedactsKeyShapedTokensNotMatchingTheConfiguredKey)
{
    // A provider that echoes only part of the key, or a stale key from elsewhere in the
    // body, still must not survive into the message.
    const std::string body = "{\"error\":\"bad key xai-9f8e7d6c5b4a and AIzaSyDzZzZz\"}";

    const std::string out = redactSecrets(body, "sk-completely-different");
    EXPECT_EQ(std::string::npos, out.find("9f8e7d6c5b4a"));
    EXPECT_EQ(std::string::npos, out.find("SyDzZzZz"));
    EXPECT_NE(std::string::npos, out.find("xai-<redacted>"));
    EXPECT_NE(std::string::npos, out.find("AIza<redacted>"));
}

TEST(Vlm_CloudEngine, RedactionLeavesOrdinaryTextAlone)
{
    const std::string body = "{\"error\":{\"type\":\"invalid_request_error\","
                             "\"message\":\"model gpt-4o-mini not found\"}}";

    EXPECT_EQ(body, redactSecrets(body, "sk-1234567890abcdef"));
}

TEST(Vlm_CloudEngine, RedactionIgnoresShortOrEmptyKeys)
{
    // An empty or implausibly short key must not turn into a match-everything pattern.
    const std::string body = "{\"error\":\"quota exceeded\"}";
    EXPECT_EQ(body, redactSecrets(body, ""));
    EXPECT_EQ(body, redactSecrets(body, "abc"));
}

TEST(Vlm_ConfigJson, GetIntPresentAndFallback)
{
    FileStorage fs("{\"a\": 5}", FileStorage::READ | FileStorage::MEMORY);
    EXPECT_EQ(getInt(fs.root(), "a", -1), 5);
    EXPECT_EQ(getInt(fs.root(), "missing", -1), -1);
}

TEST(Vlm_ConfigJson, GetFloatPresentAndFallback)
{
    FileStorage fs("{\"a\": 1.5}", FileStorage::READ | FileStorage::MEMORY);
    EXPECT_FLOAT_EQ(getFloat(fs.root(), "a", -1.f), 1.5f);
    EXPECT_FLOAT_EQ(getFloat(fs.root(), "missing", -1.f), -1.f);
}

TEST(Vlm_ConfigJson, GetVec3fPresentAndFallback)
{
    FileStorage fs("{\"a\": [1.0, 2.0, 3.0]}", FileStorage::READ | FileStorage::MEMORY);

    Vec3f value;
    getVec3f(fs.root(), "a", value, Vec3f(0.f, 0.f, 0.f));
    EXPECT_EQ(value, Vec3f(1.f, 2.f, 3.f));

    Vec3f fallbackValue;
    getVec3f(fs.root(), "missing", fallbackValue, Vec3f(9.f, 8.f, 7.f));
    EXPECT_EQ(fallbackValue, Vec3f(9.f, 8.f, 7.f));
}

TEST(Vlm_ConfigJson, GetVec3fWrongSizeThrows)
{
    FileStorage fs("{\"a\": [1.0, 2.0]}", FileStorage::READ | FileStorage::MEMORY);
    Vec3f value;
    EXPECT_THROW(getVec3f(fs.root(), "a", value, Vec3f(0.f, 0.f, 0.f)), cv::Exception);
}

TEST(Vlm_ConfigJson, GetIntWithTextConfigFallback)
{
    FileStorage topLevel("{\"eos_token_id\": 2}", FileStorage::READ | FileStorage::MEMORY);
    EXPECT_EQ(getIntWithTextConfigFallback(topLevel, "eos_token_id", -1), 2);

    FileStorage nested("{\"text_config\": {\"eos_token_id\": 7}}", FileStorage::READ | FileStorage::MEMORY);
    EXPECT_EQ(getIntWithTextConfigFallback(nested, "eos_token_id", -1), 7);

    FileStorage neither("{}", FileStorage::READ | FileStorage::MEMORY);
    EXPECT_EQ(getIntWithTextConfigFallback(neither, "eos_token_id", -1), -1);
}

TEST(Vlm_ConfigJson, GetIntWithTextConfigFallbackPrefersTopLevel)
{
    FileStorage fs("{\"eos_token_id\": 2, \"text_config\": {\"eos_token_id\": 7}}",
                   FileStorage::READ | FileStorage::MEMORY);
    EXPECT_EQ(getIntWithTextConfigFallback(fs, "eos_token_id", -1), 2);
}

TEST(Vlm_ConfigJson, OpenJsonConfigOrThrow_NonexistentPath)
{
    EXPECT_THROW(openJsonConfigOrThrow("/nonexistent/path/config.json"), cv::Exception);
}

TEST(Vlm_ConfigJson, OpenJsonConfigOrThrow_ValidFile)
{
    std::string path = cv::tempfile("vlm_test_config.json");
    {
        std::ofstream ofs(path.c_str());
        ofs << "{\"hello\": \"world\"}";
    }

    FileStorage fs = openJsonConfigOrThrow(path);
    std::string value;
    fs["hello"] >> value;
    EXPECT_EQ(value, "world");
    fs.release();

    remove(path.c_str());
}

TEST(Vlm_Generation, ArgmaxLastToken)
{
    int sizes[] = {1, 2, 4};
    Mat logits(3, sizes, CV_32F, Scalar(0));

    float* row0 = logits.ptr<float>(0, 0);
    row0[0] = 100.f; row0[1] = 100.f; row0[2] = 100.f; row0[3] = 100.f;

    float* row1 = logits.ptr<float>(0, 1);
    row1[0] = 0.1f; row1[1] = 0.2f; row1[2] = 5.0f; row1[3] = -1.0f;

    EXPECT_EQ(argmaxLastToken(logits), 2);
}

TEST(Vlm_Generation, ArgmaxLastTokenTieBreaksToFirst)
{
    int sizes[] = {1, 1, 3};
    Mat logits(3, sizes, CV_32F, Scalar(0));
    float* row = logits.ptr<float>(0, 0);
    row[0] = 5.f; row[1] = 5.f; row[2] = 1.f;

    EXPECT_EQ(argmaxLastToken(logits), 0);
}

TEST(Vlm_Generation, ScatterImageFeaturesCopiesAtImageTokenPositions)
{
    int hiddenDim = 4;
    int embedsSizes[] = {1, 3, hiddenDim};
    Mat inputsEmbeds(3, embedsSizes, CV_32F, Scalar(0));

    int featSizes[] = {1, 2, hiddenDim};
    Mat imageFeatures(3, featSizes, CV_32F);
    for (int f = 0; f < 2; f++)
    {
        float* row = imageFeatures.ptr<float>(0, f);
        for (int c = 0; c < hiddenDim; c++)
            row[c] = (float)(f * 10 + c);
    }

    int imageTokenId = 99;
    std::vector<int> tokens = {1, imageTokenId, imageTokenId};
    scatterImageFeatures(inputsEmbeds, tokens, imageTokenId, imageFeatures);

    const float* untouched = inputsEmbeds.ptr<float>(0, 0);
    for (int c = 0; c < hiddenDim; c++)
        EXPECT_EQ(untouched[c], 0.f);

    const float* first = inputsEmbeds.ptr<float>(0, 1);
    const float* second = inputsEmbeds.ptr<float>(0, 2);
    for (int c = 0; c < hiddenDim; c++)
    {
        EXPECT_EQ(first[c], (float)c);
        EXPECT_EQ(second[c], (float)(10 + c));
    }
}

TEST(Vlm_Generation, ScatterImageFeaturesThrowsWhenTooFewFeatures)
{
    int hiddenDim = 4;
    int embedsSizes[] = {1, 2, hiddenDim};
    Mat inputsEmbeds(3, embedsSizes, CV_32F, Scalar(0));

    int featSizes[] = {1, 1, hiddenDim};
    Mat imageFeatures(3, featSizes, CV_32F, Scalar(0));

    int imageTokenId = 99;
    std::vector<int> tokens = {imageTokenId, imageTokenId};
    EXPECT_THROW(scatterImageFeatures(inputsEmbeds, tokens, imageTokenId, imageFeatures), cv::Exception);
}

TEST(Vlm_Generation, ScatterImageFeaturesThrowsOnTokenCountMismatch)
{
    // More tokens than embedding rows used to walk off the end of the buffer.
    int hiddenDim = 4;
    int embedsSizes[] = {1, 2, hiddenDim};
    Mat inputsEmbeds(3, embedsSizes, CV_32F, Scalar(0));

    int featSizes[] = {1, 4, hiddenDim};
    Mat imageFeatures(3, featSizes, CV_32F, Scalar(0));

    int imageTokenId = 99;
    std::vector<int> tokens = {1, imageTokenId, imageTokenId, imageTokenId};
    EXPECT_THROW(scatterImageFeatures(inputsEmbeds, tokens, imageTokenId, imageFeatures),
                 cv::Exception);
}

TEST(Vlm_Generation, ScatterImageFeaturesRejectsWrongRankAndType)
{
    int hiddenDim = 4;
    int featSizes[] = {1, 1, hiddenDim};
    Mat imageFeatures(3, featSizes, CV_32F, Scalar(0));
    std::vector<int> tokens = {99};

    int flatSizes[] = {1, hiddenDim};
    Mat rank2(2, flatSizes, CV_32F, Scalar(0));
    EXPECT_THROW(scatterImageFeatures(rank2, tokens, 99, imageFeatures), cv::Exception);

    int embedsSizes[] = {1, 1, hiddenDim};
    Mat wrongType(3, embedsSizes, CV_64F, Scalar(0));
    EXPECT_THROW(scatterImageFeatures(wrongType, tokens, 99, imageFeatures), cv::Exception);
}

TEST(Vlm_Generation, ArgmaxLastTokenRejectsWrongRankAndType)
{
    int sizes2d[] = {1, 4};
    Mat rank2(2, sizes2d, CV_32F, Scalar(0));
    EXPECT_THROW(argmaxLastToken(rank2), cv::Exception);

    int sizes3d[] = {1, 1, 4};
    Mat wrongType(3, sizes3d, CV_64F, Scalar(0));
    EXPECT_THROW(argmaxLastToken(wrongType), cv::Exception);
}

TEST(Vlm_GraniteDoclingPreprocess, TileGridDimensionsForKnownAspectRatio)
{
    Mat image(400, 800, CV_8UC3, Scalar(0, 0, 0));
    int rows, cols;
    Mat pixelValues = tileImage(image, /*longestEdge=*/512, /*tileSize=*/256,
                                 Vec3f(0.f, 0.f, 0.f), Vec3f(1.f, 1.f, 1.f), rows, cols);

    EXPECT_EQ(rows, 1);
    EXPECT_EQ(cols, 2);

    int numTiles = rows * cols + 1;
    EXPECT_EQ(pixelValues.dims, 5);
    EXPECT_EQ(pixelValues.size[1], numTiles);
    EXPECT_EQ(pixelValues.size[3], 256);
    EXPECT_EQ(pixelValues.size[4], 256);
}

TEST(Vlm_GraniteDoclingPreprocess, TileGridDimensionsForPortraitImage)
{
    Mat image(800, 400, CV_8UC3, Scalar(0, 0, 0));
    int rows, cols;
    tileImage(image, /*longestEdge=*/512, /*tileSize=*/256, Vec3f(0.f, 0.f, 0.f), Vec3f(1.f, 1.f, 1.f),
              rows, cols);

    EXPECT_EQ(rows, 2);
    EXPECT_EQ(cols, 1);
}

TEST(Vlm_GraniteDoclingPreprocess, BuildPromptContainsRowColAndUserText)
{
    String prompt = buildGraniteDoclingPrompt(1, 1, 1, "hello");
    EXPECT_NE(prompt.find("<row_1_col_1>"), String::npos);
    EXPECT_NE(prompt.find("<global-img>"), String::npos);
    EXPECT_NE(prompt.find("hello"), String::npos);
    EXPECT_EQ(prompt.find("<|start_of_role|>user<|end_of_role|>"), (size_t)0);
    EXPECT_NE(prompt.find("<|start_of_role|>assistant<|end_of_role|>"), String::npos);
}

TEST(Vlm_PaddleOCRVLPreprocess, SmartResizeSnapsToFactorMultiples)
{
    int outHeight, outWidth;
    smartResize(10, 20, /*factor=*/2, /*minPixels=*/1, /*maxPixels=*/1000000, outHeight, outWidth);
    EXPECT_EQ(outHeight, 10);
    EXPECT_EQ(outWidth, 20);
}

TEST(Vlm_PaddleOCRVLPreprocess, SmartResizeUpscalesBelowFactor)
{
    int outHeight, outWidth;
    smartResize(1, 10, /*factor=*/4, /*minPixels=*/1, /*maxPixels=*/1000000, outHeight, outWidth);
    EXPECT_EQ(outHeight, 4);
    EXPECT_EQ(outWidth, 40);
}

TEST(Vlm_PaddleOCRVLPreprocess, SmartResizeThrowsOnExtremeAspectRatio)
{
    int outHeight, outWidth;
    EXPECT_THROW(smartResize(1, 300, /*factor=*/1, /*minPixels=*/1, /*maxPixels=*/1000000,
                              outHeight, outWidth),
                 cv::Exception);
}

TEST(Vlm_PaddleOCRVLPreprocess, PreprocessImageComputesGridAndShape)
{
    Mat image(28, 28, CV_8UC3, Scalar(0, 0, 0));
    int gridH, gridW;
    Mat pixelValues = preprocessImage(image, /*patchSize=*/14, /*mergeSize=*/1, /*minPixels=*/1,
                                       /*maxPixels=*/1000000, Vec3f(0.f, 0.f, 0.f), Vec3f(1.f, 1.f, 1.f),
                                       1.f, gridH, gridW);

    EXPECT_EQ(gridH, 2);
    EXPECT_EQ(gridW, 2);
    EXPECT_EQ(pixelValues.dims, 5);
    EXPECT_EQ(pixelValues.size[1], gridH * gridW);
    EXPECT_EQ(pixelValues.size[3], 14);
    EXPECT_EQ(pixelValues.size[4], 14);
}

TEST(Vlm_PaddleOCRVLPreprocess, BuildPromptRepeatsImagePlaceholder)
{
    String prompt = buildPaddleOCRVLPrompt("OCR", 3);
    EXPECT_EQ(prompt, "<|begin_of_sentence|>User: <|IMAGE_START|>"
                       "<|IMAGE_PLACEHOLDER|><|IMAGE_PLACEHOLDER|><|IMAGE_PLACEHOLDER|>"
                       "<|IMAGE_END|>OCR\nAssistant:\n");
}

class TestLocalVLMModel : public LocalVLMModelBase
{
public:
    using LocalVLMModelBase::registerNets;
    using LocalVLMModelBase::setLastTokensUsed;

protected:
    Mat runVisionEncoder(const Mat&, Vec2i& dimsOut) CV_OVERRIDE { dimsOut = Vec2i(0, 0); return Mat(); }
    String buildPrompt(const Vec2i&, const String&) const CV_OVERRIDE { return String(); }
    String defaultPrompt() const CV_OVERRIDE { return String(); }
};

TEST(Vlm_LocalModelBase, ResetWithoutNetsThrows)
{
    TestLocalVLMModel model;
    EXPECT_THROW(model.reset(), cv::Exception);
}

TEST(Vlm_LocalModelBase, SetDeviceWithoutNetsThrows)
{
    TestLocalVLMModel model;
    EXPECT_THROW(model.setPreferableDevice("cpu"), cv::Exception);
}

TEST(Vlm_LocalModelBase, SetDeviceUnknownDeviceThrows)
{
    TestLocalVLMModel model;
    dnn::Net vision, embed, decoder;
    model.registerNets(vision, embed, decoder);
    EXPECT_THROW(model.setPreferableDevice("opencl"), cv::Exception);
}

TEST(Vlm_LocalModelBase, SetDeviceCpuAndCudaDoNotThrow)
{
    TestLocalVLMModel model;
    dnn::Net vision, embed, decoder;
    model.registerNets(vision, embed, decoder);
    EXPECT_NO_THROW(model.setPreferableDevice("cpu"));
    EXPECT_NO_THROW(model.setPreferableDevice("cuda"));
}

TEST(Vlm_LocalModelBase, LastTokensUsedDefaultsToUnknown)
{
    TestLocalVLMModel model;
    EXPECT_EQ(model.lastTokensUsed(), -1);
}

TEST(Vlm_LocalModelBase, LastTokensUsedReflectsSetValue)
{
    TestLocalVLMModel model;
    model.setLastTokensUsed(123);
    EXPECT_EQ(model.lastTokensUsed(), 123);

    model.setLastTokensUsed(456);
    EXPECT_EQ(model.lastTokensUsed(), 456);
}

}} // namespace opencv_test::(anonymous)
