// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "opencv2/core/utils/configuration.private.hpp"

namespace opencv_test { namespace {

using namespace cv::vlm;

class MinimalVLMModel CV_FINAL : public VLMModel
{
public:
    void setPreferableDevice(const String&) CV_OVERRIDE {}
    void reset() CV_OVERRIDE {}
    String infer(InputArray, const String&, int) CV_OVERRIDE { return String(); }
    std::vector<String> inferDocument(const String&, const String&, int) CV_OVERRIDE
    {
        return std::vector<String>();
    }
};

TEST(Vlm_Model, LastTokensUsedBaseDefaultIsUnknown)
{
    MinimalVLMModel model;
    EXPECT_EQ(model.lastTokensUsed(), -1);
}

TEST(Vlm_Model, LastTokensUsedDefaultsToUnknown_CloudModel)
{
    cv::Ptr<VLMModel> model = create(VLM_MODEL_OPENAI, "gpt-4o", "new", "cloud", "fake-api-key");
    EXPECT_EQ(model->lastTokensUsed(), -1);
}

TEST(Vlm_Model, NonexistentModelDirThrows_PaddleOCRVL)
{
    EXPECT_THROW(create(VLM_MODEL_PADDLEOCR_VL, "/nonexistent/model/dir"), cv::Exception);
}

TEST(Vlm_Model, NonexistentModelDirThrows_GraniteDocling)
{
    EXPECT_THROW(create(VLM_MODEL_GRANITE_DOCLING, "/nonexistent/model/dir"), cv::Exception);
}

TEST(Vlm_Model, MissingApiKeyThrows_OpenAI)
{
    EXPECT_THROW(create(VLM_MODEL_OPENAI, ""), cv::Exception);
}

TEST(Vlm_Model, MissingApiKeyThrows_Anthropic)
{
    EXPECT_THROW(create(VLM_MODEL_ANTHROPIC, ""), cv::Exception);
}

TEST(Vlm_Model, MissingApiKeyThrows_Gemini)
{
    EXPECT_THROW(create(VLM_MODEL_GEMINI, ""), cv::Exception);
}

TEST(Vlm_Model, MissingApiKeyThrows_Grok)
{
    EXPECT_THROW(create(VLM_MODEL_GROK, ""), cv::Exception);
}

TEST(Vlm_Model, MissingModelNameThrows_OpenAI)
{
    EXPECT_THROW(create(VLM_MODEL_OPENAI, "", "new", "cloud", "some-api-key"), cv::Exception);
}

TEST(Vlm_Model, MissingModelNameThrows_Gemini)
{
    EXPECT_THROW(create(VLM_MODEL_GEMINI, "", "new", "cloud", "some-api-key"), cv::Exception);
}

TEST(Vlm_Model, UnknownEngineThrows_PaddleOCRVL)
{
    EXPECT_THROW(create(VLM_MODEL_PADDLEOCR_VL, "/nonexistent/model/dir", "tensorrt"), cv::Exception);
}

TEST(Vlm_Model, UnknownEngineThrows_GraniteDocling)
{
    EXPECT_THROW(create(VLM_MODEL_GRANITE_DOCLING, "/nonexistent/model/dir", "tensorrt"), cv::Exception);
}

TEST(Vlm_Model, EndToEnd_PaddleOCRVL)
{
    std::string modelDir = cv::utils::getConfigurationParameterString("OPENCV_TEST_VLM_PADDLEOCR_VL_DIR");
    if (modelDir.empty())
        throw SkipTestException("OPENCV_TEST_VLM_PADDLEOCR_VL_DIR is not set; skipping end-to-end test");

    std::string imagePath = cv::utils::getConfigurationParameterString("OPENCV_TEST_VLM_IMAGE");
    ASSERT_FALSE(imagePath.empty()) << "OPENCV_TEST_VLM_IMAGE must be set together with OPENCV_TEST_VLM_PADDLEOCR_VL_DIR";

    cv::Ptr<VLMModel> model = create(VLM_MODEL_PADDLEOCR_VL, modelDir);
    std::vector<cv::String> results = model->inferDocument(imagePath);
    ASSERT_EQ((size_t)1, results.size());
    EXPECT_FALSE(results[0].empty());
    EXPECT_GT(model->lastTokensUsed(), 0);
}

TEST(Vlm_Model, EndToEnd_GraniteDocling)
{
    std::string modelDir = cv::utils::getConfigurationParameterString("OPENCV_TEST_VLM_GRANITE_DOCLING_DIR");
    if (modelDir.empty())
        throw SkipTestException("OPENCV_TEST_VLM_GRANITE_DOCLING_DIR is not set; skipping end-to-end test");

    std::string imagePath = cv::utils::getConfigurationParameterString("OPENCV_TEST_VLM_IMAGE");
    ASSERT_FALSE(imagePath.empty()) << "OPENCV_TEST_VLM_IMAGE must be set together with OPENCV_TEST_VLM_GRANITE_DOCLING_DIR";

    cv::Ptr<VLMModel> model = create(VLM_MODEL_GRANITE_DOCLING, modelDir, "opencv");
    std::vector<cv::String> results = model->inferDocument(imagePath);
    ASSERT_EQ((size_t)1, results.size());
    EXPECT_FALSE(results[0].empty());
    EXPECT_GT(model->lastTokensUsed(), 0);
}

}} // namespace
