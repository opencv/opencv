// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"
#include "opencv2/core/utils/configuration.private.hpp"
#include "../src/base64.hpp"
#include "../src/vlm_generation.hpp"

namespace opencv_test {

PERF_TEST(Vlm_Base64, EncodePngSizedBuffer)
{
    std::vector<unsigned char> data(512 * 1024);
    for (size_t i = 0; i < data.size(); i++)
        data[i] = (unsigned char)(i & 0xFF);

    std::string encoded;
    TEST_CYCLE()
    {
        encoded = cv::vlm::base64Encode(data.data(), data.size());
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST(Vlm_Generation, ArgmaxLastToken)
{
    int sizes[] = {1, 1, 151936};
    Mat logits(3, sizes, CV_32F);
    cv::randu(logits, -10.f, 10.f);

    int result = 0;
    TEST_CYCLE()
    {
        result = cv::vlm::argmaxLastToken(logits);
    }

    EXPECT_GE(result, 0);
    SANITY_CHECK_NOTHING();
}

static Ptr<VLMModel> createModelOrSkip(VLMModelType type, const char* envVar)
{
    std::string modelDir = cv::utils::getConfigurationParameterString(envVar);
    if (modelDir.empty())
        throw SkipTestException(std::string(envVar) + " is not set; skipping end-to-end perf test");
    return create(type, modelDir);
}

static Mat loadPerfImageOrSkip()
{
    std::string imagePath = cv::utils::getConfigurationParameterString("OPENCV_TEST_VLM_IMAGE");
    if (imagePath.empty())
        throw SkipTestException("OPENCV_TEST_VLM_IMAGE is not set; skipping end-to-end perf test");
    Mat image = imread(imagePath, IMREAD_COLOR);
    if (image.empty())
        throw SkipTestException("could not read OPENCV_TEST_VLM_IMAGE: " + imagePath);
    return image;
}

PERF_TEST(Vlm_Model, Infer_PaddleOCRVL)
{
    Ptr<VLMModel> model = createModelOrSkip(VLM_MODEL_PADDLEOCR_VL, "OPENCV_TEST_VLM_PADDLEOCR_VL_DIR");
    Mat image = loadPerfImageOrSkip();

    TEST_CYCLE()
    {
        model->reset();
        model->infer(image);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST(Vlm_Model, Infer_GraniteDocling)
{
    Ptr<VLMModel> model = createModelOrSkip(VLM_MODEL_GRANITE_DOCLING, "OPENCV_TEST_VLM_GRANITE_DOCLING_DIR");
    Mat image = loadPerfImageOrSkip();

    TEST_CYCLE()
    {
        model->reset();
        model->infer(image);
    }

    SANITY_CHECK_NOTHING();
}

} // namespace opencv_test
