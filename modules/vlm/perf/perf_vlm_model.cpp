// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"
#include "opencv2/core/utils/configuration.private.hpp"
#include "../src/vlm_generation.hpp"
#include "../src/engines/granite_docling_preprocess.hpp"
#include "../src/engines/paddleocr_vl_preprocess.hpp"

// Not exported from "opencv_vlm", so compile them in (as core's test_logtagmanager.cpp does).
#if 1
#include "../src/vlm_generation.cpp"
#include "../src/engines/preprocess_common.cpp"
#include "../src/engines/granite_docling_preprocess.cpp"
#include "../src/engines/paddleocr_vl_preprocess.cpp"
#endif

namespace opencv_test {

// Realistic page size, synthetic so these run with no model or test data.
static Mat syntheticPage()
{
    Mat page(818, 1125, CV_8UC3);
    randu(page, Scalar::all(0), Scalar::all(255));
    return page;
}

// Uses OPENCV_TEST_VLM_IMAGE when set, otherwise a synthetic page.
static Mat perfPage()
{
    const std::string path =
        cv::utils::getConfigurationParameterString("OPENCV_TEST_VLM_IMAGE");
    if (!path.empty())
    {
        Mat page = imread(path, IMREAD_COLOR);
        if (!page.empty())
            return page;
    }
    return syntheticPage();
}

PERF_TEST(Vlm_GraniteDoclingPreprocess, TileImage)
{
    const Mat page = perfPage();
    int rows = 0, cols = 0;
    Mat pixelValues;

    TEST_CYCLE()
        pixelValues = tileImage(page, 1536, 512, Vec3f(0.5f, 0.5f, 0.5f), Vec3f(0.5f, 0.5f, 0.5f),
                                rows, cols);

    EXPECT_FALSE(pixelValues.empty());
    std::cout << "[   INFO   ] " << page.cols << "x" << page.rows
              << " -> " << cols << "x" << rows << " tiles + thumbnail" << std::endl;
    SANITY_CHECK_NOTHING();
}

PERF_TEST(Vlm_PaddleOCRVLPreprocess, PreprocessImage)
{
    const Mat page = syntheticPage();
    int gridH = 0, gridW = 0;
    Mat pixelValues;

    TEST_CYCLE()
        pixelValues = preprocessImage(page, 14, 2, 112896, 1003520,
                                      Vec3f(0.5f, 0.5f, 0.5f), Vec3f(0.5f, 0.5f, 0.5f),
                                      1.f / 255.f, gridH, gridW);

    EXPECT_FALSE(pixelValues.empty());
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
    Ptr<VLMModel> model =
        createModelOrSkip(VLM_MODEL_PADDLEOCR_VL, "OPENCV_TEST_VLM_PADDLEOCR_VL_DIR");
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
    Ptr<VLMModel> model =
        createModelOrSkip(VLM_MODEL_GRANITE_DOCLING, "OPENCV_TEST_VLM_GRANITE_DOCLING_DIR");
    Mat image = loadPerfImageOrSkip();

    TEST_CYCLE()
    {
        model->reset();
        model->infer(image);
    }

    SANITY_CHECK_NOTHING();
}

} // namespace opencv_test
