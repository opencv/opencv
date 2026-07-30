// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "paddleocr_vl_engine.hpp"
#include "../local_vlm_model_base.hpp"
#include "../vlm_generation.hpp"
#include "../config_json.hpp"
#include "paddleocr_vl_preprocess.hpp"

namespace cv { namespace vlm {

using namespace cv::dnn;

namespace {

const String DEFAULT_PROMPT = "OCR";

class PaddleOCRVLModel CV_FINAL : public LocalVLMModelBase
{
public:
    PaddleOCRVLModel(const String& model_dir, int engine, const String& device)
    {
        tokenizer_ = Tokenizer::load(model_dir + "/");

        FileStorage config = openJsonConfigOrThrow(model_dir + "/config.json");
        FileStorage processorFs = openJsonConfigOrThrow(model_dir + "/processor_config.json");
        FileNode preprocessor = processorFs["image_processor"];

        imageTokenId_ = getIntWithTextConfigFallback(config, "image_token_id", 0);
        eosTokenId_ = getIntWithTextConfigFallback(config, "eos_token_id", 2);

        patchSize_ = getInt(preprocessor, "patch_size", 14);
        mergeSize_ = getInt(preprocessor, "merge_size", 2);
        CV_CheckGT(patchSize_, 0, "vlm: patch_size must be positive");
        CV_CheckGT(mergeSize_, 0, "vlm: merge_size must be positive");
        minPixels_ = getInt(preprocessor, "min_pixels", 28 * 28 * 130);
        maxPixels_ = getInt(preprocessor, "max_pixels", 28 * 28 * 1280);
        rescaleFactor_ = getFloat(preprocessor, "rescale_factor", 1.0f / 255.0f);
        getVec3f(preprocessor, "image_mean", mean_, Vec3f(0.5f, 0.5f, 0.5f));
        getVec3f(preprocessor, "image_std", std_, Vec3f(0.5f, 0.5f, 0.5f));

        visionNet_ = readNetFromONNX(model_dir + "/onnx/vision_encoder.onnx", engine);
        embedNet_ = readNetFromONNX(model_dir + "/onnx/embedding.onnx", engine);
        decoderNet_ = readNetFromONNX(model_dir + "/onnx/decoder.onnx", engine);
        registerNets(visionNet_, embedNet_, decoderNet_);
        setPreferableDevice(device);
    }

protected:
    Mat runVisionEncoder(const Mat& imageBgr, Vec2i& dimsOut) CV_OVERRIDE
    {
        int gridH, gridW;
        Mat pixelValues = preprocessImage(imageBgr, patchSize_, mergeSize_, minPixels_, maxPixels_,
                                          mean_, std_, rescaleFactor_, gridH, gridW);
        int gridShape[] = {1, 3};
        Mat imageGridThw(2, gridShape, CV_64S);
        imageGridThw.at<int64_t>(0, 0) = 1;
        imageGridThw.at<int64_t>(0, 1) = gridH;
        imageGridThw.at<int64_t>(0, 2) = gridW;

        visionNet_.setInput(pixelValues, "pixel_values");
        visionNet_.setInput(imageGridThw, "image_grid_thw");
        dimsOut = Vec2i(gridH, gridW);
        return visionNet_.forward();
    }

    String buildPrompt(const Vec2i& dims, const String& userPrompt) const CV_OVERRIDE
    {
        int imageTokenRepeats = (int)((1LL * dims[0] * dims[1]) / mergeSize_ / mergeSize_);
        return buildPaddleOCRVLPrompt(userPrompt, imageTokenRepeats);
    }

    String defaultPrompt() const CV_OVERRIDE { return DEFAULT_PROMPT; }

private:
    Net visionNet_, embedNet_, decoderNet_;
    int patchSize_ = 14, mergeSize_ = 2, minPixels_ = 28 * 28 * 130, maxPixels_ = 28 * 28 * 1280;
    float rescaleFactor_ = 1.0f / 255.0f;
    Vec3f mean_ = Vec3f(0.5f, 0.5f, 0.5f), std_ = Vec3f(0.5f, 0.5f, 0.5f);
};

} // namespace

Ptr<VLMModel> createPaddleOCRVLModel(const String& model_dir, int engine, const String& device)
{
    return makePtr<PaddleOCRVLModel>(model_dir, engine, device);
}

}} // namespace cv::vlm
