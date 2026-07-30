// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "granite_docling_engine.hpp"
#include "../local_vlm_model_base.hpp"
#include "../vlm_generation.hpp"
#include "../config_json.hpp"
#include "granite_docling_preprocess.hpp"

namespace cv { namespace vlm {

using namespace cv::dnn;

namespace {

const String DEFAULT_PROMPT =
    "Convert this page to docling. Preserve OCR text, table structure, "
    "form fields, and layout/section structure.";

class GraniteDoclingModel CV_FINAL : public LocalVLMModelBase
{
public:
    GraniteDoclingModel(const String& model_dir, int engine, const String& device)
    {
        tokenizer_ = Tokenizer::load(model_dir + "/");

        FileStorage config = openJsonConfigOrThrow(model_dir + "/config.json");
        FileStorage preprocessor = openJsonConfigOrThrow(model_dir + "/preprocessor_config.json");
        FileStorage processor = openJsonConfigOrThrow(model_dir + "/processor_config.json");

        imageTokenId_ = getIntWithTextConfigFallback(config, "image_token_id", 0);
        eosTokenId_ = getIntWithTextConfigFallback(config, "eos_token_id", 2);
        imageSeqLen_ = getInt(processor.root(), "image_seq_len", 64);
        longestEdge_ = getInt(preprocessor["size"], "longest_edge", 1536);
        maxTileEdge_ = getInt(preprocessor["max_image_size"], "longest_edge", 512);
        CV_CheckGT(longestEdge_, 0, "vlm: longest_edge must be positive");
        CV_CheckGT(maxTileEdge_, 0, "vlm: max_image_size.longest_edge must be positive");
        getVec3f(preprocessor.root(), "image_mean", mean_, Vec3f(0.5f, 0.5f, 0.5f));
        getVec3f(preprocessor.root(), "image_std", std_, Vec3f(0.5f, 0.5f, 0.5f));

        visionNet_ = readNetFromONNX(model_dir + "/onnx/vision_encoder.onnx", engine);
        embedNet_ = readNetFromONNX(model_dir + "/onnx/embed_tokens.onnx", engine);
        decoderNet_ = readNetFromONNX(model_dir + "/onnx/decoder_model_merged.onnx", engine);
        registerNets(visionNet_, embedNet_, decoderNet_);
        setPreferableDevice(device);
    }

protected:
    Mat runVisionEncoder(const Mat& imageBgr, Vec2i& dimsOut) CV_OVERRIDE
    {
        int rows, cols;
        Mat pixelValues = tileImage(imageBgr, longestEdge_, maxTileEdge_, mean_, std_, rows, cols);

        int maskShape[] = {1, pixelValues.size[1], pixelValues.size[3], pixelValues.size[4]};
        Mat pixelAttentionMask(4, maskShape, CV_Bool, Scalar(1));

        visionNet_.setInput(pixelValues, "pixel_values");
        visionNet_.setInput(pixelAttentionMask, "pixel_attention_mask");
        dimsOut = Vec2i(rows, cols);
        return visionNet_.forward();
    }

    String buildPrompt(const Vec2i& dims, const String& userPrompt) const CV_OVERRIDE
    {
        return buildGraniteDoclingPrompt(dims[0], dims[1], imageSeqLen_, userPrompt);
    }

    String defaultPrompt() const CV_OVERRIDE { return DEFAULT_PROMPT; }

private:
    Net visionNet_, embedNet_, decoderNet_;
    int imageSeqLen_ = 64;
    int longestEdge_ = 1536, maxTileEdge_ = 512;
    Vec3f mean_ = Vec3f(0.5f, 0.5f, 0.5f), std_ = Vec3f(0.5f, 0.5f, 0.5f);
};

} // namespace

Ptr<VLMModel> createGraniteDoclingModel(const String& model_dir, int engine, const String& device)
{
    return makePtr<GraniteDoclingModel>(model_dir, engine, device);
}

}} // namespace cv::vlm
