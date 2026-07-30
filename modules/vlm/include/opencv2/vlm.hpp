// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_VLM_HPP
#define OPENCV_VLM_HPP

#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"

/**
  @defgroup vlm Vision-Language Model based OCR / Document Understanding

  This module wraps ONNX-exported vision-language models behind a single inference
  API so that document OCR / understanding engines can be swapped without rewriting
  the surrounding preprocessing and generation-loop code. See VLMModel.
 */

namespace cv { namespace vlm {

//! @addtogroup vlm
//! @{

/** @brief Supported vision-language OCR / document-understanding model types.

VLM_MODEL_PADDLEOCR_VL and VLM_MODEL_GRANITE_DOCLING run locally from ONNX weights (see
create()'s model_dir param). VLM_MODEL_OPENAI/ANTHROPIC/GEMINI/GROK call the provider's
hosted vision API over HTTPS instead (requires OpenCV built with libcurl available, and an
api_key passed to create()).
*/
enum VLMModelType
{
    VLM_MODEL_PADDLEOCR_VL    = 0,  //!< PaddleOCR-VL-1.5, see https://huggingface.co/PaddlePaddle/PaddleOCR-VL
    VLM_MODEL_GRANITE_DOCLING = 1,  //!< Granite-Docling-258M, see https://huggingface.co/ibm-granite/granite-docling-258M

    VLM_MODEL_OPENAI          = 2,  //!< OpenAI hosted vision API (e.g. gpt-4o), requires api_key
    VLM_MODEL_ANTHROPIC       = 3,  //!< Anthropic Claude hosted vision API, requires api_key
    VLM_MODEL_GEMINI          = 4,  //!< Google Gemini hosted vision API, requires api_key
    VLM_MODEL_GROK            = 5   //!< xAI Grok hosted vision API, requires api_key
};

/** @brief Base class for vision-language OCR / document-understanding engines.

Each engine wraps a set of ONNX sub-models (a vision encoder, a text embedding model,
and a KV-cached decoder), its Tokenizer, and its own image-preprocessing pipeline
behind one inference call. Construct an engine with create().
 */
class CV_EXPORTS_W VLMModel
{
public:
    virtual ~VLMModel();

    /** @brief Change the compute device used by all underlying nets: "cpu" or "cuda". */
    CV_WRAP virtual void setPreferableDevice(const String& device) = 0;

    /** @brief Run inference on a single already-decoded image/page.

    @param image           BGR image (e.g. from imread()).
    @param prompt          Task prompt; an empty string uses the engine's default prompt.
    @param max_new_tokens  Maximum number of tokens to generate.
    @return The engine's raw generated text (an OCR string, or doctags-style markup,
    depending on the engine) -- not a final structured result. A downstream API is
    expected to parse this into blocks/tables/bounding boxes; infer() intentionally
    stays at the raw-text level.
    */
    CV_WRAP virtual String infer(InputArray image, const String& prompt = String(),
                                  int max_new_tokens = 512) = 0;

    /** @brief Run inference on a document file.

    @param input_path      Path to a `.png`/`.jpg`/`.jpeg` image (treated as a single page).
                            PDF input is not supported yet -- rasterize pages to images
                            first (e.g. with poppler-utils' pdftoppm).
    @param prompt          Task prompt; an empty string uses the engine's default prompt.
    @param max_new_tokens  Maximum number of tokens to generate, per page.
    @return One result string per page (always size 1 for now, since only single-image
    input is supported). Each page is generated from a clean state -- reset() runs
    internally before every page, so pages never see each other's KV-cache / context.
    */
    CV_WRAP virtual std::vector<String> inferDocument(CV_WRAP_FILE_PATH const String& input_path,
                                                       const String& prompt = String(),
                                                       int max_new_tokens = 512) = 0;

    /** @brief Clear KV-cache / generation state.

    Only needed if you call infer() manually multiple times on the same instance --
    inferDocument() already does this per page internally.
    */
    CV_WRAP virtual void reset() = 0;

    /** @brief Total tokens (prompt + completion) reported by the most recent infer() call.

    Cloud model types report this from the provider's response; local model types report
    prompt length + generated token count from their own tokenizer/generation loop. The
    base class default of -1 (unknown) only applies to a VLMModel subclass that overrides
    neither.
    */
    CV_WRAP virtual int lastTokensUsed() const { return -1; }
};

/** @brief Create a vision-language OCR / document-understanding engine.

@param model_type Which VLM to load. VLM_MODEL_PADDLEOCR_VL/VLM_MODEL_GRANITE_DOCLING run
                   locally; VLM_MODEL_OPENAI/ANTHROPIC/GEMINI/GROK call a hosted API.
@param model_dir  For local model types: path to the local ONNX export directory, following
                   the upstream layout documented in samples/dnn/granite_docling_inference.py
                   and samples/dnn/paddleocr_vl_inference.py.
                   For cloud model types: the provider's model name, e.g. "gpt-4o",
                   "claude-3-5-sonnet-20241022", "gemini-2.0-flash", "grok-2-vision-1212" --
                   required, with no built-in default, since provider model names change over
                   time and a hardcoded guess would eventually go stale and fail confusingly.
@param engine     cv::dnn::Net engine used to load each underlying ONNX sub-model: "opencv"
                  (dnn::ENGINE_OPENCV) or "ort" (dnn::ENGINE_ORT). Ignored by cloud model types.
@param device     For local model types: compute device for all underlying nets, "cpu" or
                  "cuda". For cloud model types: pass "cloud" (there is no local compute
                  device to pick; any value is accepted here and has no effect).
@param api_key    API key for cloud model types; required for VLM_MODEL_OPENAI/ANTHROPIC/
                  GEMINI/GROK, ignored by local model types.
*/
CV_EXPORTS_W Ptr<VLMModel> create(VLMModelType model_type, CV_WRAP_FILE_PATH const String& model_dir,
                                   const String& engine = "new",
                                   const String& device = "cpu",
                                   const String& api_key = String());

//! @}

}} // namespace cv::vlm

#endif // OPENCV_VLM_HPP
