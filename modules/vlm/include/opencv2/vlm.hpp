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
    //! PaddleOCR-VL-1.5, see https://huggingface.co/PaddlePaddle/PaddleOCR-VL
    VLM_MODEL_PADDLEOCR_VL    = 0,
    //! Granite-Docling-258M, see https://huggingface.co/ibm-granite/granite-docling-258M
    VLM_MODEL_GRANITE_DOCLING = 1,

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

    /// @sa dnn::Net::setPreferableBackend
    CV_WRAP virtual void setPreferableBackend(dnn::Backend backendId) = 0;

    /// @sa dnn::Net::setPreferableTarget
    CV_WRAP virtual void setPreferableTarget(dnn::Target targetId) = 0;

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

    /** @brief Clear KV-cache / generation state.

    Needed only when calling infer() more than once on the same instance, so that one
    call does not see the previous one's KV-cache / context.
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

PADDLEOCR_VL and GRANITE_DOCLING run locally from an ONNX export; OPENAI, ANTHROPIC, GEMINI
and GROK call a hosted API.

@param model_type Which VLM to load.
@param model_dir  Local types: the ONNX export directory, laid out as below. Cloud types: the
                  provider's model name, taken from the provider's own model list.
@param api_key    Cloud types only. Empty reads OPENAI_API_KEY, ANTHROPIC_API_KEY,
                  GEMINI_API_KEY or XAI_API_KEY instead; create() throws naming the variable
                  if neither is set.

Local types load on dnn::ENGINE_OPENCV, the only engine whose KV cache these decoders need.
Set a backend or target afterwards with VLMModel::setPreferableBackend() and
VLMModel::setPreferableTarget(); note ENGINE_OPENCV supports CPU only for now, so a non-CPU
target is logged and ignored by dnn.

### Model directory layout

VLM_MODEL_GRANITE_DOCLING, exported from
<https://huggingface.co/onnx-community/granite-docling-258M-ONNX>:

    <model_dir>/
      config.json               OpenCV tokenizer config -- see below
      preprocessor_config.json  image_mean, image_std, size and max_image_size longest_edge
      processor_config.json     image_seq_len
      tokenizer.json
      onnx/vision_encoder.onnx
      onnx/embed_tokens.onnx
      onnx/decoder_model_merged.onnx

VLM_MODEL_PADDLEOCR_VL, exported from <https://huggingface.co/PaddlePaddle/PaddleOCR-VL>:

    <model_dir>/
      config.json               OpenCV tokenizer config -- see below
      processor_config.json     patch size, merge size, pixel bounds and normalization
      tokenizer.json
      onnx/vision_encoder.onnx
      onnx/embedding.onnx
      onnx/decoder.onnx

Note the two differ in their ONNX file names, and that PaddleOCR-VL reads no
preprocessor_config.json.

`config.json` is **not** HuggingFace's: it is the descriptor dnn::Tokenizer::load() consumes
(`model_type`, `method`, `vocab_size`, `tokenizer_class`, the token strings), plus
`image_token_id` and `eos_token_id`, which may instead sit under a `text_config` object. A
stock upstream export needs that one file added. Use the full-precision decoder: the
int4-kquant variant needs MatMulNBits' asymmetric form and the int8 variant needs
ai.onnx MatMulInteger, neither of which is implemented.
*/
CV_EXPORTS_W Ptr<VLMModel> create(VLMModelType model_type, const String& model_dir,
                                  const String& api_key = String());

/** @brief Convenience wrapper: reads an image file and runs VLMModel::infer() on it.

Equivalent to `model->reset()` followed by `model->infer(imread(path), ...)`.

@param model           Engine to run, from create().
@param path            Path to a `.png`/`.jpg`/`.jpeg` image. PDF is not supported --
                       rasterize pages first (e.g. with poppler-utils' pdftoppm).
@param prompt          Task prompt; an empty string uses the engine's default prompt.
@param max_new_tokens  Maximum number of tokens to generate.
*/
CV_EXPORTS_W String inferFile(const Ptr<VLMModel>& model, CV_WRAP_FILE_PATH const String& path,
                              const String& prompt = String(), int max_new_tokens = 512);

//! @}

}} // namespace cv::vlm

#endif // OPENCV_VLM_HPP
