# VLM Model Directory Layout {#vlm_model_dir_layout}

## Detailed Description

For the local model types — `cv::vlm::VLM_MODEL_GRANITE_DOCLING` and
`cv::vlm::VLM_MODEL_PADDLEOCR_VL` — the `model_dir` argument of `cv::vlm::create()` is a
directory holding an ONNX export of the model plus the JSON configuration that describes how
to tokenize text and preprocess the image. Cloud model types ignore `model_dir` as a path and
use it as the provider's model name instead.

A stock export downloaded from HuggingFace is **not** usable as-is: it has no OpenCV tokenizer
config. See *The config.json file* below, which is the one file you have to add.

### Granite-Docling-258M

Upstream export: https://huggingface.co/onnx-community/granite-docling-258M-ONNX

    <model_dir>/
      config.json               OpenCV tokenizer config, plus image_token_id and eos_token_id
      preprocessor_config.json  image_mean, image_std, size.longest_edge,
                                max_image_size.longest_edge
      processor_config.json     image_seq_len
      tokenizer.json            vocabulary and merges
      onnx/
        vision_encoder.onnx
        embed_tokens.onnx
        decoder_model_merged.onnx

### PaddleOCR-VL

Upstream export: https://huggingface.co/PaddlePaddle/PaddleOCR-VL

    <model_dir>/
      config.json               OpenCV tokenizer config, plus image_token_id and eos_token_id
      processor_config.json     patch size, merge size and image token placeholder
      tokenizer.json            vocabulary and merges
      onnx/
        vision_encoder.onnx
        embedding.onnx
        decoder.onnx

Note that the two models name their ONNX files differently, and that PaddleOCR-VL reads no
`preprocessor_config.json` — its normalization constants come from `processor_config.json`.

### The config.json file

This is the file that differs from the upstream export, and the usual reason a first attempt
fails. It is **not** HuggingFace's `config.json`, and it is not their `tokenizer_config.json`
either. It is the descriptor `cv::dnn::Tokenizer::load()` consumes, with two extra fields the
VLM engines read directly:

    {
      "model_type":     "...",   read by cv::dnn::Tokenizer::load()
      "method":         "...",   read by cv::dnn::Tokenizer::load()
      "vocab_size":     ...,
      "tokenizer_class": "...",
      "eos_token":      "...",
      "bos_token":      "...",
      "pad_token":      "...",
      "image_token_id": ...,     the token the vision features are scattered into
      "eos_token_id":   ...      where generation stops
    }

`image_token_id` and `eos_token_id` may instead sit under a `text_config` object, which is
where a HuggingFace config keeps them; both placements are accepted.

The samples `samples/dnn/granite_docling_inference.py` and
`samples/dnn/paddleocr_vl_inference.py` run the same pipeline directly on `cv::dnn`, without
this module, and consume an identically laid out directory. A directory that works with one
works with the other.

### Engine and device

Only `engine = "opencv"` (`cv::dnn::ENGINE_OPENCV`) is supported for local models. These
decoders need `cv::dnn::Net::enableKVCache()`, which the ONNX Runtime path does not implement,
so any other value is rejected rather than silently ignored.
