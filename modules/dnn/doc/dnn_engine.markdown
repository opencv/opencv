# DNN Engine Selection {#api_dnn_engine_selection}

## Detailed Description

OpenCV 5 introduces a selectable inference backend for the DNN module, referred to as the *engine*. All engines share the same public API - `cv::dnn::readNet()`, `net.forward()`, and related functions - so switching between them requires changing at most a single argument.

The engine is specified at model-load time and cannot be changed after the model has been loaded, as each engine uses a different internal graph representation.

### ENGINE_AUTO

`ENGINE_AUTO` is the default value for the engine parameter on all `readNet*()` functions. It lets OpenCV pick the engine: it currently resolves to `ENGINE_OPENCV`. The resolution is intentionally an implementation detail and may change as more engines are added, so code that does not need a specific engine should leave the default in place.

Because `ENGINE_AUTO` is the default, existing code that does not pass an engine argument requires no modification.

### ENGINE_OPENCV

`ENGINE_OPENCV` is OpenCV's built-in DNN engine, a ground-up rewrite of the inference graph introduced in OpenCV 5. It is built around a typed operation graph with shape inference, constant folding, and operator fusion, covering approximately 75-80% of the ONNX operator specification. Models that failed to load under OpenCV 4.x due to dynamic shapes or unsupported operators will typically load and run correctly under this engine.

The engine performs automatic attention fusion: the `MatMul` → `Softmax` → `MatMul` subgraph common to transformer architectures is recognised and collapsed into a single fused operation at load time, with no changes required to the model or calling code.

`ENGINE_OPENCV` also introduces native support for Large Language Models and Vision-Language Models. Built-in tokenizers, attention layers, decoding blocks, and KV-caching allow models such as Qwen, Gemma, and PaliGemma to run end-to-end through the standard `Net` API, with no external runtime required.

In OpenCV 5.0, `ENGINE_OPENCV` runs on CPU only. Support for CUDA and other non-CPU backends is planned for a subsequent release. Users requiring GPU acceleration should use `ENGINE_ORT`.

@note The Darknet and Caffe parsers have been removed in OpenCV 5; ONNX is the recommended format. TFLite and TensorFlow models are still supported and are executed via `ENGINE_OPENCV`.

### ENGINE_ORT

`ENGINE_ORT` routes inference through a bundled ONNX Runtime (ORT) wrapper. OpenCV uses its own ONNX parser to construct the ORT graph internally, so only the ORT library is required at runtime, not the standalone onnx package. It applies to ONNX models only.

`ENGINE_ORT` must be enabled at compile time:

```bash
# CPU only
cmake -DWITH_ONNXRUNTIME=ON ..

# With NVIDIA GPU execution providers
cmake -DWITH_ONNXRUNTIME=ON -DDOWNLOAD_ONNXRUNTIME_GPU=ON ..
```

ORT execution providers are supported, including CUDA for NVIDIA hardware. This makes `ENGINE_ORT` the recommended choice for GPU-accelerated inference while native GPU support for `ENGINE_OPENCV` is still in development.

### Selecting an Engine

The engine parameter is accepted by the `readNet*()` family of functions.

@add_toggle_cpp
@code{.cpp}
// ENGINE_AUTO is the default (resolves to ENGINE_OPENCV)
cv::dnn::Net net = cv::dnn::readNetFromONNX("model.onnx");

// Explicitly select OpenCV's built-in engine
cv::dnn::Net net = cv::dnn::readNetFromONNX("model.onnx", cv::dnn::ENGINE_OPENCV);

// Use ONNX Runtime (ONNX models only, requires WITH_ONNXRUNTIME=ON)
cv::dnn::Net net = cv::dnn::readNetFromONNX("model.onnx", cv::dnn::ENGINE_ORT);
@endcode
@end_toggle

@add_toggle_python
@code{.py}
import cv2

# ENGINE_AUTO is the default (resolves to ENGINE_OPENCV)
net = cv2.dnn.readNetFromONNX("model.onnx")

# Explicitly select OpenCV's built-in engine
net = cv2.dnn.readNetFromONNX("model.onnx", engine=cv2.dnn.ENGINE_OPENCV)

# Use ONNX Runtime (ONNX models only, requires WITH_ONNXRUNTIME=ON)
net = cv2.dnn.readNetFromONNX("model.onnx", engine=cv2.dnn.ENGINE_ORT)
@endcode
@end_toggle

The engine cannot be changed after a model has been loaded. To use a different engine, the model must be reloaded with the new engine argument.

### Engine Selection via Environment Variable

The engine can be overridden at the process level using the `OPENCV_FORCE_DNN_ENGINE` environment variable. The integer values correspond directly to the `EngineType` enum: 0 for `ENGINE_AUTO`, 1 for `ENGINE_OPENCV`, and 2 for `ENGINE_ORT`. Leaving the variable unset (or set to `ENGINE_AUTO`) does not force any engine, so the value passed to `readNet*()` is honored.

```bash
# Linux / macOS
OPENCV_FORCE_DNN_ENGINE=1 python3 inference.py

# Windows
set OPENCV_FORCE_DNN_ENGINE=1
python inference.py
```
