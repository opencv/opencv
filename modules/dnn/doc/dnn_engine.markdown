# DNN Engine Selection {#api_dnn_engine_selection}

## Detailed Description

OpenCV 5 introduces a selectable inference backend for the DNN module, referred to as the *engine*. All engines share the same public API - `cv::dnn::readNet()`, `net.forward()`, and related functions - so switching between them requires changing at most a single argument.

The engine is specified at model-load time and cannot be changed after the model has been loaded, as each engine uses a different internal graph representation.

### ENGINE_NEW

`ENGINE_NEW` is a ground-up rewrite of the DNN inference graph, introduced in OpenCV 5. It is built around a typed operation graph with shape inference, constant folding, and operator fusion, covering approximately 75-80% of the ONNX operator specification. Models that failed to load under OpenCV 4.x due to dynamic shapes or unsupported operators will typically load and run correctly under this engine.

The engine performs automatic attention fusion: the `MatMul` → `Softmax` → `MatMul` subgraph common to transformer architectures is recognised and collapsed into a single fused operation at load time, with no changes required to the model or calling code.

`ENGINE_NEW` also introduces native support for Large Language Models and Vision-Language Models. Built-in tokenizers, attention layers, decoding blocks, and KV-caching allow models such as Qwen, Gemma, and PaliGemma to run end-to-end through the standard `Net` API, with no external runtime required.

In OpenCV 5.0, `ENGINE_NEW` runs on CPU only. Support for CUDA and other non-CPU backends is planned for a subsequent release. Users requiring GPU acceleration should use `ENGINE_CLASSIC` or `ENGINE_ORT`.

### ENGINE_CLASSIC

`ENGINE_CLASSIC` is the inference engine carried over from OpenCV 4.x. It supports the full set of DNN backends and hardware targets, including `DNN_BACKEND_CUDA` for NVIDIA GPUs, `DNN_BACKEND_OPENVINO` for Intel hardware, and FP16 inference targets. Its ONNX operator coverage is approximately 22% of the specification. Models with dynamic shapes or transformer-style subgraphs will generally not load under this engine.

@note The Darknet and Caffe parsers have been removed in OpenCV 5; ONNX is the recommended format for all engines. TFLite models are still supported and currently executed via `ENGINE_CLASSIC`.

### ENGINE_AUTO

`ENGINE_AUTO` is the default value for the engine parameter on all `readNet*()` functions. When active, OpenCV first attempts to load the model with `ENGINE_NEW`. If the model cannot be loaded - for example, because it uses an operator not yet implemented in the new engine - the load is automatically retried with `ENGINE_CLASSIC`.

Because `ENGINE_AUTO` is the default, existing code that does not pass an engine argument requires no modification.

### ENGINE_ORT

`ENGINE_ORT` routes inference through a bundled ONNX Runtime (ORT) wrapper. OpenCV uses its own ONNX parser to construct the ORT graph internally, so only the ORT library is required at runtime, not the standalone onnx package.

`ENGINE_ORT` must be enabled at compile time:

```bash
# CPU only
cmake -DWITH_ONNXRUNTIME=ON ..

# With NVIDIA GPU execution providers
cmake -DWITH_ONNXRUNTIME=ON -DDOWNLOAD_ONNXRUNTIME_GPU=ON ..
```

ORT execution providers are supported, including CUDA for NVIDIA hardware. This makes `ENGINE_ORT` the recommended choice for GPU-accelerated inference on models that `ENGINE_CLASSIC` cannot load, while native GPU support for `ENGINE_NEW` is still in development.

### Selecting an Engine

The engine parameter is accepted by the `readNet*()` family of functions.

@add_toggle_cpp
@code{.cpp}
// ENGINE_AUTO is the default
cv::dnn::Net net = cv::dnn::readNetFromONNX("model.onnx");

// Force the new engine
cv::dnn::Net net = cv::dnn::readNetFromONNX("model.onnx", cv::dnn::ENGINE_NEW);

// Force the classic engine with CUDA backend
cv::dnn::Net net = cv::dnn::readNetFromONNX("model.onnx", cv::dnn::ENGINE_CLASSIC);
net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

// Use ONNX Runtime
cv::dnn::Net net = cv::dnn::readNetFromONNX("model.onnx", cv::dnn::ENGINE_ORT);
@endcode
@end_toggle

@add_toggle_python
@code{.py}
import cv2

# ENGINE_AUTO is the default
net = cv2.dnn.readNetFromONNX("model.onnx")

# Force the new engine
net = cv2.dnn.readNetFromONNX("model.onnx", engine=cv2.dnn.ENGINE_NEW)

# Force the classic engine with CUDA backend
net = cv2.dnn.readNetFromONNX("model.onnx", engine=cv2.dnn.ENGINE_CLASSIC)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Use ONNX Runtime
net = cv2.dnn.readNetFromONNX("model.onnx", engine=cv2.dnn.ENGINE_ORT)
@endcode
@end_toggle

The engine cannot be changed after a model has been loaded. To use a different engine, the model must be reloaded with the new engine argument.

### Engine Selection via Environment Variable

The engine can be overridden at the process level using the `OPENCV_FORCE_DNN_ENGINE` environment variable. The integer values correspond directly to the `EngineType` enum: 1 for `ENGINE_CLASSIC`, 2 for `ENGINE_NEW`, 3 for `ENGINE_AUTO`, and 4 for `ENGINE_ORT`.

```bash
# Linux / macOS
OPENCV_FORCE_DNN_ENGINE=1 python3 inference.py

# Windows
set OPENCV_FORCE_DNN_ENGINE=1
python inference.py
```
