Deep Neural Networks (dnn module) {#tutorial_table_of_content_dnn}
=====================================

@tableofcontents

-   @subpage tutorial_dnn_googlenet
-   @subpage tutorial_dnn_halide
-   @subpage tutorial_dnn_halide_scheduling
-   @subpage tutorial_dnn_openvino
-   @subpage tutorial_dnn_yolo
-   @subpage tutorial_dnn_javascript
-   @subpage tutorial_dnn_custom_layers
-   @subpage tutorial_dnn_OCR
-   @subpage tutorial_dnn_text_spotting
-   @subpage tutorial_dnn_face

### Backend selection and fallback behavior

When using the DNN module, selecting a preferred backend (for example, OpenVINO,
CUDA, or the default OpenCV backend) represents a request rather than a strict
guarantee. If the selected backend cannot support certain layers, model
configurations, or execution requirements, OpenCV may automatically fall back to
another backend to ensure successful execution.

This fallback may occur without an explicit error or warning. Users who rely on a
specific backend for performance evaluation or benchmarking are encouraged to
verify which backend is actually used at runtime.

#### PyTorch models with OpenCV
In this section you will find the guides, which describe how to run classification, segmentation and detection PyTorch DNN models with OpenCV.
-   @subpage pytorch_cls_tutorial_dnn_conversion
-   @subpage pytorch_cls_c_tutorial_dnn_conversion
-   @subpage pytorch_segm_tutorial_dnn_conversion

#### TensorFlow models with OpenCV
In this section you will find the guides, which describe how to run classification, segmentation and detection TensorFlow DNN models with OpenCV.
-   @subpage tf_cls_tutorial_dnn_conversion
-   @subpage tf_det_tutorial_dnn_conversion
-   @subpage tf_segm_tutorial_dnn_conversion
