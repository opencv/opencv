# Deep Neural Networks (dnn module)

```{list-table}
:class: opencv-module-table
:widths: 35 65
:header-rows: 1

* - Topic
  - Description

* - [Load ONNX framework models](dnn_googlenet.md)
  - In this tutorial you will learn how to use opencv_dnn module for image classification by using GoogLeNet trained network from ONNX model zoo.
* - [OpenCV usage with OpenVINO](dnn_openvino.md)
  - This tutorial provides OpenCV installation guidelines for how to use OpenCV with OpenVINO.
* - [YOLO DNNs](dnn_yolo.md)
  - Running pre-trained YOLO networks (YOLOv3 / YOLOv4 / YOLOX) through the OpenCV `dnn` module — preprocessing, forward pass, and post-processing of detections.
* - [How to run deep networks in browser](dnn_javascript.md)
  - Use the OpenCV.js `cv.dnn` module to run inference in a browser.
* - [Custom deep learning layers support](dnn_custom_layers.md)
  - Add custom layers to OpenCV's DNN module to extend support for new model architectures.
* - [How to run custom OCR model](dnn_OCR.md)
  - Train and run a custom OCR model with OpenCV's `cv::dnn::TextRecognitionModel`.
* - [High Level API: TextDetectionModel and TextRecognitionModel](dnn_text_spotting.md)
  - End-to-end text detection + recognition pipeline using `TextDetectionModel` and `TextRecognitionModel`.
* - [DNN-based Face Detection And Recognition](dnn_face.md)
  - Face detection and identification using `FaceDetectorYN` and `FaceRecognizerSF`.
```

## PyTorch models with OpenCV

In this section you will find the guides, which describe how to run classification, segmentation and detection PyTorch DNN models with OpenCV.

```{list-table}
:class: opencv-module-table
:widths: 35 65
:header-rows: 1

* - Topic
  - Description

* - [Conversion of PyTorch Classification Models and Launch with OpenCV Python](pytorch_cls_model_conversion_tutorial.md)
  - Export a PyTorch classification model to ONNX and run inference via OpenCV's Python `dnn` API.
* - [Conversion of PyTorch Classification Models and Launch with OpenCV C++](pytorch_cls_model_conversion_c_tutorial.md)
  - Same as above for the C++ `dnn` API.
* - [Conversion of PyTorch Segmentation Models and Launch with OpenCV](pytorch_sem_segm_model_conversion_tutorial.md)
  - Export a PyTorch semantic-segmentation model to ONNX and run it through OpenCV's `dnn` module.
```

## TensorFlow models with OpenCV

In this section you will find the guides, which describe how to run classification, segmentation and detection TensorFlow DNN models with OpenCV.

```{list-table}
:class: opencv-module-table
:widths: 35 65
:header-rows: 1

* - Topic
  - Description

* - [Conversion of TensorFlow Classification Models and Launch with OpenCV Python](tf_cls_model_conversion_tutorial.md)
  - Convert a TensorFlow classification model to a frozen graph and load it with OpenCV's `dnn` module.
* - [Conversion of TensorFlow Detection Models and Launch with OpenCV Python](tf_det_model_conversion_tutorial.md)
  - Convert a TensorFlow detection model (SSD / Faster R-CNN / Mask R-CNN) and run it through OpenCV.
* - [Conversion of TensorFlow Segmentation Models and Launch with OpenCV](tf_sem_segm_model_conversion_tutorial.md)
  - Convert a TensorFlow semantic-segmentation model and infer it via OpenCV's `dnn` module.
```

```{toctree}
:hidden:
:maxdepth: 1

Load ONNX framework models <dnn_googlenet>
OpenCV usage with OpenVINO <dnn_openvino>
YOLO DNNs <dnn_yolo>
How to run deep networks in browser <dnn_javascript>
Custom deep learning layers support <dnn_custom_layers>
How to run custom OCR model <dnn_OCR>
High Level API: TextDetectionModel and TextRecognitionModel <dnn_text_spotting>
DNN-based Face Detection And Recognition <dnn_face>
Conversion of PyTorch Classification Models (Python) <pytorch_cls_model_conversion_tutorial>
Conversion of PyTorch Classification Models (C++) <pytorch_cls_model_conversion_c_tutorial>
Conversion of PyTorch Segmentation Models <pytorch_sem_segm_model_conversion_tutorial>
Conversion of TensorFlow Classification Models <tf_cls_model_conversion_tutorial>
Conversion of TensorFlow Detection Models <tf_det_model_conversion_tutorial>
Conversion of TensorFlow Segmentation Models <tf_sem_segm_model_conversion_tutorial>
```
