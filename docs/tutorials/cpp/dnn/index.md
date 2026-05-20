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
* - [How to run deep networks in browser](https://docs.opencv.org/5.x/d5/d86/tutorial_dnn_javascript.html)
  - Use the OpenCV.js `cv.dnn` module to run inference in a browser.
* - [Custom deep learning layers support](https://docs.opencv.org/5.x/dc/db1/tutorial_dnn_custom_layers.html)
  - Add custom layers to OpenCV's DNN module to extend support for new model architectures.
* - [How to run custom OCR model](https://docs.opencv.org/5.x/d9/d1e/tutorial_dnn_OCR.html)
  - Train and run a custom OCR model with OpenCV's `cv::dnn::TextRecognitionModel`.
* - [High Level API: TextDetectionModel and TextRecognitionModel](https://docs.opencv.org/5.x/d4/d43/tutorial_dnn_text_spotting.html)
  - End-to-end text detection + recognition pipeline using `TextDetectionModel` and `TextRecognitionModel`.
* - [DNN-based Face Detection And Recognition](https://docs.opencv.org/5.x/d0/dd4/tutorial_dnn_face.html)
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

* - [Conversion of PyTorch Classification Models and Launch with OpenCV Python](https://docs.opencv.org/5.x/dc/d70/pytorch_cls_tutorial_dnn_conversion.html)
  - Export a PyTorch classification model to ONNX and run inference via OpenCV's Python `dnn` API.
* - [Conversion of PyTorch Classification Models and Launch with OpenCV C++](https://docs.opencv.org/5.x/dd/d55/pytorch_cls_c_tutorial_dnn_conversion.html)
  - Same as above for the C++ `dnn` API.
* - [Conversion of PyTorch Segmentation Models and Launch with OpenCV](https://docs.opencv.org/5.x/d7/d9a/pytorch_segm_tutorial_dnn_conversion.html)
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

* - [Conversion of TensorFlow Classification Models and Launch with OpenCV Python](https://docs.opencv.org/5.x/d1/d8f/tf_cls_tutorial_dnn_conversion.html)
  - Convert a TensorFlow classification model to a frozen graph and load it with OpenCV's `dnn` module.
* - [Conversion of TensorFlow Detection Models and Launch with OpenCV Python](https://docs.opencv.org/5.x/d4/d2f/tf_det_tutorial_dnn_conversion.html)
  - Convert a TensorFlow detection model (SSD / Faster R-CNN / Mask R-CNN) and run it through OpenCV.
* - [Conversion of TensorFlow Segmentation Models and Launch with OpenCV](https://docs.opencv.org/5.x/dc/db4/tf_segm_tutorial_dnn_conversion.html)
  - Convert a TensorFlow semantic-segmentation model and infer it via OpenCV's `dnn` module.
```

```{toctree}
:hidden:
:maxdepth: 1

Load ONNX framework models <dnn_googlenet>
OpenCV usage with OpenVINO <dnn_openvino>
YOLO DNNs <dnn_yolo>
How to run deep networks in browser <https://docs.opencv.org/5.x/d5/d86/tutorial_dnn_javascript.html>
Custom deep learning layers support <https://docs.opencv.org/5.x/dc/db1/tutorial_dnn_custom_layers.html>
How to run custom OCR model <https://docs.opencv.org/5.x/d9/d1e/tutorial_dnn_OCR.html>
High Level API: TextDetectionModel and TextRecognitionModel <https://docs.opencv.org/5.x/d4/d43/tutorial_dnn_text_spotting.html>
DNN-based Face Detection And Recognition <https://docs.opencv.org/5.x/d0/dd4/tutorial_dnn_face.html>
Conversion of PyTorch Classification Models (Python) <https://docs.opencv.org/5.x/dc/d70/pytorch_cls_tutorial_dnn_conversion.html>
Conversion of PyTorch Classification Models (C++) <https://docs.opencv.org/5.x/dd/d55/pytorch_cls_c_tutorial_dnn_conversion.html>
Conversion of PyTorch Segmentation Models <https://docs.opencv.org/5.x/d7/d9a/pytorch_segm_tutorial_dnn_conversion.html>
Conversion of TensorFlow Classification Models <https://docs.opencv.org/5.x/d1/d8f/tf_cls_tutorial_dnn_conversion.html>
Conversion of TensorFlow Detection Models <https://docs.opencv.org/5.x/d4/d2f/tf_det_tutorial_dnn_conversion.html>
Conversion of TensorFlow Segmentation Models <https://docs.opencv.org/5.x/dc/db4/tf_segm_tutorial_dnn_conversion.html>
```
