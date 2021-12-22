# How to run custom OCR model {#tutorial_dnn_OCR}

@tableofcontents

@prev_tutorial{tutorial_dnn_custom_layers}
@next_tutorial{tutorial_dnn_text_spotting}

|    |    |
| -: | :- |
| Original author | Zihao Mu |
| Compatibility | OpenCV >= 4.3 |

## Introduction

In this tutorial, we first introduce how to obtain the custom OCR model, then how to transform your own OCR models so that they can be run correctly by the opencv_dnn module. and finally we will provide some pre-trained models.

## Train your own OCR model

[This repository](https://github.com/zihaomu/deep-text-recognition-benchmark) is a good start point for training your own OCR model. In repository, the MJSynth+SynthText was set as training set by default. In addition, you can configure the model structure and data set you want.

## Transform OCR model to ONNX format and Use it in OpenCV DNN

After completing the model training, please use [transform_to_onnx.py](https://github.com/zihaomu/deep-text-recognition-benchmark/blob/master/transform_to_onnx.py) to convert the model into onnx format.

#### Execute in webcam
The Python version example code can be found at [here](https://github.com/opencv/opencv/blob/4.x/samples/dnn/text_detection.py).

Example:
@code{.bash}
$ text_detection -m=[path_to_text_detect_model] -ocr=[path_to_text_recognition_model]
@endcode

## Pre-trained ONNX models are provided

Some pre-trained models can be found at https://drive.google.com/drive/folders/1cTbQ3nuZG-EKWak6emD_s8_hHXWz7lAr?usp=sharing.

Their performance at different text recognition datasets is shown in the table below:

| Model name           | IIIT5k(%) | SVT(%) | ICDAR03(%) | ICDAR13(%) | ICDAR15(%) | SVTP(%) | CUTE80(%) | average acc (%) | parameter( x10^6 ) |
| -------------------- | --------- | ------ | ---------- | ---------- | ---------- | ------- | --------- | --------------- | ------------------ |
| DenseNet-CTC         | 72.267    | 67.39  | 82.81     | 80         | 48.38     | 49.45  | 42.50    | 63.26       | 0.24              |
| DenseNet-BiLSTM-CTC  | 73.76    | 72.33 | 86.15     | 83.15     | 50.67     | 57.984  | 49.826    | 67.69       | 3.63              |
| VGG-CTC              | 75.96    | 75.42 | 85.92     | 83.54     | 54.89     | 57.52  | 50.17    | 69.06       | 5.57              |
| CRNN_VGG-BiLSTM-CTC | 82.63    | 82.07 | 92.96     | 88.867     | 66.28     | 71.01  | 62.37    | 78.03       | 8.45              |
| ResNet-CTC           | 84.00        | 84.08  | 92.39     | 88.96     | 67.74     | 74.73  | 67.60    | 79.93    | 44.28             |

The performance of the text recognition model were tesred on OpenCV DNN, and does not include the text detection model.

#### Model selection suggestion:
The input of text recognition model is the output of the text detection model, which causes the performance of text detection to greatly affect the performance of text recognition.

DenseNet_CTC has the smallest parameters and best FPS, and it is suitable for edge devices, which are very sensitive to the cost of calculation. If you have limited computing resources and want to achieve better accuracy, VGG_CTC is a good choice.

CRNN_VGG_BiLSTM_CTC is suitable for scenarios that require high recognition accuracy.
