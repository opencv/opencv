YOLO DNNs  {#tutorial_dnn_yolo}
===============================

@tableofcontents

@prev_tutorial{tutorial_dnn_android}
@next_tutorial{tutorial_dnn_javascript}

|    |    |
| -: | :- |
| Original author | Alessandro de Oliveira Faria |
| Compatibility | OpenCV >= 3.3.1 |

Introduction
------------

In this text you will learn how to use opencv_dnn module using yolo_object_detection (Sample of using OpenCV dnn module in real time with device capture, video and image).

We will demonstrate results of this example on the following picture.
![Picture example](images/yolo.jpg)

Examples
--------

VIDEO DEMO:
@youtube{NHtRlndE2cg}

Source Code
-----------

Use a universal sample for object detection models written
[in C++](https://github.com/opencv/opencv/blob/4.x/samples/dnn/object_detection.cpp) and
[in Python](https://github.com/opencv/opencv/blob/4.x/samples/dnn/object_detection.py) languages

Usage examples
--------------

Execute in webcam:

@code{.bash}

$ example_dnn_object_detection --config=[PATH-TO-DARKNET]/cfg/yolo.cfg --model=[PATH-TO-DARKNET]/yolo.weights --classes=object_detection_classes_pascal_voc.txt --width=416 --height=416 --scale=0.00392 --rgb

@endcode

Execute with image or video file:

@code{.bash}

$ example_dnn_object_detection --config=[PATH-TO-DARKNET]/cfg/yolo.cfg --model=[PATH-TO-DARKNET]/yolo.weights --classes=object_detection_classes_pascal_voc.txt --width=416 --height=416 --scale=0.00392 --input=[PATH-TO-IMAGE-OR-VIDEO-FILE] --rgb

@endcode

Questions and suggestions email to: Alessandro de Oliveira Faria cabelo@opensuse.org or OpenCV Team.
