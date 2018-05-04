YOLO DNNs  {#tutorial_dnn_yolo}
===============================

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
[in C++](https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.cpp) and
[in Python](https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.py) languages

Usage examples
--------------

Execute in webcam:

@code{.bash}

$ object_detection --config=[PATH-TO-DARKNET]/cfg/yolov3.cfg --model=[PATH-TO-DARKNET]/yolov3.weights --classes=[PATH-TO-DARKNET]/data/coco.names --width=416 --height=416 --scale=0.00392 --target=0 

@endcode

Execute with image or video file:

@code{.bash}

$ object_detection --config=[PATH-TO-DARKNET]/cfg/yolov3.cfg --model=[PATH-TO-DARKNET]/yolov3.weights --classes=[PATH-TO-DARKNET]/data/coco.names --width=416 --height=416 --scale=0.00392 --target=0 --rgb --input=[PATH-TO-IMAGE-OR-VIDEO-FILE]

@endcode

Questions and suggestions email to: Alessandro de Oliveira Faria cabelo@opensuse.org or OpenCV Team.
