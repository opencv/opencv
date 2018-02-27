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

The latest version of sample source code can be downloaded [here](https://github.com/opencv/opencv/blob/master/samples/dnn/yolo_object_detection.cpp).

@include dnn/yolo_object_detection.cpp

How to compile in command line with pkg-config
----------------------------------------------

@code{.bash}

# g++ `pkg-config --cflags opencv` `pkg-config --libs opencv` yolo_object_detection.cpp -o yolo_object_detection

@endcode

Execute in webcam:

@code{.bash}

$ yolo_object_detection -camera_device=0  -cfg=[PATH-TO-DARKNET]/cfg/yolo.cfg -model=[PATH-TO-DARKNET]/yolo.weights   -class_names=[PATH-TO-DARKNET]/data/coco.names

@endcode

Execute with image:

@code{.bash}

$ yolo_object_detection -source=[PATH-IMAGE]  -cfg=[PATH-TO-DARKNET]/cfg/yolo.cfg -model=[PATH-TO-DARKNET]/yolo.weights   -class_names=[PATH-TO-DARKNET]/data/coco.names

@endcode

Execute in video file:

@code{.bash}

$ yolo_object_detection -source=[PATH-TO-VIDEO] -cfg=[PATH-TO-DARKNET]/cfg/yolo.cfg -model=[PATH-TO-DARKNET]/yolo.weights   -class_names=[PATH-TO-DARKNET]/data/coco.names

@endcode

Questions and suggestions email to: Alessandro de Oliveira Faria cabelo@opensuse.org or OpenCV Team.
