Load Caffe framework models  {#tutorial_dnn_googlenet}
===========================

@tableofcontents

@next_tutorial{tutorial_dnn_halide}

|    |    |
| -: | :- |
| Original author | Vitaliy Lyudvichenko |
| Compatibility | OpenCV >= 3.3 |

Introduction
------------

In this tutorial you will learn how to use opencv_dnn module for image classification by using
GoogLeNet trained network from [Caffe model zoo](http://caffe.berkeleyvision.org/model_zoo.html).

We will demonstrate results of this example on the following picture.
![Buran space shuttle](dnn/images/space_shuttle.jpg)

Source Code
-----------

We will be using snippets from the example application, that can be downloaded [here](https://github.com/opencv/opencv/blob/4.x/samples/dnn/classification.cpp).

@include dnn/classification.cpp

Explanation
-----------

-# Firstly, download GoogLeNet model files:
   [bvlc_googlenet.prototxt  ](https://github.com/opencv/opencv_extra/blob/4.x/testdata/dnn/bvlc_googlenet.prototxt) and
   [bvlc_googlenet.caffemodel](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel)

   Also you need file with names of [ILSVRC2012](http://image-net.org/challenges/LSVRC/2012/browse-synsets) classes:
   [classification_classes_ILSVRC2012.txt](https://github.com/opencv/opencv/blob/4.x/samples/data/dnn/classification_classes_ILSVRC2012.txt).

   Put these files into working dir of this program example.

-# Read and initialize network using path to .prototxt and .caffemodel files
   @snippet dnn/classification.cpp Read and initialize network

   You can skip an argument `framework` if one of the files `model` or `config` has an
   extension `.caffemodel` or `.prototxt`.
   This way function cv::dnn::readNet can automatically detects a model's format.

-# Read input image and convert to the blob, acceptable by GoogleNet
   @snippet dnn/classification.cpp Open a video file or an image file or a camera stream

   cv::VideoCapture can load both images and videos.

   @snippet dnn/classification.cpp Create a 4D blob from a frame
   We convert the image to a 4-dimensional blob (so-called batch) with `1x3x224x224` shape
   after applying necessary pre-processing like resizing and mean subtraction
   `(-104, -117, -123)` for each blue, green and red channels correspondingly using cv::dnn::blobFromImage function.

-# Pass the blob to the network
   @snippet dnn/classification.cpp Set input blob

-# Make forward pass
   @snippet dnn/classification.cpp Make forward pass
   During the forward pass output of each network layer is computed, but in this example we need output from the last layer only.

-# Determine the best class
   @snippet dnn/classification.cpp Get a class with a highest score
   We put the output of network, which contain probabilities for each of 1000 ILSVRC2012 image classes, to the `prob` blob.
   And find the index of element with maximal value in this one. This index corresponds to the class of the image.

-# Run an example from command line
   @code
   ./example_dnn_classification --model=bvlc_googlenet.caffemodel --config=bvlc_googlenet.prototxt --width=224 --height=224 --classes=classification_classes_ILSVRC2012.txt --input=space_shuttle.jpg --mean="104 117 123"
   @endcode
   For our image we get prediction of class `space shuttle` with more than 99% sureness.
