YOLO DNNs  {#tutorial_dnn_yolo}
===============================

@tableofcontents

@prev_tutorial{tutorial_dnn_openvino}
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


Building a YOLO Detection Model using an ONNX Graph
---------------------------------

This guide provides a step-by-step walkthrough for running a YOLO model utilizing an ONNX graph. We focus on the YOLOX model for demonstration purposes.

## Preparing the ONNX Graph

### Generating the ONNX Model
To begin, you need to generate the ONNX graph of a YOLO model. We will use the YOLOX model as an example. Detailed instructions for generating the ONNX model are available in the YOLOX [README](https://dl.opencv.org/models/yolox/README.md).

Additionally, you can access a pre-trained YOLOX small ONNX graph via this [link](https://dl.opencv.org/models/yolox/yolox_s_inf_decoder.onnx).


### Note on Model Conversion

For models other than YOLOX, you can generally follow the standard PyTorch to ONNX conversion process. However, for the YOLOX model, it's crucial to include the generation of anchor points within the ONNX graph. This inclusion simplifies the inference process by eliminating the need to create anchor points manually, as they are already integrated into the ONNX graph.

### Running the YOLOX Model

This section demonstrates two methods for running the YOLOX model. These methods are applicable to any model within the YOLO family.

### Method 1: Building a Custom Pipeline

This method involves constructing the pipeline manually. Instructions for this approach will be detailed in subsequent sections.

- Import required libraries

@code{.cpp}
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "iostream"
@endcode

- Read ONNX graph and create neural network model.

@code{.cpp}
Net net = readNet("<path_to_onnx_file>");
@endcode

- Read image and preprocess it

@code{.cpp}
Size size(640, 640);
Scalar mean(0);
Scalar scale(1.0);
ImagePaddingMode paddingMode = DNN_PMODE_LETTERBOX;
bool swapRB  = true;
bool nmsAcrossClasses = true;
float paddingValue  = 114.0;
float confThreshold = 0.5;
float nmsThreshold  = 0.5;

Image2BlobParams imgParams(
                        scale,
                        size,
                        mean,
                        swapRB,
                        CV_32F,
                        DNN_LAYOUT_NCHW,
                        paddingMode,
                        paddingValue);

Mat img = imread("~/yolox_s_inf_decode.onnx");
Mat inp = blobFromImageWithParams(img, imgParams);
@endcode

- Inference

@code{.cpp}
net.setInput(inp);
std::vector<Mat> outputs;
net.forward(outputs, net.getUnconnectedOutLayersNames());
@endcode

- Post-Processessing

For post-procssing of the model output we will need to use function [`yoloPostProcess`](https://github.com/opencv/opencv/blob/ef8a5eb6207925e8f3055a82e90dbd9b8d10f3e3/modules/dnn/test/test_onnx_importer.cpp#L2650).


@code{.cpp}
std::vector<int> classIds;
std::vector<float> confidences;
std::vector<Rect2d> boxes;

yoloPostProcessing(
    outputs, classIds, confidences, boxes,
    confThreshold, nmsThreshold,
    "yolox");
@endcode

- Draw predicted boxes

@code{.cpp}

std::vector<Rect> boxes;
for (auto box : keep_boxes){
    boxes.push_back(Rect(box.x, box.y, box.width, box.height));
}

Image2BlobParams paramNet;
        paramNet.scalefactor = scale;
        paramNet.size = size;
        paramNet.mean = mean;
        paramNet.swapRB = swapRB;
        paramNet.paddingmode = paddingMode;
        paramNet.blobRectsToImageRects(boxes, boxes, frameSize);

for (size_t idx = 0; idx < boxes.size(); ++idx)
    {
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx],
        box.x, box.y, box.width, box.height, img);
    }

imwrite("image.png", img);
@endcode


### Method 2: Using a Predefined Pipeline from the Command Line
If you prefer a simpler approach without building a custom pipeline, you can run the model directly using command-line instructions.

Steps to Run the Predefined Pipeline:

1. Ensure OpenCV is built on your platform.
2. Navigate to the `build` directory by executing `cd build`.
3. Run the following command:

@code{.cpp}
./bin/example_dnn_yolo_detector --model=<path_to_your_onnx_model> --input=<path_to_your_input_file> --width=<input_width> --height=<input_height> --classes=<path_to_class_names_file> --mns=<minimum_score_threshold> --thr=<confidence_threshold> --mean=<mean_normalization_value> --scale=<scale_factor>
@endcode

- <path_to_your_onnx_model>: Replace with the file path to your ONNX model.
- <path_to_your_input_file>: Replace with the file path to your input image or video.
- <input_width>: Specify the yolo's input width for the model.
- <input_height>: Specify the yolo's input height for the model (for example, 640).
- <path_to_class_names_file>: Replace with the file path to the text file containing class names (e.g., /path/to/yolox_classes.txt).
- <minimum_score_threshold>: Set the minimum score threshold for detection (e.g., 0.5).
- <confidence_threshold>: Set the confidence threshold for detection (e.g., 0.4).
- <mean_normalization_value>: Specify the mean normalization value (e.g., 0 for no mean normalization).
- <scale_factor>: Specify the scale factor for input normalization (e.g., 1.0).
- <padvalue>: Specify the padding value use for filling image after resize operaiton. defaul 114


Questions and suggestions email to: Alessandro de Oliveira Faria cabelo@opensuse.org or OpenCV Team.
