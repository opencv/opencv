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


Running pre-trained YOLO model in OpenCV
----------------------------------------

Deploying pre-trained models is a common task in machine learning, particularly when working with hardware that does not support certain frameworks like PyTorch. This guide provides a comprehensive overview of exporting pre-trained YOLO family models from PyTorch and deploying them using OpenCV's runtime framework. For demonstration purposes, we will focus on the [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX/blob/main) model, but the methodology applies to other supported<sup>*</sup> models.

<sup>*</sup> Currently, OpenCV supports the following YOLO models: [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX/blob/main), [YoloNas](https://github.com/Deci-AI/super-gradients/tree/master), [YOLOv8](https://github.com/ultralytics/ultralytics/tree/main), [YOLOv7](https://github.com/WongKinYiu/yolov7/tree/main), [YOLOv6](https://github.com/meituan/YOLOv6/blob/main), [YOLOv5](https://github.com/ultralytics/yolov5), and [YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4). This support includes pre and post-processing routines specific to these models. While other older models of yolo are also supported by OpenCV, they are out of the scope of this tutorial.


Assuming that we have successfuly trained YOLOX model, the subsequent step involves exporting and running this model in OpenCV environment. There are several critical considerations to address before proceeding with this process. Let's delve into these aspects.

### YOLO's Preproccessing & Output
Understanding the nature of inputs and outputs associated with YOLO family detectors is pivotal. These detectors, akin to most Deep Neural Networks (DNN), typically exhibit variation in input sizes contingent upon the model's scale.

| Model Scale  | Input Size   |
|--------------|--------------|
| Small Models <sup>[1](https://github.com/Megvii-BaseDetection/YOLOX/tree/main#standard-models)</sup>| 416x416      |
| Midsize Models <sup>[2](https://github.com/Megvii-BaseDetection/YOLOX/tree/main#standard-models)</sup>| 640x640    |
| Large Models <sup>[3](https://github.com/meituan/YOLOv6/tree/main#benchmark)</sup>| 1280x1280    |

This table provides a quick reference to understand the different input dimensions commonly used in various YOLO models inputs. These are standart input shapes. Make sure you use input size that you trained model with, if it is differed from from the size mentioned in the table.

The next critical element in the process involves understanding the specifics of image preprocessing for YOLO detectors. While the fundamental preprocessing approach remains consistent across the YOLO family, there are subtle yet crucial differences that must be accounted for to avoid any degradation in performance. Key among these are the `resize type` and the `padding value` applied post-resize. For instance, the [YOLOX model](https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/yolox/data/data_augment.py#L142) utilizes a `LetterBox` resize method and a padding value of `114.0`. It is imperative to ensure that these parameters, along with the normalization constants, are appropriately matched to the model being exported.

Regarding the model's output, it typically takes the form of a tensor with dimensions [BxNxC+5] or [BxNxC+4], where 'B' represents the batch size, 'N' denotes the number of anchors, and 'C' signifies the number of classes (for instance, 80 classes if the model is trained on the COCO dataset). The additional 5 in the former tensor structure corresponds to the objectness score (obj), confidence score (conf), and the bounding box coordinates (cx, cy, w, h). Notably, the `yolov8` model's output is shaped as [BxNxC+4], where there is no explicit objectness score, and the object score is directly inferred from the class score. For the `yolox` model, specifically, it is also necessary to incorporate anchor points to rescale predictions back to the image domain. This step will be integrated into the ONNX graph, a process that we will detail further in the subsequent sections.


### PyTorch Model Export
Now that we know know the parameters of the preprecessing we can go on and export the model from Pytorch to ONNX graph. Since in this tutorial we are using Yolox as our sample model, lets use its export for demonstration purposes (the proccess is  identical for the rest of the yolo detectors). To exporting yolox we can just use [export script](https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/tools/export_onnx.py). Particularly we need following commands:

@code{.bash}
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth # download pre-trained weights
python3 -m tools.export_onnx --output-name yolox_s.onnx -n yolox-s -c yolox_s.pth --decode_in_inference
@endcode

**NOTE:** Here `--decode_in_inference` is to incude anchor box creation in the ONNX graph itself. It sets [this value](https://github.com/Megvii-BaseDetection/YOLOX/blob/ac58e0a5e68e57454b7b9ac822aced493b553c53/yolox/models/yolo_head.py#L210C16-L210C39) to `True`,
which subsequently includes anchor generation function.

Below we demonstared the minimal version of the export script (which could be used for models other than yolox) in case it is needed. However, usually each yolo repository has predefined export script.

@code{.py}
    import onnx
    import torch
    from onnxsim import simplify

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt)

    # prepare dummy input
    dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])

    #export the model
    torch.onnx._export(
        model,
        dummy_input,
        "yolox.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: 'batch'},
                      "output": {0: 'batch'}})

    # use onnx-simplifier to reduce reduent model.
    onnx_model = onnx.load(args.output_name)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, args.output_name)
@endcode

### Running Yolo ONNX detector with OpenCV Sample
Once we have our ONNX graph of the model, we just simply can run with OpenCV's sample. To that we need to make sure:

1. OpenCV is build with -DBUILD_EXAMLES=ON flag.
2. Navigate to the OpenCV's `build` directory
3. Run the following command:

@code{.cpp}
./bin/example_dnn_yolo_detector --input=<path_to_your_input_file> \
                                --classes=<path_to_class_names_file> \
                                --thr=<confidence_threshold> \
                                --nms=<non_maximum_suppression_threshold> \
                                --mean=<mean_normalization_value> \
                                --scale=<scale_factor> \
                                --yolo=<yolo_model_version> \
                                --padvalue=<padding_value> \
                                --paddingmode=<padding_mode> \
                                --backend=<computation_backend> \
                                --target=<target_computation_device>
@endcode

- --input: File path to your input image or video. If omitted, it will capture frames from a camera.
- --classes: File path to a text file containing class names for object detection.
- --thr: Confidence threshold for detection (e.g., 0.5).
- --nms: Non-maximum suppression threshold (e.g., 0.4).
- --mean: Mean normalization value (e.g., 0.0 for no mean normalization).
- --scale: Scale factor for input normalization (e.g., 1.0).
- --yolo: YOLO model version (e.g., YOLOv3, YOLOv4, etc.).
- --padvalue: Padding value used in preprocessing (e.g., 114.0).
- --paddingmode: Method for handling image resizing and padding. Options: 0 (resize without extra processing), 1 (crop after resize), 2 (resize with aspect ratio preservation).
- --backend: Selection of computation backend (0 for automatic, 1 for Halide, 2 for OpenVINO, etc.).
- --target: Selection of target computation device (0 for CPU, 1 for OpenCL, etc.).
- --device: Camera device number (0 for default camera). If --input is not provided camera with index 0 will used by default.

Here `mean`, `scale`, `padvalue`, `paddingmode` should exactly match those that we dicussed in preprocessing section in oder for the model to match result in PyTorch

### Building a Custom Pipeline

Sometimes there is a need to make some custom adjustments in the inference pipeline. With OpenCV's framework this is also quite easy to achieve. Below we will outline simple starter pipeline that does the same thing as CLI cammand. You can adjust it according to your needs.

- Import required libraries

@snippet samples/dnn/yolo_detector.cpp includes

- Read ONNX graph and create neural network model.

@snippet samples/dnn/yolo_detector.cpp read_net

- Read image and preprocess it

@snippet samples/dnn/yolo_detector.cpp preprocess_params
@snippet samples/dnn/yolo_detector.cpp preprocess_call
@snippet samples/dnn/yolo_detector.cpp preprocess_call_func

- Inference

@snippet samples/dnn/yolo_detector.cpp forward_buffers
@snippet samples/dnn/yolo_detector.cpp forward


- Post-Processessing

For post-procssing of the model output we will need to use function [`yoloPostProcess`](https://github.com/opencv/opencv/blob/ef8a5eb6207925e8f3055a82e90dbd9b8d10f3e3/modules/dnn/test/test_onnx_importer.cpp#L2650).

@snippet samples/dnn/yolo_detector.cpp postprocess

- Draw predicted boxes

@snippet samples/dnn/yolo_detector.cpp draw_boxes

Questions and suggestions email to: Alessandro de Oliveira Faria cabelo@opensuse.org or OpenCV Team.
