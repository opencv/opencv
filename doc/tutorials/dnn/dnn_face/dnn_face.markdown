# DNN-based Face Detection And Recognition

@tableofcontents

@prev_tutorial{tutorial_dnn_text_spotting}
@next_tutorial{pytorch_cls_tutorial_dnn_conversion}

| | |
| -: | :- |
| Original Author | Chengrui Wang, Yuantao Feng |
| Compatibility | OpenCV >= 4.5 |

## Introduction

In this section, we introduce the DNN-based module for face detection and face recognition. Models can be obtained in [Models](#Models). The usage of `DNNFaceDetector` and `DNNFaceRecognizer` are presented in [Usage](#Usage).

## Models

There are two models (ONNX format) pre-trained and required for this module:
- [Face Detection](https://github.com/ShiqiYu/libfacedetection.train/tree/master/tasks/task1/onnx):
    - Size: 337KB
    - Results on WIDER Face Val set: 0.830(easy), 0.824(medium), 0.708(hard)
- [Face Recognition](https://drive.google.com/file/d/1ClK9WiB492c5OZFKveF3XiHCejoOxINW/view?usp=sharing)
    - Size: 36.9MB
    - Results:

    | Database | Accuracy | Threshold (normL2) | Threshold (cosine) |
    | -------- | -------- | ------------------ | ------------------ |
    | LFW      | 99.60%   | 1.272              | 0.363              |
    | CALFW    | 93.95%   | 1.320              | 0.340              |
    | CPLFW    | 91.05%   | 1.450              | 0.275              |
    | AgeDB-30 | 94.90%   | 1.446              | 0.277              |
    | CFP-FP   | 94.80%   | 1.571              | 0.212              |

## Usage

### DNNFaceDetector

```cpp
// Initialize DNNFaceDetector
Ptr<DNNFaceDetector> faceDetector = DNNFaceDetector::create(onnx_path, image.size(), score_thresh, nms_thresh, top_k);

// Forward
Mat faces;
faceDetector->detect(image, faces);
```

The detection output `faces` is a two-dimension array, whose rows are the detected face instances, columns are the location of a face and 5 facial landmarks. The format of each row is as follows:

```
x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm
```
, where `x1, y1, w, h` are the top-left coordinates, width and height of the face bounding box, `{x, y}_{re, le, nt, rcm, lcm}` stands for the coordinates of right eye, left eye, nose tip, the right corner and left corner of the mouth respectively.
