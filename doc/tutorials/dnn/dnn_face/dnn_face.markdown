# DNN-based Face Detection And Recognition {#tutorial_dnn_face}

@tableofcontents

@prev_tutorial{tutorial_dnn_text_spotting}
@next_tutorial{pytorch_cls_tutorial_dnn_conversion}

| | |
| -: | :- |
| Original Author | Chengrui Wang, Yuantao Feng |
| Compatibility | OpenCV >= 4.5.1 |

## Introduction

In this section, we introduce the DNN-based module for face detection and face recognition. Models can be obtained in [Models](#Models). The usage of `FaceDetectorYN` and `FaceRecognizerSF` are presented in [Usage](#Usage).

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
    | LFW      | 99.60%   | 1.128              | 0.363              |
    | CALFW    | 93.95%   | 1.149              | 0.340              |
    | CPLFW    | 91.05%   | 1.204              | 0.275              |
    | AgeDB-30 | 94.90%   | 1.202              | 0.277              |
    | CFP-FP   | 94.80%   | 1.253              | 0.212              |

## Usage

### DNNFaceDetector

```cpp
// Initialize FaceDetectorYN
Ptr<FaceDetectorYN> faceDetector = FaceDetectorYN::create(onnx_path, "", image.size(), score_thresh, nms_thresh, top_k);

// Forward
Mat faces;
faceDetector->detect(image, faces);
```

The detection output `faces` is a two-dimension array of type CV_32F, whose rows are the detected face instances, columns are the location of a face and 5 facial landmarks. The format of each row is as follows:

```
x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm
```
, where `x1, y1, w, h` are the top-left coordinates, width and height of the face bounding box, `{x, y}_{re, le, nt, rcm, lcm}` stands for the coordinates of right eye, left eye, nose tip, the right corner and left corner of the mouth respectively.


### Face Recognition

Following Face Detection, run codes below to extract face feature from facial image.

```cpp
// Initialize FaceRecognizerSF with model path (cv::String)
Ptr<FaceRecognizerSF> faceRecognizer = FaceRecognizerSF::create(model_path, "");

// Aligning and cropping facial image through the first face of faces detected by dnn_face::DNNFaceDetector
Mat aligned_face;
faceRecognizer->alignCrop(image, faces.row(0), aligned_face);

// Run feature extraction with given aligned_face (cv::Mat)
Mat feature;
faceRecognizer->feature(aligned_face, feature);
feature = feature.clone();
```

After obtaining face features *feature1* and *feature2* of two facial images, run codes below to calculate the identity discrepancy between the two faces.

```cpp
// Calculating the discrepancy between two face features by using cosine distance.
double cos_score = faceRecognizer->match(feature1, feature2, FaceRecognizer::DisType::COSINE);
// Calculating the discrepancy between two face features by using normL2 distance.
double L2_score = faceRecognizer->match(feature1, feature2, FaceRecognizer::DisType::NORM_L2);
```

For example, two faces have same identity if the cosine distance is greater than or equal to 0.363, or the normL2 distance is less than or equal to 1.128.

## Reference:

- https://github.com/ShiqiYu/libfacedetection
- https://github.com/ShiqiYu/libfacedetection.train
- https://github.com/zhongyy/SFace

## Acknowledgement

Thanks [Professor Shiqi Yu](https://github.com/ShiqiYu/) and [Yuantao Feng](https://github.com/fengyuentau) for training and providing the face detection model.

Thanks [Professor Deng](http://www.whdeng.cn/), [PhD Candidate Zhong](https://github.com/zhongyy/) and [Master Candidate Wang](https://github.com/crywang/) for training and providing the face recognition model.
