# DNN-based Face Detection And Recognition {#tutorial_dnn_face}

@tableofcontents

@prev_tutorial{tutorial_dnn_text_spotting}
@next_tutorial{pytorch_cls_tutorial_dnn_conversion}

| | |
| -: | :- |
| Original Author | Chengrui Wang, Yuantao Feng |
| Compatibility | OpenCV >= 4.5.4 |

## Introduction

In this section, we introduce cv::FaceDetectorYN class for face detection and cv::FaceRecognizerSF class for face recognition.

## Models

There are two models (ONNX format) pre-trained and required for this module:
- [Face Detection](https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet):
    - Size: 338KB
    - Results on WIDER Face Val set: 0.830(easy), 0.824(medium), 0.708(hard)
- [Face Recognition](https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface)
    - Size: 36.9MB
    - Results:

    | Database | Accuracy | Threshold (normL2) | Threshold (cosine) |
    | -------- | -------- | ------------------ | ------------------ |
    | LFW      | 99.60%   | 1.128              | 0.363              |
    | CALFW    | 93.95%   | 1.149              | 0.340              |
    | CPLFW    | 91.05%   | 1.204              | 0.275              |
    | AgeDB-30 | 94.90%   | 1.202              | 0.277              |
    | CFP-FP   | 94.80%   | 1.253              | 0.212              |

## Code

@add_toggle_cpp
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/4.x/samples/dnn/face_detect.cpp)

-   **Code at glance:**
    @include samples/dnn/face_detect.cpp
@end_toggle

@add_toggle_python
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/4.x/samples/dnn/face_detect.py)

-   **Code at glance:**
    @include samples/dnn/face_detect.py
@end_toggle

Explanation
-----------

@add_toggle_cpp
@snippet dnn/face_detect.cpp initialize_FaceDetectorYN
@snippet dnn/face_detect.cpp inference
@end_toggle

@add_toggle_python
@snippet dnn/face_detect.py initialize_FaceDetectorYN
@snippet dnn/face_detect.py inference
@end_toggle

The detection output `faces` is a two-dimension array of type CV_32F, whose rows are the detected face instances, columns are the location of a face and 5 facial landmarks. The format of each row is as follows:

```
x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm
```
, where `x1, y1, w, h` are the top-left coordinates, width and height of the face bounding box, `{x, y}_{re, le, nt, rcm, lcm}` stands for the coordinates of right eye, left eye, nose tip, the right corner and left corner of the mouth respectively.


### Face Recognition

Following Face Detection, run codes below to extract face feature from facial image.

@add_toggle_cpp
@snippet dnn/face_detect.cpp initialize_FaceRecognizerSF
@snippet dnn/face_detect.cpp facerecognizer
@end_toggle

@add_toggle_python
@snippet dnn/face_detect.py initialize_FaceRecognizerSF
@snippet dnn/face_detect.py facerecognizer
@end_toggle

After obtaining face features *feature1* and *feature2* of two facial images, run codes below to calculate the identity discrepancy between the two faces.

@add_toggle_cpp
@snippet dnn/face_detect.cpp match
@end_toggle

@add_toggle_python
@snippet dnn/face_detect.py match
@end_toggle

For example, two faces have same identity if the cosine distance is greater than or equal to 0.363, or the normL2 distance is less than or equal to 1.128.

## Reference:

- https://github.com/ShiqiYu/libfacedetection
- https://github.com/ShiqiYu/libfacedetection.train
- https://github.com/zhongyy/SFace

## Acknowledgement

Thanks [Professor Shiqi Yu](https://github.com/ShiqiYu/) and [Yuantao Feng](https://github.com/fengyuentau) for training and providing the face detection model.

Thanks [Professor Deng](http://www.whdeng.cn/), [PhD Candidate Zhong](https://github.com/zhongyy/) and [Master Candidate Wang](https://github.com/crywang/) for training and providing the face recognition model.
