# DNN-based Face Detection And Recognition {#tutorial_dnn_face}

@tableofcontents

@prev_tutorial{tutorial_dnn_text_spotting}
@next_tutorial{pytorch_cls_tutorial_dnn_conversion}

| | |
| -: | :- |
| Original Author | Chengrui Wang, Yuantao Feng |
| Compatibility | OpenCV >= 5.0.0 |

## Introduction

In this section, we introduce the DNN-based module for face detection and face recognition. Models can be obtained in [Models](#models). The usage of `FaceDetectionModel_YN` and `FaceRecognitionModel_SF` are presented in [Usage](#face-detection).

## Models

There are two models (ONNX format) pre-trained and required for this module:
- [Face Detection Model (YuNet)](https://github.com/ShiqiYu/libfacedetection.train/tree/master/tasks/task1/onnx):
    - Size: 337KB
    - Results on WIDER Face Val set: 0.830(easy), 0.824(medium), 0.708(hard)
- [Face Recognition Model (SFace))](https://drive.google.com/file/d/1ClK9WiB492c5OZFKveF3XiHCejoOxINW/view?usp=sharing)
    - Size: 36.9MB
    - Results:

    | Database | Accuracy | Threshold (normL2) | Threshold (cosine) |
    | -------- | -------- | ------------------ | ------------------ |
    | LFW      | 99.60%   | 1.128              | 0.363              |
    | CALFW    | 93.95%   | 1.149              | 0.340              |
    | CPLFW    | 91.05%   | 1.204              | 0.275              |
    | AgeDB-30 | 94.90%   | 1.202              | 0.277              |
    | CFP-FP   | 94.80%   | 1.253              | 0.212              |

## Face Detection

### Code

@add_toggle_cpp
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/5.x/samples/dnn/face_detector.cpp)

-   **Code at glance:**
    @include samples/dnn/face_detector.cpp
@end_toggle

@add_toggle_python
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/5.x/samples/dnn/face_detector.py)

-   **Code at glance:**
    @include samples/dnn/face_detector.py
@end_toggle

### Explanation

@add_toggle_cpp
@snippet dnn/face_detector.cpp Create_FaceDetectionModel_YN
@end_toggle

@add_toggle_python
@snippet dnn/face_detector.py Create_FaceDetectionModel_YN
@end_toggle

Create `FaceDetectionModel_YN` by specifying the face detection model file path.

@add_toggle_cpp
@snippet dnn/face_detector.cpp Face_Detection
@end_toggle

@add_toggle_python
@snippet dnn/face_detector.py Face_Detection
@end_toggle

The detection output `boxes` and `confidences` is bounding box (`Rect` style) and confidencie of each faces.

@add_toggle_cpp
@snippet dnn/face_detector.cpp Face_Landmarks
@end_toggle

@add_toggle_python
@snippet dnn/face_detector.py Face_Landmarks
@end_toggle

In the case of `FaceDetectionModel_YN`, the detection output `landmarks` is facial landmarks (5 points) of each faces.
The facial landmarks are listed in the following order.

```
Right-Eye, Left-Eye, Nose, Right-Mouth Corner, Right-Mouth Corner
```

## Face Recognition

### Code

@add_toggle_cpp
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/5.x/samples/dnn/face_recognizer.cpp)

-   **Code at glance:**
    @include samples/dnn/face_recognizer.cpp
@end_toggle

@add_toggle_python
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/5.x/samples/dnn/face_recognizer.py)

-   **Code at glance:**
    @include samples/dnn/face_recognizer.py
@end_toggle

### Explanation

@add_toggle_cpp
@snippet dnn/face_recognizer.cpp Create_FaceRecognitionModel_SF
@end_toggle

@add_toggle_python
@snippet dnn/face_recognizer.py Create_FaceRecognitionModel_SF
@end_toggle

Create `FaceRecognitionModel_SF` by specifying the face recognition model file path.

@add_toggle_cpp
@snippet dnn/face_recognizer.cpp Align_Crop
@end_toggle

@add_toggle_python
@snippet dnn/face_recognizer.py Align_Crop
@end_toggle

Aligning and cropping facial image using facial landmarks.
In the case of `FaceRecognitionModel_SF`, it requires five points of facial landmarks that listed in the following order (Right-Eye, Left-Eye, Nose, Right-Mouth Corner, and Right-Mouth Corner).
If you want to use facial landmarks that prepared by other method than `FaceDetectionModel_YN`, please adust to this format.
The aligning and cropping output is aligned face image that image size of 112x112, and centerized nose.

@add_toggle_cpp
@snippet dnn/face_recognizer.cpp Extract_Feature
@end_toggle

@add_toggle_python
@snippet dnn/face_recognizer.py Extract_Feature
@end_toggle

Calculate facial features from each aligned face images.
In the case of `FaceRecognitionModel_SF`, the face features are 128 dimensional.

@add_toggle_cpp
@snippet dnn/face_recognizer.cpp Match_Features
@end_toggle

@add_toggle_python
@snippet dnn/face_recognizer.py Match_Features
@end_toggle

After obtaining face features *face_feature1* and *face_feature2* of two facial images, run codes below to calculate the identity discrepancy between the two faces.

@add_toggle_cpp
@snippet dnn/face_recognizer.cpp Check_Identity
@end_toggle

@add_toggle_python
@snippet dnn/face_recognizer.py Check_Identity
@end_toggle

For example, two faces have same identity if the cosine distance is greater than or equal to 0.363, or the normL2 distance is less than or equal to 1.128.

## Reference:

- https://github.com/ShiqiYu/libfacedetection
- https://github.com/ShiqiYu/libfacedetection.train
- https://github.com/zhongyy/SFace

## Acknowledgement

Thanks [Professor Shiqi Yu](https://github.com/ShiqiYu/) and [Yuantao Feng](https://github.com/fengyuentau) for training and providing the face detection model.

Thanks [Professor Deng](http://www.whdeng.cn/), [PhD Candidate Zhong](https://github.com/zhongyy/) and [Master Candidate Wang](https://github.com/crywang/) for training and providing the face recognition model.
