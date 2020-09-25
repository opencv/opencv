# High Level API: TextDetectionModel & TextRecognitionModel {#tutorial_dnn_text_spotting}

@prev_tutorial{tutorial_dnn_OCR}

## Introduction
In this tutorial, we will introduce the APIs for TextRecognitionModel & TextDetectionModel in detail.

---
#### TextRecognitionModel:

In the current version, `TextRecognitionModel` only supports CNN+RNN+CTC based algorithms,
and the greedy decoding method for CTC is provided.
For more information, please refer to the [original paper](https://arxiv.org/abs/1507.05717)

`TextRecognitionModel::recognize()` is the main function for text recognition.
- The parameter `String decodeType` is for decoding, and other methods will be added in the future.
    If decodeType is "CTC-greedy", the output of the text recognition model should be a probability matrix.
    The shape should be (T, B, Dim), where
    - T is the sequence length,
    - B is the batch size (only support B=1 in inference),
    - and Dim is the length of vocabulary +1('Blank' of CTC is at the index=0 of Dim).

- The last input parameter `const std::vector<std::vector<Point>> & roiPolygons` is optional.
    - If it is not empty, we will crop all the ROIs as the network inputs, and all the results are returned in `std::vector<String> & results`.
    - If it is empty, the whole image will be inputted.


---

#### TextDetectionModel:
In the current version, `TextDetectionModel` supports "DB" and "EAST" algorithms.
The following provided pretrained models are variants of DB (w/o deformable convolution),
and the performance can be referred to the Table.1 in the [paper]((https://arxiv.org/abs/1911.08947)).
For more information, please refer to the [official code](https://github.com/MhLiao/DB)

`TextDetectionModel::detect()` is the main function for text detection.
- All the results are returned in the parameter `std::vector<std::vector<Point>> & results`.

---

You can train your own model with more data, and convert it into ONNX format.
We encourage you to add new algorithms to these APIs.


## Pretrained Models

#### TextRecognitionModel:
```
crnn.onnx:
url: https://drive.google.com/uc?export=dowload&id=1ooaLR-rkTl8jdpGy1DoQs0-X0lQsB6Fj
sha: 270d92c9ccb670ada2459a25977e8deeaf8380d3,
alphabet_36.txt: https://drive.google.com/uc?export=dowload&id=1oPOYx5rQRp8L6XQciUwmwhMCfX0KyO4b
parameter setting: -rgb=0;
description: The classification number of this model is 36 (0~9 + a~z).
             The training dataset is MJSynth.

crnn_cs.onnx:
url: https://drive.google.com/uc?export=dowload&id=12diBsVJrS9ZEl6BNUiRp9s0xPALBS7kt
sha: a641e9c57a5147546f7a2dbea4fd322b47197cd5
alphabet_94.txt: https://drive.google.com/uc?export=dowload&id=1oKXxXKusquimp7XY1mFvj9nwLzldVgBR
parameter setting: -rgb=1;
description: The classification number of this model is 94 (0~9 + a~z + A~Z + punctuations).
             The training datasets are MJsynth and SynthText90k.

crnn_cs_CN.onnx:
url: https://drive.google.com/uc?export=dowload&id=1is4eYEUKH7HR7Gl37Sw4WPXx6Ir8oQEG
sha: 3940942b85761c7f240494cf662dcbf05dc00d14
alphabet_3944.txt: https://drive.google.com/uc?export=dowload&id=18IZUUdNzJ44heWTndDO6NNfIpJMmN-ul
parameter setting: -rgb=1;
description: The classification number of this model is 3944 (0~9 + a~z + A~Z + Chinese characters + special characters).
             The training dataset is ReCTS (https://rrc.cvc.uab.es/?ch=12).
```

More models can be found in [here](https://drive.google.com/drive/folders/1cTbQ3nuZG-EKWak6emD_s8_hHXWz7lAr?usp=sharing),
which are taken from [clovaai](https://github.com/clovaai/deep-text-recognition-benchmark).
You can train more models by [CRNN](https://github.com/meijieru/crnn.pytorch), and convert models by `torch.onnx.export`.

#### TextDetectionModel:
```
- DB_IC15_resnet50.onnx:
url: https://drive.google.com/uc?export=dowload&id=1cmAs91yFRyC2LC3nTTBNNYooyEmb0i5t
sha: 9a562c2a8fa3d95fbecd2d45690fb4f787fd496a
parameter setting: -inputHeight=736, -inputWidth=1280;
description: This model is trained on ICDAR2015, so it is only able to detect English text instances.

- DB_IC15_resnet18.onnx:
url: https://drive.google.com/uc?export=dowload&id=1JQ_-R7_feD25qE2uibpKM1S-R3keTD1U
sha: b215b5fb85d1c606d1f56f03e8c841025e72733d
parameter setting: -inputHeight=736, -inputWidth=1280;
description: This model is trained on ICDAR2015, so it is only able to detect English text instances.

- DB_TD500_resnet50.onnx:
url: https://drive.google.com/uc?export=dowload&id=1_9KB6QkIghp2VNZti_dJ3Araz-nDZBsL
sha: 6d3b7b84fc23906be7d583a8cedef727cc351fad
parameter setting: -inputHeight=736, -inputWidth=736;
description: This model is trained on MSRA-TD500, so it is able to detect English and Chinese text instances.

- DB_TD500_resnet18.onnx:
url: https://drive.google.com/uc?export=dowload&id=1f0dfqG_-Fiq4ywdrRhq8JH9PXCcJDO01
sha: dd19abe7e2326ac660ce2733be1cb7032f76d458
parameter setting: -inputHeight=736, -inputWidth=736;
description: This model is trained on MSRA-TD500, so it is able to detect English and Chinese text instances.

```

```
- EAST:
Download link: https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1
This model is based on https://github.com/argman/EAST
```

## Images for Testing
```
Text Recognition:
url: https://drive.google.com/uc?export=dowload&id=1nMcEy68zDNpIlqAn6xCk_kYcUTIeSOtN
sha: 89205612ce8dd2251effa16609342b69bff67ca3

Text Detection:
url: https://drive.google.com/uc?export=dowload&id=149tAhIcvfCYeyufRoZ9tmc2mZDKE_XrF
sha: ced3c03fb7f8d9608169a913acf7e7b93e07109b
```

## Example for Text Recognition

Step1. Loading images and models with a vocabulary
```cpp
    // Load a cropped text line image
    // you can find cropped images for testing in "Images for Testing"
    int rgb = IMREAD_COLOR; // This should be changed according to the model input requirement.
    Mat image = imread("path/to/text_rec_test.png", rgb);

    // Load models weights
    TextRecognitionModel model("path/to/crnn_cs.onnx");

    // Load vocabulary
    // vocabulary should be changed according to the text recognition model
    std::ifstream vocFile;
    vocFile.open("path/to/alphabet_94.txt");
    CV_Assert(vocFile.is_open());
    String vocLine;
    std::vector<String> vocabulary;
    while (std::getline(vocFile, vocLine)) {
        vocabulary.push_back(vocLine);
    }
    model.setVocabulary(vocabulary);
```
Step2. Setting Parameters
```cpp
    // Normalization parameters
    double scale = 1.0 / 127.5;
    Scalar mean = Scalar(127.5, 127.5, 127.5);

    // The input shape
    Size inputSize = Size(100, 32);

    model.setInputParams(scale, inputSize, mean);

    // The decoding method
    // more methods will be supported in future
    String decodeType = "CTC-greedy";
```
Step3. Inference
```cpp
    std::vector<String> recResults;
    model.recognize(image, decodeType, recResults);
    std::cout << recResults[0] << std::endl;
```

Input image:

![Picture example](text_rec_test.png)

Output:
```
welcome
```


## Example for Text Detection

Step1. Loading images and models
```cpp
    // Load an image
    // you can find some images for testing in "Images for Testing"
    Mat image = imread("/path/to/text_det_test.png");

    // Load model weights
    TextDetectionModel model("/path/to/DB_TD500_resnet50.onnx");
```
Step2. Setting Parameters
```cpp
    // Normalization parameters
    double scale = 1.0 / 255.0;
    Scalar mean = Scalar(122.67891434, 116.66876762, 104.00698793);

    // The input shape
    Size inputSize = Size(736, 736);

    model.setInputParams(scale, inputSize, mean);

    // Post-processing parameters
    float binThresh = 0.3;
    float polyThresh = 0.5;
    uint maxCandidates = 200;
    double unclipRatio = 2.0;
    int outputType = 0;  // 0: multi-oriented quadrilateral; 1: polygon
```
Step3. Inference
```cpp
    std::vector<std::vector<Point>> results;
    model.detect(image, results, outputType, binThresh, polyThresh, unclipRatio, maxCandidates);

    // Visualization
    polylines(image, results, true, Scalar(0, 255, 0), 2);
    imshow("Text Detection", image);
    waitKey();
```

Output:

![Picture example](text_det_test_results.jpg)

## Example for Text Spotting

After following the steps above, it is easy to get the detection results `std::vector<std::vector<Point>> detResults` of an input image.
Then, we can set the last optional parameter of `recognize` with `detResults`, and get the final results.
```cpp
    std::vector<String> recResults;
    recognizer.recognize(recInput, decodeType, recResults, detResults);

    // Visualization
    for (uint i = 0; i < detResults.size(); i++) {
        polylines(image, detResults[i], true, Scalar(0, 255, 0), 2);
        putText(image, recResults[i], detResults[i][3], FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 0, 255), 2);
    }
    imshow("Text Spotting", image);
    waitKey();
```

Output Examples:

![Picture example](detect_test1.jpg)

![Picture example](detect_test2.jpg)

## Source Code
The [source code](https://github.com/opencv/opencv/blob/master/modules/dnn/src/model.cpp)
of these APIs can be found in the DNN module.

## Detailed Sample
For more information, please refer to:
- [samples/dnn/scene_text_recognition.cpp](https://github.com/opencv/opencv/blob/master/samples/dnn/scene_text_recognition.cpp)
- [samples/dnn/scene_text_detection.cpp](https://github.com/opencv/opencv/blob/master/samples/dnn/scene_text_detection.cpp)
- [samples/dnn/text_detection.cpp](https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.cpp)
- [samples/dnn/scene_text_spotting.cpp](https://github.com/opencv/opencv/blob/master/samples/dnn/scene_text_spotting.cpp)

#### Test with an image
Examples:
```bash
scene_text_recognition -mp=path/to/crnn_cs.onnx -i=path/to/an/image -rgb=1 -vp=/path/to/alphabet_94.txt
scene_text_detection -mp=path/to/DB_TD500_resnet50.onnx -i=path/to/an/image -ih=736 -iw=736
scene_text_spotting -dmp=path/to/DB_IC15_resnet50.onnx -rmp=path/to/crnn_cs.onnx -i=path/to/an/image -iw=1280 -ih=736 -rgb=1 -vp=/path/to/alphabet_94.txt
```

#### Test on public datasets
Text Recognition:

The download link for testing images can be found in the **Images for Testing**


Examples:
```bash
scene_text_recognition -mp=path/to/crnn.onnx -e=true -edp=path/to/evaluation_data_rec -vp=/path/to/alphabet_36.txt -rgb=0
scene_text_recognition -mp=path/to/crnn_cs.onnx -e=true -edp=path/to/evaluation_data_rec -vp=/path/to/alphabet_94.txt -rgb=1
```

Text Detection:

The download links for testing images can be found in the **Images for Testing**

Examples:
```bash
scene_text_detection -mp=path/to/DB_TD500_resnet50.onnx -e=true -edp=path/to/evaluation_data_det/TD500 -ih=736 -iw=736
scene_text_detection -mp=path/to/DB_IC15_resnet50.onnx -e=true -edp=path/to/evaluation_data_det/IC15 -ih=736 -iw=1280
```
