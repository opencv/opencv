# High Level API: TextDetectionModel & TextRecognitionModel {#tutorial_dnn_text_spotting}

@prev_tutorial{tutorial_dnn_custom_layers}

## Introduction
In this tutorial, we will introduce the APIs for TextRecognitionModel & TextDetectionModel in detail.

---
#### TextRecognitionModel:

In the current version, TextRecognitionModel only supports CNN+RNN+CTC based algorithms,
and the greedy decoding method for CTC is provided.
For more information, please refer to the [original paper](https://arxiv.org/abs/1507.05717)

**TextRecognitionModel::recognize()** is the main function for text recognition.
- The parameter ***String decodeType*** is for decoding, and other methods will be added in the future.
    If decodeType is "CTC-greedy", the output of the text recognition model should be a probability matrix.
    The shape should be (T, B, Dim), where
    - T is the sequence length,
    - B is the batch size (only support B=1 in inference),
    - and Dim is the length of vocabulary +1('Blank' of CTC is at the index=0 of Dim).

- The last input parameter ***const std::vector\<std::vector\<Point\>\> & roiPolygons*** is optional.
    - If it is not empty, we will crop all the ROIs as the network inputs, and all the results are returned in ***std::vector\<String\> & results***.
    - If it is empty, the whole image will be inputted.


---

#### TextDetectionModel:
In the current version, TextDetectionModel only supports "DB" algorithms.
The following provided pretrained models are variants of DB (w/o deformable convolution),
and the performance can be referred to the Table.1 in the [paper]((https://arxiv.org/abs/1911.08947)).
For more information, please refer to the [official code](https://github.com/MhLiao/DB)

**TextDetectionModel::detect()** is the main function for text detection.
- All the results are returned in the parameter ***std::vector\<std::vector\<Point\>\> & results***.

---

You can train your own model with more data, and convert it into ONNX format.
We encourage you to add new algorithms to these APIs, and there will be a simple tutorial in doc/tutorials/dnn.


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
             The training dataets are MJsynth and SynthText90k.

crnn_cs_CN.onnx:
url: https://drive.google.com/uc?export=dowload&id=1is4eYEUKH7HR7Gl37Sw4WPXx6Ir8oQEG
sha: 3940942b85761c7f240494cf662dcbf05dc00d14
alphabet_3944.txt: https://drive.google.com/uc?export=dowload&id=18IZUUdNzJ44heWTndDO6NNfIpJMmN-ul
parameter setting: -rgb=1;
description: The classification number of this model is 3944 (0~9 + a~z + A~Z + Chinese characters + special characters).
             The training dataet is ReCTS (https://rrc.cvc.uab.es/?ch=12).
```

#### TextDetectionModel:
```
- DB_IC15_resnet50.onnx:
url: https://drive.google.com/uc?export=dowload&id=1cmAs91yFRyC2LC3nTTBNNYooyEmb0i5t
sha: 9a562c2a8fa3d95fbecd2d45690fb4f787fd496a
parameter setting: -inputHeight=736, -inputWidth=1280;
description: This model is trained on ICDAR2015, so it is only able to detect English text instances.

- DB_TD500_resnet50.onnx:
url: https://drive.google.com/uc?export=dowload&id=1_9KB6QkIghp2VNZti_dJ3Araz-nDZBsL
sha: 6d3b7b84fc23906be7d583a8cedef727cc351fad
parameter setting: -inputHeight=736, -inputWidth=736;
description: This model is trained on MSRA-TD500, so it is able to detect English and Chinese text instances.
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

```cpp
#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace dnn;

int main()
{
    // Load model weights
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

    // Normalization parameters
    double scale = 1.0 / 127.5;
    Scalar mean = Scalar(127.5);

    // The input shape is fixed to (width=100, height=32)
    Size inputSize = Size(100, 32);

    model.setInputParams(scale, inputSize, mean);

    // more decoding method will be supported in future
    String decodeType = "CTC-greedy";

    // Load a cropped text line image
    // you can find cropped images for testing in "Images for Testing"
    int rgb = 1; // This should be changed according to the model input requirement.
    Mat image = imread("path/to/text_rec_test.png", rgb);

    // Inference
    std::vector<String> recResults;
    model.recognize(image, decodeType, vocabulary, recResults);
    std::cout << recResults[0] << std::endl;

    return 0;
}

```
Input image:

![Picture example](https://github.com/HannibalAPE/images/blob/master/opencv/text_rec_test.png)

Output:
```
welcome
```


## Example for Text Detection

```cpp
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace dnn;

int main()
{
    // Load a image
    // you can find some images for testing in "Images for Testing"
    Mat image = imread("/path/to/text_det_test.png");

    // Load model weights
    TextDetectionModel model("/path/to/DB_TD500_resnet50.onnx");

    // Normalization parameters
    double scale = 1.0 / 255.0;
    Scalar mean = Scalar(122.67891434, 116.66876762, 104.00698793);

    // The input shape is fixed when models are converted to ONNX format
    Size inputSize = Size(736, 736);

    model.setInputParams(scale, inputSize, mean);

    // Post-processing parameters
    float binThresh = 0.3;
    float polyThresh = 0.5;
    uint maxCandidates = 200;
    double unclipRatio = 2.0;
    int outputType = 0;

    // Inference
    std::vector<std::vector<Point>> results;
    model.detect(image, results, outputType, binThresh, polyThresh, unclipRatio, maxCandidates);

    // Visualization
    polylines(image, results, true, Scalar(0, 255, 0), 2);
    imshow("Text Detection", image);
    waitKey();

    return 0;
}

```

Output:

![Picture example](https://github.com/HannibalAPE/images/blob/master/opencv/text_det_test_results.jpg)

## Example for Text Spotting

```cpp
#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace dnn;

int main()
{
    // Load a image
    // you can find cropped images for testing in "Images for Testing"
    Mat image = imread("/path/to/an/image/");

    // Load networks
    TextDetectionModel detector("/path/to/DB_IC15_resnet50.onnx");
    TextRecognitionModel recognizer("/path/to/crnn_cs.onnx");

    // Load vocabulary
    // vocabulary should be changed according to the text recognition model
    std::ifstream vocFile;
    vocFile.open("/path/to/alphabet_94.txt");
    CV_Assert(vocFile.is_open());
    String vocLine;
    std::vector<String> vocabulary;
    while (std::getline(vocFile, vocLine)) {
        vocabulary.push_back(vocLine);
    }

    // Parameters for Detection
    float binThresh = 0.3;
    float polyThresh = 0.5;
    uint maxCandidates = 200;
    double unclipRatio = 2.0;
    int outputType = 0;
    double detScale = 1.0 / 255.0;
    int height = 736;
    int width = 1280;
    Size detInputSize = Size(width, height);
    Scalar detMean = Scalar(122.67891434, 116.66876762, 104.00698793);
    detector.setInputParams(detScale, detInputSize, detMean);

    // Parameters for Recognition
    int imreadRGB = 1;
    const String decodeType = "CTC-greedy";
    double recScale = 1.0 / 127.5;
    Scalar recMean = Scalar(127.5);
    Size recInputSize = Size(100, 32);
    recognizer.setInputParams(recScale, recInputSize, recMean);

    // Detection
    std::vector<std::vector<Point>> detResults;
    detector.detect(image, detResults, outputType, binThresh, polyThresh, unclipRatio, maxCandidates);
    if (detResults.size() == 0) {
        std::cout << "No Text Detected." << std::endl;
        return 0;
    }

    std::vector<String> recResults;
    Mat recInput;
    if (!imreadRGB) {
        cvtColor(image, recInput, cv::COLOR_BGR2GRAY);
    } else {
        recInput = image;
    }

    // Recognition
    recognizer.recognize(recInput, decodeType, vocabulary, recResults, detResults);

    // Visualization
    for (uint i = 0; i < detResults.size(); i++) {
        polylines(image, detResults[i], true, Scalar(0, 255, 0), 2);
        putText(image, recResults[i], detResults[i][3], FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 0, 255), 2);
    }
    imshow("Text Spotting", image);
    waitKey();

    return 0;
}

```

Output Examples:

![Picture example](https://github.com/HannibalAPE/images/blob/master/opencv/detect_test1.png)

![Picture example](https://github.com/HannibalAPE/images/blob/master/opencv/detect_test2.png)

## Source Code
The [source code](https://github.com/opencv/opencv/blob/master/modules/dnn/src/model.cpp)
of these APIs can be found in the DNN module.

## Detailed Sample
For more information, please refer to [samples/dnn/scene_text_detection.cpp](https://github.com/opencv/opencv/blob/master/samples/dnn/scene_text_detection.cpp),
and [samples/dnn/scene_text_spotting.cpp](https://github.com/opencv/opencv/blob/master/samples/dnn/scene_text_spotting.cpp).

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
