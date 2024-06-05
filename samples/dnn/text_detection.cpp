/*
    Text detection model (EAST): https://github.com/argman/EAST
    Download link for EAST model: https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1

    DB detector model:
    https://drive.google.com/uc?export=download&id=17_ABp79PlFt9yPCxSaarVc_DKTmrSGGf

    CRNN Text recognition model sourced from: https://github.com/meijieru/crnn.pytorch
    How to convert from .pb to .onnx:
    Using classes from: https://github.com/meijieru/crnn.pytorch/blob/master/models/crnn.py

    Additional converted ONNX text recognition models available for direct download:
    Download link: https://drive.google.com/drive/folders/1cTbQ3nuZG-EKWak6emD_s8_hHXWz7lAr?usp=sharing
    These models are taken from: https://github.com/clovaai/deep-text-recognition-benchmark

    Importing and using the CRNN model in PyTorch:
    import torch
    from models.crnn import CRNN

    model = CRNN(32, 1, 37, 256)
    model.load_state_dict(torch.load('crnn.pth'))
    dummy_input = torch.randn(1, 1, 32, 100)
    torch.onnx.export(model, dummy_input, "crnn.onnx", verbose=True)

    Usage: ./example_dnn_text_detection DB -ocr=<path to recognition model>
*/
#include <iostream>
#include <fstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "common.hpp"

using namespace cv;
using namespace cv::dnn;

// Command-line keys to parse the input arguments
std::string keys =
    "{ help  h                        |     | Print help message. }"
    "{ input i                        | box_in_scene.png | Path to an input image. }"
    "{ @alias                         |     | An alias name of model to extract preprocessing parameters from models.yml file. }"
    "{ zoo                            | models.yml | An optional path to file with preprocessing parameters }"
    "{ recModelPath ocr               |     | Path to a binary .onnx model for recognition. }"
    "{ thr                            | 0.5 | Confidence threshold for EAST detector. }"
    "{ nms                            | 0.4 | Non-maximum suppression threshold for EAST detector. }"
    "{ binaryThreshold bt             | 0.3 | Confidence threshold for the binary map in DB detector. }"
    "{ polygonThreshold pt            | 0.5 | Confidence threshold for polygons in DB detector. }"
    "{ maxCandidate max               | 200 | Max candidates for polygons in DB detector. }"
    "{ unclipRatio ratio              | 2.0 | Unclip ratio for DB detector. }"
    "{ vocabularyPath vp              | alphabet_36.txt | Path to vocabulary file. }";

// Function prototype for the four-point perspective transform
static void fourPointsTransform(const Mat& frame, const Point2f vertices[], Mat& result);

int main(int argc, char** argv) {
    // Setting up command-line parser with the specified keys
    CommandLineParser parser(argc, argv, keys);
    const std::string modelName = parser.get<String>("@alias");
    const std::string zooFile = parser.get<String>("zoo");

    keys += genPreprocArguments(modelName, zooFile);

    parser = CommandLineParser(argc, argv, keys);
    parser.about("Text Detection and Recognition using OpenCV"
                 "Example: ./example_dnn_text_detection modelName(i.e. DB or East) -ocr=<path to ResNet_CTC.onnx>" );

    // Display help message if no arguments are provided or 'help' is requested
    if (argc == 1 || parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    // Parsing command-line arguments
    String detModelPath = findFile(parser.get<String>("model"));
    String recModelPath = parser.get<String>("recModelPath");
    int height = parser.get<int>("height");
    int width = parser.get<int>("width");
    bool imreadRGB = parser.get<bool>("rgb");
    String vocPath = parser.get<String>("vocabularyPath");
    float binThresh = parser.get<float>("binaryThreshold");
    float polyThresh = parser.get<float>("polygonThreshold");
    double unclipRatio = parser.get<double>("unclipRatio");
    uint maxCandidates = parser.get<uint>("maxCandidate");
    float confThreshold = parser.get<float>("thr");
    float nmsThreshold = parser.get<float>("nms");
    Scalar mean = parser.get<Scalar>("mean");

    // Ensuring the provided arguments are valid
    if (!parser.check()) {
        parser.printErrors();
        return 1;
    }

    // Asserting detection model path is provided
    CV_Assert(!detModelPath.empty());

    std::vector<std::vector<Point>> detResults;
    // Reading the input image
    Mat frame = imread(samples::findFile(parser.get<String>("input")));

    // Initializing and configuring the text detection model based on the provided config
    if (modelName == "East") {
        // EAST Detector initialization
        TextDetectionModel_EAST detector(detModelPath);
        detector.setConfidenceThreshold(confThreshold)
                .setNMSThreshold(nmsThreshold);
        // Setting input parameters specific to EAST model
        detector.setInputParams(1.0, Size(width, height), mean, true);
        // Performing text detection
        detector.detect(frame, detResults);
    }
    else if (modelName == "DB") {
        // DB Detector initialization
        TextDetectionModel_DB detector(detModelPath);
        detector.setBinaryThreshold(binThresh)
                .setPolygonThreshold(polyThresh)
                .setUnclipRatio(unclipRatio)
                .setMaxCandidates(maxCandidates);
        // Setting input parameters specific to DB model
        detector.setInputParams(1.0 / 255.0, Size(width, height), mean);
        // Performing text detection
        detector.detect(frame, detResults);
    }
    else {
        std::cerr << "Unsupported file config for the detector model. Valid values: east/db" << std::endl;
        return 1;
    }

    // Reading and storing vocabulary for text recognition
    CV_Assert(!vocPath.empty());
    std::ifstream vocFile;
    vocFile.open(samples::findFile(vocPath));
    CV_Assert(vocFile.is_open());
    String vocLine;
    std::vector<String> vocabulary;
    while (std::getline(vocFile, vocLine)) {
        vocabulary.push_back(vocLine);
    }

    // Initializing text recognition model with the provided model path
    CV_Assert(!recModelPath.empty());
    TextRecognitionModel recognizer(recModelPath);
    recognizer.setVocabulary(vocabulary);
    recognizer.setDecodeType("CTC-greedy");

    // Setting input parameters for the recognition model
    double recScale = 1.0 / 127.5;
    Scalar recMean = Scalar(127.5);
    Size recInputSize = Size(100, 32);
    recognizer.setInputParams(recScale, recInputSize, recMean);

    // Process detected text regions for recognition
    if (detResults.size() > 0) {
        // Text Recognition
        Mat recInput;
        if (!imreadRGB) {
            cvtColor(frame, recInput, cv::COLOR_BGR2GRAY);
        } else {
            recInput = frame;
        }
        std::vector< std::vector<Point> > contours;
        for (uint i = 0; i < detResults.size(); i++) {
            const auto& quadrangle = detResults[i];
            CV_CheckEQ(quadrangle.size(), (size_t)4, "");

            contours.emplace_back(quadrangle);

            std::vector<Point2f> quadrangle_2f;
            for (int j = 0; j < 4; j++)
                quadrangle_2f.emplace_back(detResults[i][j]);

            // Cropping the detected text region using a four-point transform
            Mat cropped;
            fourPointsTransform(recInput, &quadrangle_2f[0], cropped);

            // Recognizing text from the cropped image
            std::string recognitionResult = recognizer.recognize(cropped);
            std::cout << i << ": '" << recognitionResult << "'" << std::endl;

            // Displaying the recognized text on the image
            putText(frame, recognitionResult, detResults[i][3], FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        }
        // Drawing detected text regions on the image
        polylines(frame, contours, true, Scalar(0, 255, 0), 2);
    } else {
        std::cout << "No Text Detected." << std::endl;
    }

    // Displaying the final image with detected and recognized text
    imshow("Text Detection and Recognition", frame);
    waitKey(0);

    return 0;
}

// Performs a perspective transform for a four-point region
static void fourPointsTransform(const Mat& frame, const Point2f vertices[], Mat& result) {
    const Size outputSize = Size(100, 32);
    // Defining target vertices for the perspective transform
    Point2f targetVertices[4] = {
        Point(0, outputSize.height - 1),
        Point(0, 0),
        Point(outputSize.width - 1, 0),
        Point(outputSize.width - 1, outputSize.height - 1)
    };
    // Computing the perspective transform matrix
    Mat rotationMatrix = getPerspectiveTransform(vertices, targetVertices);
    // Applying the perspective transform to the region
    warpPerspective(frame, result, rotationMatrix, outputSize);
}
