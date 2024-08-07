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

    Usage: ./example_dnn_text_detection DB --ocr=<path to recognition model>
*/
#include <iostream>
#include <fstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "common.hpp"

using namespace cv;
using namespace std;
using namespace cv::dnn;

const string about = "Use this script for Text Detection and Recognition using OpenCV. \n\n"
        "Firstly, download required models using `download_models.py` (if not already done). Set environment variable OPENCV_DOWNLOAD_CACHE_DIR to specify where models should be downloaded. Also, point OPENCV_SAMPLES_DATA_PATH to opencv/samples/data.\n"
        "To run:\n"
        "\t Example: ./example_dnn_text_detection modelName(i.e. DB or East) --ocr=<path to ResNet_CTC.onnx>\n\n"
        "Model path can also be specified using --model argument. \n\n"
        "Download link for recognition model: https://drive.google.com/drive/folders/1cTbQ3nuZG-EKWak6emD_s8_hHXWz7lAr?usp=sharing \n\n";

// Command-line keys to parse the input arguments
string keys =
    "{ help  h                        |                     | Print help message. }"
    "{ input i                        |      right.jpg      | Path to an input image. }"
    "{ @alias                         |                     | An alias name of model to extract preprocessing parameters from models.yml file. }"
    "{ zoo                            |  ../dnn/models.yml  | An optional path to file with preprocessing parameters }"
    "{ ocr                            |                     | Path to a binary .onnx model for recognition. }"
    "{ model                          |                     | Path to detection model file. }"
    "{ thr                            |        0.5          | Confidence threshold for EAST detector. }"
    "{ nms                            |        0.4          | Non-maximum suppression threshold for EAST detector. }"
    "{ binaryThreshold bt             |        0.3          | Confidence threshold for the binary map in DB detector. }"
    "{ polygonThreshold pt            |        0.5          | Confidence threshold for polygons in DB detector. }"
    "{ maxCandidate max               |        200          | Max candidates for polygons in DB detector. }"
    "{ unclipRatio ratio              |        2.0          | Unclip ratio for DB detector. }"
    "{ vocabularyPath vp              |   alphabet_36.txt   | Path to vocabulary file. }";

// Function prototype for the four-point perspective transform
static void fourPointsTransform(const Mat& frame, const Point2f vertices[], Mat& result);

int main(int argc, char** argv) {
    // Setting up command-line parser with the specified keys
    CommandLineParser parser(argc, argv, keys);

    if (!parser.has("@alias") || parser.has("help"))
    {
        cout << about << endl;
        parser.printMessage();
        return -1;
    }
    const string modelName = parser.get<String>("@alias");
    const string zooFile = findFile(parser.get<String>("zoo"));

    keys += genPreprocArguments(modelName, zooFile);
    parser = CommandLineParser(argc, argv, keys);
    parser.about(about);

    // Parsing command-line arguments
    String sha1 = parser.get<String>("sha1");
    String detModelPath = findModel(parser.get<String>("model"), sha1);
    String ocr = findFile(parser.get<String>("ocr"));
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

    vector<vector<Point>> detResults;
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
        cout << "[ERROR]: Unsupported file config for the detector model. Valid values: east/db" << endl;
        return 1;
    }

    // Reading and storing vocabulary for text recognition
    CV_Assert(!vocPath.empty());
    ifstream vocFile;
    vocFile.open(samples::findFile(vocPath));
    CV_Assert(vocFile.is_open());
    String vocLine;
    vector<String> vocabulary;
    while (getline(vocFile, vocLine)) {
        vocabulary.push_back(vocLine);
    }

    // Initializing text recognition model with the provided model path
    if (ocr.empty()) {
        cout << "[ERROR] Please pass recognition model --ocr to run the sample" << endl;
        return -1;
    }
    TextRecognitionModel recognizer(ocr);
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
        vector< vector<Point> > contours;
        for (uint i = 0; i < detResults.size(); i++) {
            const auto& quadrangle = detResults[i];
            CV_CheckEQ(quadrangle.size(), (size_t)4, "");

            contours.emplace_back(quadrangle);

            vector<Point2f> quadrangle_2f;
            for (int j = 0; j < 4; j++)
                quadrangle_2f.emplace_back(detResults[i][j]);

            // Cropping the detected text region using a four-point transform
            Mat cropped;
            fourPointsTransform(recInput, &quadrangle_2f[0], cropped);

            // Recognizing text from the cropped image
            string recognitionResult = recognizer.recognize(cropped);
            cout << i << ": '" << recognitionResult << "'" << endl;

            // Displaying the recognized text on the image
            putText(frame, recognitionResult, detResults[i][3], FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        }
        // Drawing detected text regions on the image
        polylines(frame, contours, true, Scalar(0, 255, 0), 2);
    } else {
        cout << "No Text Detected." << endl;
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
