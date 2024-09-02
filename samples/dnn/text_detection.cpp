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

    Usage: ./example_dnn_text_detection DB
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
        "Firstly, download required models using `download_models.py` (if not already done). Set environment variable OPENCV_DOWNLOAD_CACHE_DIR to point to the directory where models are downloaded. Also, point OPENCV_SAMPLES_DATA_PATH to opencv/samples/data.\n"
        "To run:\n"
        "\t Example: ./example_dnn_text_detection modelName(i.e. DB or East) --ocr_model=<path to VGG_CTC.onnx>\n\n"
        "Detection model path can also be specified using --model argument. \n\n"
        "Download ocr model using: python download_models.py OCR \n\n";

// Command-line keys to parse the input arguments
string keys =
    "{ help  h                        |                     | Print help message. }"
    "{ input i                        |      right.jpg      | Path to an input image. }"
    "{ @alias                         |                     | An alias name of model to extract preprocessing parameters from models.yml file. }"
    "{ zoo                            |  ../dnn/models.yml  | An optional path to file with preprocessing parameters }"
    "{ ocr_model                      |                     | Path to a binary .onnx model for recognition. }"
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
static void processFrame(
    const Mat& frame,
    const vector<vector<Point>>& detResults,
    String ocr_model,
    bool imreadRGB,
    Mat& board,
    int fontSize,
    int fontWeight,
    vector<String> vocabulary
);

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

    keys += genPreprocArguments(modelName, zooFile, "");
    keys += genPreprocArguments(modelName, zooFile, "ocr_");
    parser = CommandLineParser(argc, argv, keys);
    parser.about(about);

    // Parsing command-line arguments
    String sha1 = parser.get<String>("sha1");
    String ocr_sha1 = parser.get<String>("ocr_sha1");
    String detModelPath = findModel(parser.get<String>("model"), sha1);
    String ocr = findModel(parser.get<String>("ocr_model"), ocr_sha1);
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
    Mat board(frame.size(), frame.type(), Scalar(255, 255, 255));
    int stdSize = 20;
    int stdWeight = 400;
    int stdImgSize = 512;
    int imgWidth = min(frame.rows, frame.cols);
    int size = (stdSize*imgWidth)/stdImgSize;
    int weight = (stdWeight*imgWidth)/stdImgSize;

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

    processFrame(frame, detResults, ocr, imreadRGB, board, size, weight, vocabulary);
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

void processFrame(
    const Mat& frame,
    const vector<vector<Point>>& detResults,
    String ocr_model,
    bool imreadRGB,
    Mat& board,
    int fontSize,
    int fontWeight,
    vector<String> vocabulary
) {
    if (detResults.size() > 0) {
        // Text Recognition
        Mat recInput;
        if (!imreadRGB) {
            cvtColor(frame, recInput, cv::COLOR_BGR2GRAY);
        } else {
            recInput = frame;
        }

        vector<vector<Point>> contours;
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

            if(!ocr_model.empty()){
                TextRecognitionModel recognizer(ocr_model);
                recognizer.setVocabulary(vocabulary);
                recognizer.setDecodeType("CTC-greedy");

                // Setting input parameters for the recognition model
                double recScale = 1.0 / 127.5;
                Scalar recMean = Scalar(127.5);
                Size recInputSize = Size(100, 32);
                recognizer.setInputParams(recScale, recInputSize, recMean);
                // Recognizing text from the cropped image
                string recognitionResult = recognizer.recognize(cropped);
                cout << i << ": '" << recognitionResult << "'" << endl;

                // Displaying the recognized text on the image
                FontFace sans("sans");
                // Displaying the recognized text on the image
                putText(board, recognitionResult, Point(detResults[i][1].x, detResults[i][0].y), Scalar(0, 0, 0), sans, fontSize, fontWeight);
            }
            else{
                cout << "[WARN] Please pass the path to the ocr model using --ocr_model to get the recognised text." << endl;
            }
        }
        // Drawing detected text regions on the image
        polylines(board, contours, true, Scalar(200, 255, 200), 1);
        polylines(frame, contours, true, Scalar(0, 255, 0), 1);
    } else {
        cout << "No Text Detected." << endl;
    }

    // Displaying the final image with detected and recognized text
    Mat stacked;
    hconcat(frame, board, stacked);
    imshow("Text Detection and Recognition", stacked);
    waitKey(0);
}
