/*
    Text detection model: https://github.com/argman/EAST
    Download link: https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1

    Text recognition models can be downloaded directly here:
    Download link: https://drive.google.com/drive/folders/1cTbQ3nuZG-EKWak6emD_s8_hHXWz7lAr?usp=sharing
    and doc/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.markdown

    How to convert from pb to onnx:
    Using classes from here: https://github.com/meijieru/crnn.pytorch/blob/master/models/crnn.py
    import torch
    from models.crnn import CRNN
    model = CRNN(32, 1, 37, 256)
    model.load_state_dict(torch.load('crnn.pth'))
    dummy_input = torch.randn(1, 1, 32, 100)
    torch.onnx.export(model, dummy_input, "crnn.onnx", verbose=True)

    For more information, please refer to doc/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.markdown and doc/tutorials/dnn/dnn_OCR/dnn_OCR.markdown
*/
#include <iostream>
#include <fstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace cv::dnn;

const char* keys =
    "{ help  h              | | Print help message. }"
    "{ input i              | | Path to input image or video file. Skip this argument to capture frames from a camera.}"
    "{ detModel dmp         | | Path to a binary .pb file contains trained detector network.}"
    "{ width                | 320 | Preprocess input image by resizing to a specific width. It should be a multiple of 32. }"
    "{ height               | 320 | Preprocess input image by resizing to a specific height. It should be a multiple of 32. }"
    "{ thr                  | 0.5 | Confidence threshold. }"
    "{ nms                  | 0.4 | Non-maximum suppression threshold. }"
    "{ recModel rmp         | | Path to a binary .onnx file contains trained CRNN text recognition model. "
        "Download links are provided in doc/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.markdown}"
    "{ RGBInput rgb         |0| 0: imread with flags=IMREAD_GRAYSCALE; 1: imread with flags=IMREAD_COLOR. }"
    "{ vocabularyPath vp    | alphabet_36.txt | Path to benchmarks for evaluation. "
        "Download links are provided in doc/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.markdown}";

void fourPointsTransform(const Mat& frame, const Point2f vertices[], Mat& result);

int main(int argc, char** argv)
{
    // Parse command line arguments.
    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run TensorFlow implementation (https://github.com/argman/EAST) of "
                 "EAST: An Efficient and Accurate Scene Text Detector (https://arxiv.org/abs/1704.03155v2)");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    float confThreshold = parser.get<float>("thr");
    float nmsThreshold = parser.get<float>("nms");
    int width = parser.get<int>("width");
    int height = parser.get<int>("height");
    int imreadRGB = parser.get<int>("RGBInput");
    String detModelPath = parser.get<String>("detModel");
    String recModelPath = parser.get<String>("recModel");
    String vocPath = parser.get<String>("vocabularyPath");

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    // Load networks.
    CV_Assert(!detModelPath.empty() && !recModelPath.empty());
    TextDetectionModel_EAST detector(detModelPath);
    detector.setConfidenceThreshold(confThreshold)
            .setNMSThreshold(nmsThreshold);

    TextRecognitionModel recognizer(recModelPath);

    // Load vocabulary
    CV_Assert(!vocPath.empty());
    std::ifstream vocFile;
    vocFile.open(samples::findFile(vocPath));
    CV_Assert(vocFile.is_open());
    String vocLine;
    std::vector<String> vocabulary;
    while (std::getline(vocFile, vocLine)) {
        vocabulary.push_back(vocLine);
    }
    recognizer.setVocabulary(vocabulary);
    recognizer.setDecodeType("CTC-greedy");

    // Parameters for Recognition
    double recScale = 1.0 / 127.5;
    Scalar recMean = Scalar(127.5, 127.5, 127.5);
    Size recInputSize = Size(100, 32);
    recognizer.setInputParams(recScale, recInputSize, recMean);

    // Parameters for Detection
    double detScale = 1.0;
    Size detInputSize = Size(width, height);
    Scalar detMean = Scalar(123.68, 116.78, 103.94);
    bool swapRB = true;
    detector.setInputParams(detScale, detInputSize, detMean, swapRB);

    // Open a video file or an image file or a camera stream.
    VideoCapture cap;
    bool openSuccess = parser.has("input") ? cap.open(parser.get<String>("input")) : cap.open(0);
    CV_Assert(openSuccess);

    static const std::string kWinName = "EAST: An Efficient and Accurate Scene Text Detector";

    Mat frame;
    while (waitKey(1) < 0)
    {
        cap >> frame;
        if (frame.empty())
        {
            waitKey();
            break;
        }

        std::cout << frame.size << std::endl;

        // Detection
        std::vector< std::vector<Point> > detResults;
        detector.detect(frame, detResults);
        Mat frame2 = frame.clone();
        if (detResults.size() > 0) {
            // Text Recognition
            Mat recInput;
            if (!imreadRGB) {
                cvtColor(frame, recInput, cv::COLOR_BGR2GRAY);
            } else {
                recInput = frame;
            }
            std::vector< std::vector<Point> > contours;
            for (uint i = 0; i < detResults.size(); i++)
            {
                const auto& quadrangle = detResults[i];
                CV_CheckEQ(quadrangle.size(), (size_t)4, "");

                contours.emplace_back(quadrangle);

                std::vector<Point2f> quadrangle_2f;
                for (int j = 0; j < 4; j++)
                    quadrangle_2f.emplace_back(quadrangle[j]);

                Mat cropped;
                fourPointsTransform(recInput, &quadrangle_2f[0], cropped);

                std::string recognitionResult = recognizer.recognize(cropped);
                std::cout << i << ": '" << recognitionResult << "'" << std::endl;

                putText(frame2, recognitionResult, quadrangle[3], FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 2);
            }
            polylines(frame2, contours, true, Scalar(0, 255, 0), 2);
        }
        imshow(kWinName, frame2);
    }
    return 0;
}

void fourPointsTransform(const Mat& frame, const Point2f vertices[], Mat& result)
{
    const Size outputSize = Size(100, 32);

    Point2f targetVertices[4] = {
        Point(0, outputSize.height - 1),
        Point(0, 0), Point(outputSize.width - 1, 0),
        Point(outputSize.width - 1, outputSize.height - 1)
    };
    Mat rotationMatrix = getPerspectiveTransform(vertices, targetVertices);

    warpPerspective(frame, result, rotationMatrix, outputSize);
}
