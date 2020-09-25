#include <iostream>
#include <fstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>

using namespace cv;
using namespace cv::dnn;

std::string keys =
        "{ help  h                          | | Print help message. }"
        "{ inputImage i                     | | Path to an input image. Skip this argument to capture frames from a camera. }"
        "{ detModelPath dmp                 | | Path to a binary .onnx model for detection. "
            "Download links are provided in doc/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.markdown}"
        "{ recModelPath rmp                 | | Path to a binary .onnx model for recognition. "
            "Download links are provided in doc/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.markdown}"
        "{ inputHeight ih                   |736| image height of the model input. }"
        "{ inputWidth iw                    |1280| image width of the model input. }"
        "{ RGBInput rgb                     |0| 0: imread with flags=IMREAD_GRAYSCALE; 1: imread with flags=IMREAD_COLOR. }"
        "{ binaryThreshold bt               |0.3| Confidence threshold of the binary map. }"
        "{ polygonThreshold pt              |0.5| Confidence threshold of polygons. }"
        "{ maxCandidate max                 |200| Max candidates of polygons. }"
        "{ outputType type                  |0| The output type of detected text: 0: multi-oriented quadrilateral; 1: polygon }"
        "{ unclipRatio ratio                |2.0| unclip ratio. }"
        "{ vocabularyPath vp                | | Path to benchmarks for evaluation. "
            "Download links are provided in doc/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.markdown}";

int main(int argc, char** argv)
{
    // Parse arguments
    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run an end-to-end inference sample of textDetectionModel and textRecognitionModel APIs\n"
                 "Use -h for more information");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    float binThresh = parser.get<float>("binaryThreshold");
    float polyThresh = parser.get<float>("polygonThreshold");
    uint maxCandidates = parser.get<uint>("maxCandidate");
    String detModelPath = parser.get<String>("detModelPath");
    String recModelPath = parser.get<String>("recModelPath");
    String vocPath = parser.get<String>("vocabularyPath");
    int outputType = parser.get<int>("outputType");
    double unclipRatio = parser.get<double>("unclipRatio");
    int height = parser.get<int>("inputHeight");
    int width = parser.get<int>("inputWidth");
    int imreadRGB = parser.get<int>("RGBInput");

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    // Load networks
    CV_Assert(!detModelPath.empty());
    TextDetectionModel detector(detModelPath);

    CV_Assert(!recModelPath.empty());
    TextRecognitionModel recognizer(recModelPath);

    // Load vocabulary
    CV_Assert(!vocPath.empty());
    std::ifstream vocFile;
    vocFile.open(vocPath);
    CV_Assert(vocFile.is_open());
    String vocLine;
    std::vector<String> vocabulary;
    while (std::getline(vocFile, vocLine)) {
        vocabulary.push_back(vocLine);
    }
    recognizer.setVocabulary(vocabulary);

    // Parameters for Detection
    double detScale = 1.0 / 255.0;
    Size detInputSize = Size(width, height);
    Scalar detMean = Scalar(122.67891434, 116.66876762, 104.00698793);
    detector.setInputParams(detScale, detInputSize, detMean);

    // Parameters for Recognition
    const String decodeType = "CTC-greedy";
    double recScale = 1.0 / 127.5;
    Scalar recMean = Scalar(127.5);
    Size recInputSize = Size(100, 32);
    recognizer.setInputParams(recScale, recInputSize, recMean);

    // Create a window
    static const std::string winName = "Text_Spotting";
    namedWindow(winName, WINDOW_NORMAL);

    // Input data
    Mat frame = imread(samples::findFile(parser.get<String>("inputImage")));

    // Inference
    std::vector<std::vector<Point>> detResults;
    detector.detect(frame, detResults, outputType, binThresh, polyThresh, unclipRatio, maxCandidates);
    if (detResults.size() == 0) {
        std::cout << "No Text Detected." << std::endl;
        return 0;
    }

    std::vector<String> recResults;
    Mat recInput;
    if (!imreadRGB) {
        cvtColor(frame, recInput, cv::COLOR_BGR2GRAY);
    } else {
        recInput = frame;
    }

    // Recognize all ROIs at the same time
    recognizer.recognize(recInput, decodeType, recResults, detResults);

    // Visualization
    for (uint i = 0; i < detResults.size(); i++) {
        polylines(frame, detResults[i], true, Scalar(0, 255, 0), 2);
        putText(frame, recResults[i], detResults[i][3], FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 0, 255), 2);
    }
    imshow(winName, frame);
    waitKey();

    return 0;
}
