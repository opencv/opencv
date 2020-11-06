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
        "{ unclipRatio ratio                |2.0| unclip ratio. }"
        "{ vocabularyPath vp                | | Path to benchmarks for evaluation. "
            "Download links are provided in doc/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.markdown}";

void fourPointsTransform(const Mat& frame, const Point2f vertices[], Mat& result);
bool sortPts(const Point& p1, const Point& p2);

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
    recognizer.setDecodeType("CTC-greedy");

    // Parameters for Detection
    double detScale = 1.0 / 255.0;
    Size detInputSize = Size(width, height);
    Scalar detMean = Scalar(122.67891434, 116.66876762, 104.00698793);
    detector.setInputParams(detScale, detInputSize, detMean);

    // Parameters for Recognition
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
    detector.detect(frame, detResults, binThresh, polyThresh, unclipRatio, maxCandidates);

    if (detResults.size() > 0) {
        // Text Recognition
        Mat recInput;
        if (!imreadRGB) {
            cvtColor(frame, recInput, cv::COLOR_BGR2GRAY);
        } else {
            recInput = frame;
        }
        for (uint i = 0; i < detResults.size(); i++)
        {
            std::vector<Point> box = detResults[i];

            // Sort the points
            std::sort(box.begin(), box.end(), sortPts);
            int index[4] = {0, 1, 2, 3};
            if (box[1].y > box[0].y) {
                index[0] = 0;
                index[3] = 1;
            } else {
                index[0] = 1;
                index[3] = 0;
            }
            if (box[3].y > box[2].y) {
                index[1] = 2;
                index[2] = 3;
            } else {
                index[1] = 3;
                index[2] = 2;
            }
            Point2f vertices[4] = {box[index[3]], box[index[0]],
                                   box[index[1]], box[index[2]]};

            // Transform and Crop
            Mat cropped;
            fourPointsTransform(recInput, vertices, cropped);

            String recResult = recognizer.recognize(cropped);

            putText(frame, recResult, vertices[3], FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        }
        polylines(frame, detResults, true, Scalar(0, 255, 0), 2);
    } else {
        std::cout << "No Text Detected." << std::endl;
    }
    imshow(winName, frame);
    waitKey();

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

bool sortPts(const Point& p1, const Point& p2)
{
    return p1.x < p2.x;
}
