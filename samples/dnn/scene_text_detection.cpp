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
        "{ modelPath mp                     | | Path to a binary .onnx file contains trained DB detector model. "
            "Download links are provided in doc/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.markdown}"
        "{ inputHeight ih                    |736| image height of the model input. It should be multiple by 32.}"
        "{ inputWidth iw                     |736| image width of the model input. It should be multiple by 32.}"
        "{ binaryThreshold bt               |0.3| Confidence threshold of the binary map. }"
        "{ polygonThreshold pt              |0.5| Confidence threshold of polygons. }"
        "{ maxCandidate max                 |200| Max candidates of polygons. }"
        "{ unclipRatio ratio                |2.0| unclip ratio. }"
        "{ evaluate e                       |false| false: predict with input images; true: evaluate on benchmarks. }"
        "{ evalDataPath edp                  | | Path to benchmarks for evaluation. "
            "Download links are provided in doc/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.markdown}";

static
void split(const std::string& s, char delimiter, std::vector<std::string>& elems)
{
    elems.clear();
    size_t prev_pos = 0;
    size_t pos = 0;
    while ((pos = s.find(delimiter, prev_pos)) != std::string::npos)
    {
        elems.emplace_back(s.substr(prev_pos, pos - prev_pos));
        prev_pos = pos + 1;
    }
    if (prev_pos < s.size())
        elems.emplace_back(s.substr(prev_pos, s.size() - prev_pos));
}

int main(int argc, char** argv)
{
    // Parse arguments
    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run the official PyTorch implementation (https://github.com/MhLiao/DB) of "
                 "Real-time Scene Text Detection with Differentiable Binarization (https://arxiv.org/abs/1911.08947)\n"
                 "The current version of this script is a variant of the original network without deformable convolution");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    float binThresh = parser.get<float>("binaryThreshold");
    float polyThresh = parser.get<float>("polygonThreshold");
    uint maxCandidates = parser.get<uint>("maxCandidate");
    String modelPath = parser.get<String>("modelPath");
    double unclipRatio = parser.get<double>("unclipRatio");
    int height = parser.get<int>("inputHeight");
    int width = parser.get<int>("inputWidth");

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    // Load the network
    CV_Assert(!modelPath.empty());
    TextDetectionModel_DB detector(modelPath);
    detector.setBinaryThreshold(binThresh)
            .setPolygonThreshold(polyThresh)
            .setUnclipRatio(unclipRatio)
            .setMaxCandidates(maxCandidates);

    double scale = 1.0 / 255.0;
    Size inputSize = Size(width, height);
    Scalar mean = Scalar(122.67891434, 116.66876762, 104.00698793);
    detector.setInputParams(scale, inputSize, mean);

    // Create a window
    static const std::string winName = "TextDetectionModel";

    if (parser.get<bool>("evaluate")) {
        // for evaluation
        String evalDataPath = parser.get<String>("evalDataPath");
        CV_Assert(!evalDataPath.empty());
        String testListPath = evalDataPath + "/test_list.txt";
        std::ifstream testList;
        testList.open(testListPath);
        CV_Assert(testList.is_open());

        // Create a window for showing groundtruth
        static const std::string winNameGT = "GT";

        String testImgPath;
        while (std::getline(testList, testImgPath)) {
            String imgPath = evalDataPath + "/test_images/" + testImgPath;
            std::cout << "Image Path: " << imgPath << std::endl;

            Mat frame = imread(samples::findFile(imgPath), IMREAD_COLOR);
            CV_Assert(!frame.empty());
            Mat src = frame.clone();

            // Inference
            std::vector<std::vector<Point>> results;
            detector.detect(frame, results);

            polylines(frame, results, true, Scalar(0, 255, 0), 2);
            imshow(winName, frame);

            // load groundtruth
            String imgName = testImgPath.substr(0, testImgPath.length() - 4);
            String gtPath = evalDataPath + "/test_gts/" + imgName + ".txt";
            // std::cout << gtPath << std::endl;
            std::ifstream gtFile;
            gtFile.open(gtPath);
            CV_Assert(gtFile.is_open());

            std::vector<std::vector<Point>> gts;
            String gtLine;
            while (std::getline(gtFile, gtLine)) {
                size_t splitLoc = gtLine.find_last_of(',');
                String text = gtLine.substr(splitLoc+1);
                if ( text == "###\r" || text == "1") {
                    // ignore difficult instances
                    continue;
                }
                gtLine = gtLine.substr(0, splitLoc);

                std::vector<std::string> v;
                split(gtLine, ',', v);

                std::vector<int> loc;
                std::vector<Point> pts;
                for (auto && s : v) {
                    loc.push_back(atoi(s.c_str()));
                }
                for (size_t i = 0; i < loc.size() / 2; i++) {
                    pts.push_back(Point(loc[2 * i], loc[2 * i + 1]));
                }
                gts.push_back(pts);
            }
            polylines(src, gts, true, Scalar(0, 255, 0), 2);
            imshow(winNameGT, src);

            waitKey();
        }
    } else {
        // Open an image file
        CV_Assert(parser.has("inputImage"));
        Mat frame = imread(samples::findFile(parser.get<String>("inputImage")));
        CV_Assert(!frame.empty());

        // Detect
        std::vector<std::vector<Point>> results;
        detector.detect(frame, results);

        polylines(frame, results, true, Scalar(0, 255, 0), 2);
        imshow(winName, frame);
        waitKey();
    }

    return 0;
}
