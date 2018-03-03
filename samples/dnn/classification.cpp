#include <fstream>
#include <iostream>
#include <sstream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

const char* keys =
    "{ help  h     | | Print help message. }"
    "{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera.}"
    "{ model m     | | Path to a binary file of model contains trained weights. "
                      "It could be a file with extensions .caffemodel (Caffe), "
                      ".pb (TensorFlow), .t7 or .net (Torch), .weights (Darknet) }"
    "{ config c    | | Path to a text file of model contains network configuration. "
                      "It could be a file with extensions .prototxt (Caffe), .pbtxt (TensorFlow), .cfg (Darknet) }"
    "{ framework f | | Optional name of an origin framework of the model. Detect it automatically if it does not set. }"
    "{ classes     | | Optional path to a text file with names of classes. }"
    "{ mean        | | Preprocess input image by subtracting mean values. Mean values should be in BGR order and delimited by spaces. }"
    "{ scale       |  1 | Preprocess input image by multiplying on a scale factor. }"
    "{ width       | -1 | Preprocess input image by resizing to a specific width. }"
    "{ height      | -1 | Preprocess input image by resizing to a specific height. }"
    "{ rgb         |    | Indicate that model works with RGB input images instead BGR ones. }"
    "{ backend     |  0 | Choose one of computation backends: "
                         "0: default C++ backend, "
                         "1: Halide language (http://halide-lang.org/), "
                         "2: Intel's Deep Learning Inference Engine (https://software.seek.intel.com/deep-learning-deployment)}"
    "{ target      |  0 | Choose one of target computation devices: "
                         "0: CPU target (by default),"
                         "1: OpenCL }";

using namespace cv;
using namespace dnn;

std::vector<std::string> classes;

Net readNet(const std::string& model, const std::string& config = "", const std::string& framework = "");

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run classification deep learning networks using OpenCV.");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    float scale = parser.get<float>("scale");
    bool swapRB = parser.get<bool>("rgb");
    int inpWidth = parser.get<int>("width");
    int inpHeight = parser.get<int>("height");

    // Parse mean values.
    Scalar mean;
    if (parser.has("mean"))
    {
        std::istringstream meanStr(parser.get<String>("mean"));
        std::vector<float> meanValues;
        float val;
        while (meanStr >> val)
            meanValues.push_back(val);
        CV_Assert(meanValues.size() == 3);
        mean = Scalar(meanValues[0], meanValues[1], meanValues[2]);
    }

    // Open file with classes names.
    if (parser.has("classes"))
    {
        std::string file = parser.get<String>("classes");
        std::ifstream ifs(file.c_str());
        if (!ifs.is_open())
            CV_Error(Error::StsError, "File " + file + " not found");
        std::string line;
        while (std::getline(ifs, line))
        {
            classes.push_back(line);
        }
    }

    // Load a model.
    CV_Assert(parser.has("model"));
    Net net = readNet(parser.get<String>("model"), parser.get<String>("config"), parser.get<String>("framework"));
    net.setPreferableBackend(parser.get<int>("backend"));
    net.setPreferableTarget(parser.get<int>("target"));

    // Create a window
    static const std::string kWinName = "Deep learning image classification in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);

    // Open a video file or an image file or a camera stream.
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(parser.get<String>("input"));
    else
        cap.open(0);

    // Process frames.
    Mat frame, blob;
    while (waitKey(1) < 0)
    {
        cap >> frame;
        if (frame.empty())
        {
            waitKey();
            break;
        }

        // Create a 4D blob from a frame.
        blobFromImage(frame, blob, scale, Size(inpWidth, inpHeight), mean, swapRB, false);

        // Run a model.
        net.setInput(blob);
        Mat out = net.forward();
        out = out.reshape(1, 1);

        // Get a class with a highest score.
        Point classIdPoint;
        double confidence;
        minMaxLoc(out, 0, &confidence, 0, &classIdPoint);
        int classId = classIdPoint.x;

        // Put efficiency information.
        std::vector<double> layersTimes;
        double t = net.getPerfProfile(layersTimes);
        std::string label = format("Inference time: %.2f", t * 1000 / getTickFrequency());
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

        // Print predicted class.
        label = format("%s: %.4f", (classes.empty() ? format("Class #%d", classId).c_str() :
                                                      classes[classId].c_str()),
                                   confidence);
        putText(frame, label, Point(0, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

        imshow(kWinName, frame);
    }
    return 0;
}

Net readNet(const std::string& model, const std::string& config, const std::string& framework)
{
    std::string modelExt = model.substr(model.rfind('.'));
    if (framework == "caffe" || modelExt == ".caffemodel")
        return readNetFromCaffe(config, model);
    else if (framework == "tensorflow" || modelExt == ".pb")
        return readNetFromTensorflow(model, config);
    else if (framework == "torch" || modelExt == ".t7" || modelExt == ".net")
        return readNetFromTorch(model);
    else if (framework == "darknet" || modelExt == ".weights")
        return readNetFromDarknet(config, model);
    else
        CV_Error(Error::StsError, "Cannot determine an origin framework of model from file " + model);
    return Net();
}
