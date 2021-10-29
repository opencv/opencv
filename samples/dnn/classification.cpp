#include <fstream>
#include <sstream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "common.hpp"

std::string keys =
    "{ help  h          | | Print help message. }"
    "{ @alias           | | An alias name of model to extract preprocessing parameters from models.yml file. }"
    "{ zoo              | models.yml | An optional path to file with preprocessing parameters }"
    "{ input i          | | Path to input image or video file. Skip this argument to capture frames from a camera.}"
    "{ initial_width    | 0 | Preprocess input image by initial resizing to a specific width.}"
    "{ initial_height   | 0 | Preprocess input image by initial resizing to a specific height.}"
    "{ std              | 0.0 0.0 0.0 | Preprocess input image by dividing on a standard deviation.}"
    "{ crop             | false | Preprocess input image by center cropping.}"
    "{ framework f      | | Optional name of an origin framework of the model. Detect it automatically if it does not set. }"
    "{ classes          | | Optional path to a text file with names of classes. }"
    "{ backend          | 0 | Choose one of computation backends: "
                            "0: automatically (by default), "
                            "1: Halide language (http://halide-lang.org/), "
                            "2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                            "3: OpenCV implementation, "
                            "4: VKCOM, "
                            "5: CUDA },"
    "{ target           | 0 | Choose one of target computation devices: "
                            "0: CPU target (by default), "
                            "1: OpenCL, "
                            "2: OpenCL fp16 (half-float precision), "
                            "3: VPU, "
                            "4: Vulkan, "
                            "6: CUDA, "
                            "7: CUDA fp16 (half-float preprocess) }";

using namespace cv;
using namespace dnn;

std::vector<std::string> classes;

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);

    const std::string modelName = parser.get<String>("@alias");
    const std::string zooFile = parser.get<String>("zoo");

    keys += genPreprocArguments(modelName, zooFile);

    parser = CommandLineParser(argc, argv, keys);
    parser.about("Use this script to run classification deep learning networks using OpenCV.");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    int rszWidth = parser.get<int>("initial_width");
    int rszHeight = parser.get<int>("initial_height");
    float scale = parser.get<float>("scale");
    Scalar mean = parser.get<Scalar>("mean");
    Scalar std = parser.get<Scalar>("std");
    bool swapRB = parser.get<bool>("rgb");
    bool crop = parser.get<bool>("crop");
    int inpWidth = parser.get<int>("width");
    int inpHeight = parser.get<int>("height");
    String model = findFile(parser.get<String>("model"));
    String config = findFile(parser.get<String>("config"));
    String framework = parser.get<String>("framework");
    int backendId = parser.get<int>("backend");
    int targetId = parser.get<int>("target");

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

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }
    CV_Assert(!model.empty());

    //! [Read and initialize network]
    Net net = readNet(model, config, framework);
    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);
    //! [Read and initialize network]

    // Create a window
    static const std::string kWinName = "Deep learning image classification in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);

    //! [Open a video file or an image file or a camera stream]
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(parser.get<String>("input"));
    else
        cap.open(0);
    //! [Open a video file or an image file or a camera stream]

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

        if (rszWidth != 0 && rszHeight != 0)
        {
            resize(frame, frame, Size(rszWidth, rszHeight));
        }

        //! [Create a 4D blob from a frame]
        blobFromImage(frame, blob, scale, Size(inpWidth, inpHeight), mean, swapRB, crop);

        // Check std values.
        if (std.val[0] != 0.0 && std.val[1] != 0.0 && std.val[2] != 0.0)
        {
            // Divide blob by std.
            divide(blob, std, blob);
        }
        //! [Create a 4D blob from a frame]

        //! [Set input blob]
        net.setInput(blob);
        //! [Set input blob]
        //! [Make forward pass]
        Mat prob = net.forward();
        //! [Make forward pass]

        //! [Get a class with a highest score]
        Point classIdPoint;
        double confidence;
        minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
        int classId = classIdPoint.x;
        //! [Get a class with a highest score]

        // Put efficiency information.
        std::vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        std::string label = format("Inference time: %.2f ms", t);
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
