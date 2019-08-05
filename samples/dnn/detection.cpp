#include <fstream>
#include <sstream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "common.hpp"

std::string keys =
    "{ help  h     | | Print help message. }"
    "{ @alias      | | An alias name of model to extract preprocessing parameters from models.yml file. }"
    "{ zoo         | models.yml | An optional path to file with preprocessing parameters }"
    "{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera.}"
    "{ classes     | | Optional path to a text file with names of classes. }"
    "{ confidence  | 0.5 | Optional threshold to discard low confidence boxes.}"
    "{ nms_thr     | 0.0 | Optional IOU threshold to discard highly overlapping boxes.}"
    "{ backend     | 0 | Choose one of computation backends: "
                        "0: automatically (by default), "
                        "1: Halide language (http://halide-lang.org/), "
                        "2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                        "3: OpenCV implementation }"
    "{ target      | 0 | Choose one of target computation devices: "
                        "0: CPU target (by default), "
                        "1: OpenCL, "
                        "2: OpenCL fp16 (half-float precision), "
                        "3: VPU }";

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

    float scale = parser.get<float>("scale");
    Scalar mean = parser.get<Scalar>("mean");
    bool swapRB = parser.get<bool>("rgb");
    int inpWidth = parser.get<int>("width");
    int inpHeight = parser.get<int>("height");
    String model = findFile(parser.get<String>("model"));
    String config = findFile(parser.get<String>("config"));
    int backendId = parser.get<int>("backend");
    int targetId = parser.get<int>("target");
    float confThreshold = parser.get<float>("confidence");
    float nmsThreshold = parser.get<float>("nms_thr");

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
    // YOLOv3
    // https://pjreddie.com/media/files/yolov3.weights
    DetectionModel net(model, config);
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

    //! [Set Input Parameters]
    net.setInputParams(scale, Size(inpHeight, inpWidth), mean, swapRB);
    //! [Set Input Parameters]

    // Process frames.
    Mat frame;
    while (waitKey(1) < 0)
    {
        cap >> frame;
        if (frame.empty())
        {
            waitKey();
            break;
        }

        //! [Network Forward pass]
        std::vector<Rect> boxes;
        std::vector<int> classIds;
        std::vector<float> confidences;
        net.detect(frame, classIds, confidences, boxes, confThreshold, nmsThreshold);
        //! [Network Forward pass]

        //! [Iterate over every predicted box and draw them on the image with the predicted class and confidence on top]
        std::vector<Rect2d> boxesDouble(boxes.size());
        std::stringstream ss;

        for (uint i = 0; i < boxes.size(); i++) {
            ss << classIds[i];
            ss << ": ";
            ss << confidences[i];
            boxesDouble[i] = boxes[i];
            rectangle(frame, boxesDouble[i], Scalar(0, 0, 255), 1, 8, 0);
            putText(frame, ss.str(), Point(boxes[i].x, boxes[i].y), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0,0,0), 2);
            ss.str("");
        }
        //! [Iterate over every predicted box and draw them on the image with the predicted class and confidence on top]

        // Put efficiency information.
        std::vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        std::string label = format("Inference time: %.2f ms", t);
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

        imshow(kWinName, frame);
    }

    return 0;
}
