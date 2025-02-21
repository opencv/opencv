// VitTracker
// model: https://github.com/opencv/opencv_zoo/tree/main/models/object_tracking_vittrack

#include <iostream>
#include <cmath>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn_algorithms.hpp>

using namespace cv;
using namespace cv::dnn;

const char *keys =
        "{ help     h  |   | Print help message }"
        "{ input    i  |   | Full path to input video folder, the specific camera index. (empty for camera 0) }"
        "{ net    | vitTracker.onnx | Path to onnx model of vitTracker.onnx}"
        "{ tracking_score_threshold t | 0.3 | Tracking score threshold. If a bbox of score >= 0.3, it is considered as found }"
        "{ backend     | 0 | Choose one of computation backends: "
                            "0: automatically (by default), "
                            "1: Halide language (http://halide-lang.org/), "
                            "2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                            "3: OpenCV implementation, "
                            "4: VKCOM, "
                            "5: CUDA },"
        "{ target      | 0 | Choose one of target computation devices: "
                            "0: CPU target (by default), "
                            "1: OpenCL, "
                            "2: OpenCL fp16 (half-float precision), "
                            "3: VPU, "
                            "4: Vulkan, "
                            "6: CUDA, "
                            "7: CUDA fp16 (half-float preprocess) }"
;

static
int run(int argc, char** argv)
{
    // Parse command line arguments.
    CommandLineParser parser(argc, argv, keys);

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    std::string inputName = parser.get<String>("input");
    std::string net = parser.get<String>("net");
    int backend = parser.get<int>("backend");
    int target = parser.get<int>("target");
    float tracking_score_threshold = parser.get<float>("tracking_score_threshold");

    Ptr<TrackerVit> tracker;
    try
    {
        TrackerVit::Params params;
        params.net = samples::findFile(net);
        params.backend = backend;
        params.target = target;
        params.tracking_score_threshold = tracking_score_threshold;
        tracker = TrackerVit::create(params);
    }
    catch (const cv::Exception& ee)
    {
        std::cerr << "Exception: " << ee.what() << std::endl;
        std::cout << "Can't load the network by using the following files:" << std::endl;
        std::cout << "net : " << net << std::endl;
        return 2;
    }

    const std::string winName = "vitTracker";
    namedWindow(winName, WINDOW_AUTOSIZE);

    // Open a video file or an image file or a camera stream.
    VideoCapture cap;

    if (inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1))
    {
        int c = inputName.empty() ? 0 : inputName[0] - '0';
        std::cout << "Trying to open camera #" << c << " ..." << std::endl;
        if (!cap.open(c))
        {
            std::cout << "Capture from camera #" << c << " didn't work. Specify -i=<video> parameter to read from video file" << std::endl;
            return 2;
        }
    }
    else if (inputName.size())
    {
        inputName = samples::findFileOrKeep(inputName);
        if (!cap.open(inputName))
        {
            std::cout << "Could not open: " << inputName << std::endl;
            return 2;
        }
    }

    // Read the first image.
    Mat image;
    cap >> image;
    if (image.empty())
    {
        std::cerr << "Can't capture frame!" << std::endl;
        return 2;
    }

    Mat image_select = image.clone();
    putText(image_select, "Select initial bounding box you want to track.", Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
    putText(image_select, "And Press the ENTER key.", Point(0, 35), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

    Rect selectRect = selectROI(winName, image_select);
    std::cout << "ROI=" << selectRect << std::endl;
    if (selectRect.empty())
    {
        std::cerr << "Invalid ROI!" << std::endl;
        return 2;
    }

    tracker->init(image, selectRect);

    TickMeter tickMeter;

    for (int count = 0; ; ++count)
    {
        cap >> image;
        if (image.empty())
        {
            std::cerr << "Can't capture frame " << count << ". End of video stream?" << std::endl;
            break;
        }

        Rect rect;

        tickMeter.start();
        bool ok = tracker->update(image, rect);
        tickMeter.stop();

        float score = tracker->getTrackingScore();

        std::cout << "frame " << count;
        if (ok) {
            std::cout << ": predicted score=" << score <<
                         "\trect=" << rect <<
                         "\ttime=" << tickMeter.getTimeMilli() << "ms" << std::endl;

            rectangle(image, rect, Scalar(0, 255, 0), 2);

            std::string timeLabel = format("Inference time: %.2f ms", tickMeter.getTimeMilli());
            std::string scoreLabel = format("Score: %f", score);
            putText(image, timeLabel, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
            putText(image, scoreLabel, Point(0, 35), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        } else {
            std::cout << ": target lost" << std::endl;
            putText(image, "Target lost", Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
        }

        imshow(winName, image);

        tickMeter.reset();

        int c = waitKey(1);
        if (c == 27 /*ESC*/ || c == 'q' || c == 'Q')
            break;
    }

    std::cout << "Exit" << std::endl;
    return 0;
}


int main(int argc, char **argv)
{
    try
    {
        return run(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "FATAL: C++ exception: " << e.what() << std::endl;
        return 1;
    }
}
