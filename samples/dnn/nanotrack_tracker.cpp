// NanoTrack
// Link to original inference code: https://github.com/HonglinChu/NanoTrack
// Link to original training repo: https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack
// backBone model: https://github.com/HonglinChu/SiamTrackers/blob/master/NanoTrack/models/onnx/nanotrack_backbone_sim.onnx
// headNeck model: https://github.com/HonglinChu/SiamTrackers/blob/master/NanoTrack/models/onnx/nanotrack_head_sim.onnx

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
        "{ backbone    | backbone.onnx | Path to onnx model of backbone.onnx}"
        "{ headneck    | headneck.onnx | Path to onnx model of headneck.onnx }"
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
    std::string backbone = parser.get<String>("backbone");
    std::string headneck = parser.get<String>("headneck");
    int backend = parser.get<int>("backend");
    int target = parser.get<int>("target");

    Ptr<TrackerNano> tracker;
    try
    {
        TrackerNano::Params params;
        params.backbone = samples::findFile(backbone);
        params.neckhead = samples::findFile(headneck);
        params.backend = backend;
        params.target = target;
        tracker = TrackerNano::create(params);
    }
    catch (const cv::Exception& ee)
    {
        std::cerr << "Exception: " << ee.what() << std::endl;
        std::cout << "Can't load the network by using the following files:" << std::endl;
        std::cout << "backbone : " << backbone << std::endl;
        std::cout << "headneck : " << headneck << std::endl;
        return 2;
    }

    const std::string winName = "NanoTrack";
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

        std::cout << "frame " << count <<
            ": predicted score=" << score <<
            "  rect=" << rect <<
            "  time=" << tickMeter.getTimeMilli() << "ms" <<
            std::endl;

        Mat render_image = image.clone();

        if (ok)
        {
            rectangle(render_image, rect, Scalar(0, 255, 0), 2);

            std::string timeLabel = format("Inference time: %.2f ms", tickMeter.getTimeMilli());
            std::string scoreLabel = format("Score: %f", score);
            putText(render_image, timeLabel, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
            putText(render_image, scoreLabel, Point(0, 35), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        }

        imshow(winName, render_image);

        tickMeter.reset();

        int c = waitKey(1);
        if (c == 27 /*ESC*/)
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
