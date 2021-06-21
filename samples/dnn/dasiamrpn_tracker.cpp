// DaSiamRPN tracker.
// Original paper: https://arxiv.org/abs/1808.06048
// Link to original repo: https://github.com/foolwood/DaSiamRPN
// Links to onnx models:
// - network:     https://www.dropbox.com/s/rr1lk9355vzolqv/dasiamrpn_model.onnx?dl=0
// - kernel_r1:   https://www.dropbox.com/s/999cqx5zrfi7w4p/dasiamrpn_kernel_r1.onnx?dl=0
// - kernel_cls1: https://www.dropbox.com/s/qvmtszx5h339a0w/dasiamrpn_kernel_cls1.onnx?dl=0

#include <iostream>
#include <cmath>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace cv::dnn;

const char *keys =
        "{ help     h  |   | Print help message }"
        "{ input    i  |   | Full path to input video folder, the specific camera index. (empty for camera 0) }"
        "{ net         | dasiamrpn_model.onnx | Path to onnx model of net}"
        "{ kernel_cls1 | dasiamrpn_kernel_cls1.onnx | Path to onnx model of kernel_r1 }"
        "{ kernel_r1   | dasiamrpn_kernel_r1.onnx | Path to onnx model of kernel_cls1 }"
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
    std::string kernel_cls1 = parser.get<String>("kernel_cls1");
    std::string kernel_r1 = parser.get<String>("kernel_r1");
    int backend = parser.get<int>("backend");
    int target = parser.get<int>("target");

    Ptr<TrackerDaSiamRPN> tracker;
    try
    {
        TrackerDaSiamRPN::Params params;
        params.model = samples::findFile(net);
        params.kernel_cls1 = samples::findFile(kernel_cls1);
        params.kernel_r1 = samples::findFile(kernel_r1);
        params.backend = backend;
        params.target = target;
        tracker = TrackerDaSiamRPN::create(params);
    }
    catch (const cv::Exception& ee)
    {
        std::cerr << "Exception: " << ee.what() << std::endl;
        std::cout << "Can't load the network by using the following files:" << std::endl;
        std::cout << "siamRPN : " << net << std::endl;
        std::cout << "siamKernelCL1 : " << kernel_cls1 << std::endl;
        std::cout << "siamKernelR1 : " << kernel_r1 << std::endl;
        return 2;
    }

    const std::string winName = "DaSiamRPN";
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
