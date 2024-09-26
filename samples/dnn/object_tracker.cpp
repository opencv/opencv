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
#include "common.hpp"

using namespace cv;
using namespace std;
using namespace cv::dnn;

const string param_keys =
        "{ help     h    |                            | Print help message }"
        "{ @alias        |                            | An alias name of model to extract preprocessing parameters from models.yml file. }"
        "{ input    i    |                            | Full path to input video folder, the specific camera index. (empty for camera 0) }"
        "{ backbone      |         backbone.onnx      | Path to onnx model of backbone.onnx}"
        "{ headneck      |         headneck.onnx      | Path to onnx model of headneck.onnx }"
        "{ vit_net       |       vitTracker.onnx      | Path to onnx model of vitTracker.onnx}"
        "{ tracking_thrs |             0.3            | Tracking score threshold. If a bbox of score >= 0.3, it is considered as found }"
        "{ dasiamrpn_net |     dasiamrpn_model.onnx   | Path to onnx model of net}"
        "{ kernel_r1     |  dasiamrpn_kernel_r1.onnx  | Path to onnx model of kernel_cls1 }"
        "{ kernel_cls1   | dasiamrpn_kernel_cls1.onnx | Path to onnx model of kernel_r1 }";

const string backend_keys = format(
    "{ backend | default | Choose one of computation backends: "
                    "default: automatically (by default), "
                    "openvino: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                    "opencv: OpenCV implementation, "
                    "vkcom: VKCOM, "
                    "cuda: CUDA, "
                    "webnn: WebNN }");

const string target_keys = format(
    "{ target | cpu | Choose one of target computation devices: "
                    "cpu: CPU target (by default), "
                    "opencl: OpenCL, "
                    "opencl_fp16: OpenCL fp16 (half-float precision), "
                    "vpu: VPU, "
                    "vulkan: Vulkan, "
                    "cuda: CUDA, "
                    "cuda_fp16: CUDA fp16 (half-float preprocess) }");

string keys = param_keys + backend_keys + target_keys;

float getTrackingScore(Ptr<Tracker> tracker)
{
    // Try casting to TrackerDaSiamRPN
    if (Ptr<TrackerDaSiamRPN> trackerDaSiam = dynamic_pointer_cast<TrackerDaSiamRPN>(tracker))
    {
        return trackerDaSiam->getTrackingScore();
    }

    // Try casting to TrackerVit
    if (Ptr<TrackerVit> trackerVit = dynamic_pointer_cast<TrackerVit>(tracker))
    {
        return trackerVit->getTrackingScore();
    }

    // Try casting to TrackerVit
    if (Ptr<TrackerNano> trackerVit = dynamic_pointer_cast<TrackerNano>(tracker))
    {
        return trackerVit->getTrackingScore();
    }

    // If tracker type does not have getTrackingScore
    return -1;  // Return -1 or some other default value indicating no score available
}

static int trackObject(const string& windowName, Ptr<Tracker> tracker, const string& inputName)
{
    namedWindow(windowName, WINDOW_AUTOSIZE);

    // Open a video file or an image file or a camera stream.
    VideoCapture cap;

    if (inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1))
    {
        int c = inputName.empty() ? 0 : inputName[0] - '0';
        cout << "Trying to open camera #" << c << " ..." << endl;
        if (!cap.open(c))
        {
            cout << "Capture from camera #" << c << " didn't work. Specify -i=<video> parameter to read from video file" << endl;
            return 2;
        }
    }
    else if (inputName.size())
    {
        string filePath = findFile(inputName);
        if (!cap.open(filePath))
        {
            cout << "Could not open: " << inputName << endl;
            return 2;
        }
    }

    // Read the first image.
    Mat image;
    cap >> image;
    if (image.empty())
    {
        cerr << "Can't capture frame!" << endl;
        return 2;
    }

    Mat image_select = image.clone();
    putText(image_select, "Select initial bounding box you want to track.", Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
    putText(image_select, "And Press the ENTER key.", Point(0, 35), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

    Rect selectRect = selectROI(windowName, image_select);
    cout << "ROI=" << selectRect << endl;

    tracker->init(image, selectRect);

    TickMeter tickMeter;

    for (int count = 0; ; ++count)
    {
        cap >> image;
        if (image.empty())
        {
            cerr << "Can't capture frame " << count << ". End of video stream?" << endl;
            break;
        }

        Rect rect;

        tickMeter.start();
        bool ok = tracker->update(image, rect);
        tickMeter.stop();

        float score = getTrackingScore(tracker);

        Mat render_image = image.clone();

        if (ok)
        {
            rectangle(render_image, rect, Scalar(0, 255, 0), 2);

            string timeLabel = format("Inference time: %.2f ms", tickMeter.getTimeMilli());
            string scoreLabel = format("Score: %f", score);
            putText(render_image, timeLabel, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
            putText(render_image, scoreLabel, Point(0, 35), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        }

        imshow(windowName, render_image);

        tickMeter.reset();

        int c = waitKey(50);
        if (c == 27 /*ESC*/)
            exit(0);
    }
    return 0;
}

static int run(int argc, char** argv)
{
    // Parse command line arguments.
    CommandLineParser parser(argc, argv, keys);

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string inputName = parser.get<String>("input");
    string modelName = parser.get<String>("@alias");
    string net, headneck, backbone, kernel_cls1, kernel_r1;

    int backend = getBackendID(parser.get<String>("backend"));
    int target = getTargetID(parser.get<String>("target"));

    if (modelName == "vit"){
        net = parser.get<String>("vit_net");
        float tracking_score_threshold = parser.get<float>("tracking_thrs");

        Ptr<TrackerVit> tracker;
        try
        {
            TrackerVit::Params params;
            params.net = findModel(net, "");
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

        trackObject("vitTracker", tracker, inputName);
    }
    else if (modelName == "nano"){
        backbone = parser.get<String>("backbone");
        headneck = parser.get<String>("headneck");

        Ptr<TrackerNano> tracker;
        try
        {
            TrackerNano::Params params;
            params.backbone = findModel(backbone, "");
            params.neckhead = findModel(headneck, "");
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
        trackObject("nanoTracker", tracker, inputName);
        trackObject("DaSiamRPNTracker", tracker, inputName);
    }
    else if (modelName == "dasiamrpn"){
        net = parser.get<String>("dasiamrpn_net");
        kernel_cls1 = parser.get<String>("kernel_cls1");
        kernel_r1 = parser.get<String>("kernel_r1");

        Ptr<TrackerDaSiamRPN> tracker;
        try
        {
            TrackerDaSiamRPN::Params params;
            params.model = findModel(net, "");
            params.kernel_cls1 = findModel(kernel_cls1, "");
            params.kernel_r1 = findModel(kernel_r1, "");
            params.backend = backend;
            params.target = target;
            tracker = TrackerDaSiamRPN::create(params);
        }
        catch (const cv::Exception& ee)
        {
            cerr << "Exception: " << ee.what() << endl;
            cout << "Can't load the network by using the following files:" << endl;
            cout << "siamRPN : " << net << endl;
            cout << "siamKernelCL1 : " << kernel_cls1 << endl;
            cout << "siamKernelR1 : " << kernel_r1 << endl;
            return 2;
        }
        trackObject("DaSiamRPNTracker", tracker, inputName);
    }
    else{
        cout<<"Pass the valid alias. Choices are { nano, vit, dasiamrpn }"<<endl;
        return -1;
    }
}


int main(int argc, char **argv)
{
    try
    {
        return run(argc, argv);
    }
    catch (const exception& e)
    {
        cerr << "FATAL: C++ exception: " << e.what() << endl;
        return 1;
    }
}
