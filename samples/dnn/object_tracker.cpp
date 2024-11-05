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

const string about = "Use this script for Object Tracking using OpenCV. \n\n"
        "Firstly, download required models using the download_models.py. For vit tracker download model using `python download_models.py vit`\n"
        "To run provide alias/model_name:\n"
            "\t Nano: \n"
                "\t\t e.g: ./example_dnn_object_tracker nano\n\n"
            "\t vit: \n"
                "\t\t e.g: ./example_dnn_object_tracker vit\n\n"
            "\t dasiamrpn: \n"
                "\t\t e.g: ./example_dnn_object_tracker dasiamrpn\n\n";

const string param_keys =
        "{ help     h    |                            | Print help message }"
        "{ @alias        |                            | An alias name of model to extract preprocessing parameters from models.yml file. }"
        "{ zoo           |      ../dnn/models.yml     | An optional path to file with preprocessing parameters }"
        "{ input    i    |                            | Full path to input video folder, the specific camera index. (empty for camera 0) }"
        "{ tracking_thrs |             0.3            | Tracking score threshold. If a bbox of score >= 0.3, it is considered as found }";

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

static void loadParser(const string &modelName, const string &zooFile)
{
    // Load appropriate preprocessing arguments based on model name
    if (modelName == "vit")
    {
        keys += genPreprocArguments(modelName, zooFile, "");
    }
    else if (modelName == "nano")
    {
        keys += genPreprocArguments(modelName, zooFile, "nanotrack_head_");
        keys += genPreprocArguments(modelName, zooFile, "nanotrack_back_");
    }
    else if (modelName == "dasiamrpn")
    {
        keys += genPreprocArguments(modelName, zooFile, "dasiamrpn_");
        keys += genPreprocArguments(modelName, zooFile, "dasiamrpn_kernel_r1_");
        keys += genPreprocArguments(modelName, zooFile, "dasiamrpn_kernel_cls_");
    }
    return;
}


static void createTracker(const string &modelName, CommandLineParser &parser, Ptr<Tracker> &tracker) {
    int backend = getBackendID(parser.get<String>("backend"));
    int target = getTargetID(parser.get<String>("target"));
    if (modelName == "dasiamrpn") {
        const string net = parser.get<String>("dasiamrpn_model");
        const string sha1 = parser.get<String>("dasiamrpn_sha1");
        const string kernel_cls1 = parser.get<String>("dasiamrpn_kernel_cls_model");
        const string kernel_cls_sha1 = parser.get<String>("dasiamrpn_kernel_cls_sha1");
        const string kernel_r1 = parser.get<String>("dasiamrpn_kernel_r1_model");
        const string kernel_sha1 = parser.get<String>("dasiamrpn_kernel_r1_sha1");

        TrackerDaSiamRPN::Params params;
        params.model = findModel(net, sha1);
        params.kernel_cls1 = findModel(kernel_cls1, kernel_cls_sha1);
        params.kernel_r1 = findModel(kernel_r1, kernel_sha1);
        params.backend = backend;
        params.target = target;
        tracker = TrackerDaSiamRPN::create(params);
    } else if (modelName == "nano") {
        const string backbone = parser.get<String>("nanotrack_back_model");
        const string backSha1 = parser.get<String>("nanotrack_back_sha1");
        const string headneck = parser.get<String>("nanotrack_head_model");
        const string headSha1 = parser.get<String>("nanotrack_head_sha1");

        TrackerNano::Params params;
        params.backbone = findModel(backbone, backSha1);
        params.neckhead = findModel(headneck, headSha1);
        params.backend = backend;
        params.target = target;
        tracker = TrackerNano::create(params);
    } else if (modelName == "vit") {
        const string net = parser.get<String>("model");
        const string sha1 = parser.get<String>("sha1");
        float tracking_score_threshold = parser.get<float>("tracking_thrs");

        TrackerVit::Params params;
        params.net = findModel(net, sha1);
        params.backend = backend;
        params.target = target;
        params.tracking_score_threshold = tracking_score_threshold;
        tracker = TrackerVit::create(params);
    } else {
        cout<<"Pass the valid alias. Choices are { nano, vit, dasiamrpn }"<<endl;
    }
    return;
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);
    if (!parser.has("@alias") || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string modelName = parser.get<String>("@alias");
    const string zooFile = findFile(parser.get<String>("zoo"));
    loadParser(modelName, zooFile);
    parser = CommandLineParser(argc, argv, keys);

    Ptr<Tracker> tracker;
    createTracker(modelName, parser, tracker);

    const string windowName = "TRACKING";
    namedWindow(windowName, WINDOW_NORMAL);
    FontFace fontFace("sans");
    int stdSize = 20;
    int stdWeight = 400;
    int stdImgSize = 512;
    int imgWidth = -1;
    int fontSize = 50;
    int fontWeight = 500;
    double alpha = 0.4;
    Rect selectRect;
    string inputName = parser.get<String>("input");
    string instructionLabel = "Press space bar to pause video to draw bounding box.";
    Rect banner;
    // Open a video file or an image file or a camera stream.
    VideoCapture cap;

    if (inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1))
    {
        int c = inputName.empty() ? 0 : inputName[0] - '0';
        cout << "Trying to open camera #" << c << " ..." << endl;
        if (!cap.open(c))
        {
            cout << "Capture from camera #" << c << " didn't work. Specify -i=<video> parameter to read from video file" << endl;
            return 0;
        }
    }
    else if (inputName.size())
    {
        string filePath = findFile(inputName);
        if (!cap.open(filePath))
        {
            cout << "Could not open: " << inputName << endl;
            return 0;
        }
    }

    Mat image;

    for (;;)
    {
        cap >> image;
        if (image.empty())
        {
            cerr << "Can't capture frame. End of video stream?" << endl;
            return 0;
        }
        else if (imgWidth == -1){
            imgWidth = min(image.rows, image.cols);
            fontSize = (stdSize*imgWidth)/stdImgSize;
            fontWeight = (stdWeight*imgWidth)/stdImgSize;
            banner = getTextSize(Size(), instructionLabel, Point(), fontFace, fontSize, fontWeight);
            banner.height += 2 * fontSize; // padding
            banner.width += 10; // padding
        }
        Mat org_img = image.clone();
        rectangle(image, banner, Scalar::all(255), FILLED);
        addWeighted(image, alpha, org_img, 1 - alpha, 0, image);
        putText(image, instructionLabel, Point(10, fontSize), Scalar(0,0,0), fontFace, fontSize, fontWeight);
        putText(image, "Press space bar after selecting.", Point(10, 2*fontSize), Scalar(0,0,0), fontFace, fontSize, fontWeight);
        imshow(windowName, image);
        int key = waitKey(30); //Simulating 30 FPS, if reduced frames move really fast
        if (key == ' ')
        {
            selectRect = selectROI(windowName, image);
            break;
        }
        if (key == 27) // ESC key to exit
        {
            exit(0);
        }
    }

    cout << "ROI=" << selectRect << endl;
    tracker->init(image, selectRect);
    instructionLabel = "Press space bar to select new target";
    banner = getTextSize(Size(), instructionLabel, Point(), fontFace, fontSize, fontWeight);
    banner.height += 4 * fontSize; // padding
    banner.width += 10; // padding

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

        float score = tracker->getTrackingScore();

        Mat render_image = image.clone();

        int key = waitKey(30); //Simulating 30 FPS, if reduced frames move really fast
        int h = image.rows;
        int w = image.cols;
        rectangle(render_image, banner, Scalar::all(255), FILLED);
        rectangle(render_image, cv::Point(0, int(h - int(1.5*fontSize))), cv::Point(w, h), Scalar::all(255), FILLED);
        addWeighted(render_image, alpha, image, 1 - alpha, 0, render_image);
        putText(render_image, instructionLabel, Point(10, fontSize), Scalar(0,0,0), fontFace, fontSize, fontWeight);
        putText(render_image, "For switching between trackers: press 'v' for ViT, 'n' for Nano, and 'd' for DaSiamRPN.", Point(10, h-10), Scalar(0,0,0), fontFace, int(0.8*fontSize), fontWeight);

        if (ok){
            if (key == ' '){
                putText(render_image, "Select the new target", Point(10, 2*fontSize), Scalar(0,0,0), fontFace, fontSize, fontWeight);
                selectRect = selectROI(windowName, render_image);
                tracker->init(image, selectRect);
            }
            else if (key == 'v'){
                modelName = "vit";
                loadParser(modelName, zooFile);
                parser = CommandLineParser(argc, argv, keys);
                createTracker(modelName, parser, tracker);
                tracker->init(image, rect);
            }
            else if (key == 'n'){
                modelName = "nano";
                loadParser(modelName, zooFile);
                parser = CommandLineParser(argc, argv, keys);
                createTracker(modelName, parser, tracker);
                tracker->init(image, rect);
            }
            else if (key == 'd'){
                modelName = "dasiamrpn";
                loadParser(modelName, zooFile);
                parser = CommandLineParser(argc, argv, keys);
                createTracker(modelName, parser, tracker);
                tracker->init(image, rect);
            }
            rectangle(render_image, rect, Scalar(0, 255, 0), 2);
        }

        string timeLabel = format("Inference time: %.2f ms", tickMeter.getTimeMilli());
        string scoreLabel = format("Score: %f", score);
        string algoLabel = "Algorithm: " + modelName;
        putText(render_image, timeLabel, Point(10, 2*fontSize), Scalar(0,0,0), fontFace, fontSize, fontWeight);
        putText(render_image, scoreLabel, Point(10, 3*fontSize), Scalar(0,0,0), fontFace, fontSize, fontWeight);
        putText(render_image, algoLabel, Point(10, 4*fontSize), Scalar(0,0,0), fontFace, fontSize, fontWeight);

        imshow(windowName, render_image);

        tickMeter.reset();

        if (key == 27 /*ESC*/)
            exit(0);
    }
    return 0;
}
