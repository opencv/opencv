#include <opencv2/core.hpp>

#include <opencv2/highgui.hpp>
#include "opencv2/objdetect/mcc_checker_detector.hpp"
#include <opencv2/dnn.hpp>
#include <iostream>
#include "../dnn/common.hpp"

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace mcc;

const string about =
    "This sample demonstrates mcc checker detection with DNN based model and thresholding (default) techniques.\n\n"
    "To run default:\n"
    "\t ./example_cpp_macbeth_chart_detection --input=path/to/your/input/image/or/video (don't give --input flag if want to use device camera)\n"
    "With DNN model:\n"
    "\t ./example_cpp_macbeth_chart_detection mcc --input=path/to/your/input/image/or/video\n\n"
    "Model path can also be specified using --model argument. And config path can be specified using --config. Download it using python download_models.py mcc from dnn samples directory\n\n";

const string param_keys =
    "{ help h          |                   | Print help message. }"
    "{ @alias          |                   | An alias name of model to extract preprocessing parameters from models.yml file. }"
    "{ zoo             | ../dnn/models.yml | An optional path to file with preprocessing parameters }"
    "{ input i         |                   | Path to input image or video file. Skip this argument to capture frames from a camera.}"
    "{ type            |         0         | chartType: 0-Standard, 1-DigitalSG, 2-Vinyl, default:0 }"
    "{ num_charts      |         1         | Maximum number of charts in the image }"
    "{ use_gpu         |                   | Add this flag if you want to use gpu}"
    "{ model           |                   | Path to the model file for using dnn model. }";

const string backend_keys = format(
    "{ backend          | default | Choose one of computation backends: "
                            "default: automatically (by default), "
                            "openvino: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                            "opencv: OpenCV implementation, "
                            "vkcom: VKCOM, "
                            "cuda: CUDA, "
                            "webnn: WebNN }");

const string target_keys = format(
    "{ target           | cpu | Choose one of target computation devices: "
                            "cpu: CPU target (by default), "
                            "opencl: OpenCL, "
                            "opencl_fp16: OpenCL fp16 (half-float precision), "
                            "vpu: VPU, "
                            "vulkan: Vulkan, "
                            "cuda: CUDA, "
                            "cuda_fp16: CUDA fp16 (half-float preprocess) }");

string keys = param_keys + backend_keys + target_keys;

int main(int argc, char *argv[])
{
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if (parser.has("help"))
    {
        cout << about << endl;
        parser.printMessage();
        return -1;
    }
    string modelName = parser.get<String>("@alias");
    string zooFile = parser.get<String>("zoo");
    const char* path = getenv("OPENCV_SAMPLES_DATA_PATH");
    if ((path != NULL) || parser.has("@alias") || (parser.get<String>("model") != "")) {
        zooFile = findFile(zooFile);
    }
    else{
        cout<<"[WARN] set the environment variables or pass the arguments --model, --config and models.yml file using --zoo for using dnn based detector. Continuing with default detector.\n\n";
    }

    keys += genPreprocArguments(modelName, zooFile);
    parser = CommandLineParser(argc, argv, keys);

    int t = parser.get<int>("type");

    CV_Assert(0 <= t && t <= 2);
    COLORCHART chartType = COLORCHART(t);

    const string sha1 = parser.get<String>("sha1");
    const string model_path = findModel(parser.get<string>("model"), sha1);
    const string config_sha1 = parser.get<String>("config_sha1");
    const string pbtxt_path = findModel(parser.get<string>("config"), config_sha1);

    int nc = parser.get<int>("num_charts");
    bool use_gpu = parser.has("use_gpu");

    int waitTime=10;
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(findFile(parser.get<String>("input")));
    else
        cap.open(0);

    Ptr<CCheckerDetector> detector;
    if (model_path != "" && pbtxt_path != ""){
        Net net = readNetFromTensorflow(model_path, pbtxt_path);

        if (use_gpu)
        {
            net.setPreferableBackend(getBackendID("cuda"));
            net.setPreferableTarget(getTargetID("cuda"));
        }else{
            net.setPreferableBackend(getBackendID(parser.get<String>("backend")));
            net.setPreferableTarget(getTargetID(parser.get<String>("target")));
        }

        detector = CCheckerDetector::create(net);
        cout<<"Detecting checkers using neural network."<<endl;
    }
    else{
        detector = CCheckerDetector::create();
    }

    while (cap.grab())
    {
        Mat image, imageCopy;
        cap.retrieve(image);

        imageCopy = image.clone();

        // Marker type to detect
        if (!detector->process(image, chartType, nc, true))
        {
            printf("ChartColor not detected \n");
        }
        else
        {
            vector<Ptr<CChecker>> checkers = detector->getListColorChecker();
            detector->draw(checkers, image);

            Mat chartRgb = checkers[0]->getChartsRGB();
            Mat src = chartRgb.col(1).clone().reshape(3, chartRgb.rows/3);
            Mat tgt;
            detector->getRefColor(MCC24, tgt);
            cout<<"Reference colors: "<<tgt<<endl<<"--------------------"<<endl;
            cout<<"Actual colors: "<<src<<endl;
        }

        imshow("image result | q or esc to quit", image);
        imshow("original", imageCopy);
        char key = (char)waitKey(waitTime);
        if (key == 27)
            break;
    }
    waitKey(0);
    return 0;
}
