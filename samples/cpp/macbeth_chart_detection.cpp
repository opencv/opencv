#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#ifdef HAVE_OPENCV_DNN
#include <opencv2/dnn.hpp>
#include "../dnn/common.hpp"
#endif
#include <iostream>

using namespace std;
using namespace cv;
#ifdef HAVE_OPENCV_DNN
using namespace cv::dnn;
#endif
using namespace mcc;

const string about =
    "This sample demonstrates mcc checker detection with DNN based model and thresholding (default) techniques.\n\n"
    "To run default:\n"
    "\t ./example_cpp_macbeth_chart_detection --input=path/to/your/input/image/or/video (don't give --input flag if want to use device camera)\n"
#ifdef HAVE_OPENCV_DNN
    "With DNN model:\n"
    "\t ./example_cpp_macbeth_chart_detection mcc --input=path/to/your/input/image/or/video\n\n"
    "Model path can also be specified using --model argument. And config path can be specified using --config. Download it using python download_models.py mcc from dnn samples directory\n\n"
#else
    "Note: DNN-based detection is not available in this build.\n\n"
#endif
    ;

const string param_keys =
    "{ help h          |                   | Print help message. }"
#ifdef HAVE_OPENCV_DNN
    "{ @alias          |                   | An alias name of model to extract preprocessing parameters from models.yml file. }"
    "{ zoo             | ../dnn/models.yml | An optional path to file with preprocessing parameters }"
#endif
    "{ input i         |                   | Path to input image or video file. Skip this argument to capture frames from a camera.}"
    "{ type            |         0         | chartType: 0-Standard, 1-DigitalSG, 2-Vinyl, default:0 }"
    "{ num_charts      |         1         | Maximum number of charts in the image }"
#ifdef HAVE_OPENCV_DNN
    "{ model           |                   | Path to the model file for using dnn model. }";
#else
    ;
#endif

#ifdef HAVE_OPENCV_DNN
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
#endif

// Initialize keys before use
string keys = param_keys;
static void initKeys() {
#ifdef HAVE_OPENCV_DNN
    keys += backend_keys + target_keys;
#endif
}

static bool processFrame(const Mat& frame, Ptr<CCheckerDetector> detector, Mat& src, Mat& tgt, int nc){
    Mat imageCopy = frame.clone();
    if (!detector->process(frame, nc))
    {
        return false;
    }
    vector<Ptr<CChecker>> checkers = detector->getListColorChecker();
    detector->draw(checkers, frame);
    src = checkers[0]->getChartsRGB(false);
    tgt = detector->getRefColors();

    imshow("Image result", frame);
    imshow("Original", imageCopy);

    return true;
}

int main(int argc, char *argv[])
{
    initKeys();
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if (parser.has("help"))
    {
        cout << about << endl;
        parser.printMessage();
        return -1;
    }

#ifdef HAVE_OPENCV_DNN
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
#endif

    int t = parser.get<int>("type");

    CV_Assert(0 <= t && t <= 2);
    ColorChart chartType = ColorChart(t);

#ifdef HAVE_OPENCV_DNN
    const string sha1 = parser.get<String>("sha1");
    const string model_path = findModel(parser.get<string>("model"), sha1);
    const string config_sha1 = parser.get<String>("config_sha1");
    const string pbtxt_path = findModel(parser.get<string>("config"), config_sha1);
    const string backend = parser.get<String>("backend");
    const string target = parser.get<String>("target");
#endif

    int nc = parser.get<int>("num_charts");

    Ptr<CCheckerDetector> detector;
#ifdef HAVE_OPENCV_DNN
    if (model_path != "" && pbtxt_path != ""){
        EngineType engine = ENGINE_AUTO;
        if (backend != "default" || target != "cpu"){
            engine = ENGINE_CLASSIC;
        }
        Net net = readNetFromTensorflow(model_path, pbtxt_path, engine);
        net.setPreferableBackend(getBackendID(backend));
        net.setPreferableTarget(getTargetID(target));

        detector = CCheckerDetector::create(net);
        cout<<"Detecting checkers using neural network."<<endl;
    }
    else{
        detector = CCheckerDetector::create();
    }
#else
    detector = CCheckerDetector::create();
#endif
    detector->setColorChartType(chartType);

    bool isVideo = true;
    Mat image;
    VideoCapture cap;

    if (parser.has("input")){
        const string inputFile = parser.get<String>("input");
        image = imread(findFile(inputFile));
        if (!image.empty())
        {
            isVideo = false;
        }
        else
        {
            // Not an image, so try opening it as a video.
            cap.open(findFile(inputFile));
            if (!cap.isOpened())
            {
                cout << "[ERROR] Could not open file as an image or video: " << inputFile << endl;
                return -1;
            }
        }
    }
    else
        cap.open(0);

    Mat src, tgt;
    bool found = false;
    if (isVideo){
        cout<<"To print the actual colors and reference colors for current frame press SPACEBAR. To resume press SPACEBAR again"<<endl;

        while (cap.grab())
        {
            Mat frame;
            cap.retrieve(frame);

            found = processFrame(frame, detector, src, tgt, nc);

            int key = waitKey(10);
            if (key == ' '){
                if(found){
                    cout<<"Reference colors: "<<tgt<<endl<<"--------------------"<<endl;
                    cout<<"Actual colors: "<<src<<endl<<endl;
                    cout<<"Press spacebar to resume."<<endl;

                    waitKey(0);
                    cout << "Resumed! Processing continues..." << endl;
                }
                else{
                    cout<<"No color chart detected!!"<<endl;
                }
            }
            else if (key == 27) exit(0);
        }
        if(found){
            cout<<"Reference colors: "<<tgt<<endl<<"--------------------"<<endl;
            cout<<"Actual colors: "<<src<<endl<<endl;
        }
    }
    else{
        found = processFrame(image, detector, src, tgt, nc);
        if(found){
            cout<<"Reference colors: "<<tgt<<endl<<"--------------------"<<endl;
            cout<<"Actual colors: "<<src<<endl<<endl;
            waitKey(0);
        }
        else{
            cout<<"No chart detected!!"<<endl;
        }
    }
    return 0;
}
