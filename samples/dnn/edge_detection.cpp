#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>

#include "common.hpp"
// Define namespace to simplify code
using namespace cv;
using namespace cv::dnn;
using namespace std;

Mat gray;
int threshold1 = 20;
int threshold2 = 50;
int blurAmount = 5;

// Function to apply sigmoid activation
static void sigmoid(Mat& input) {
    exp(-input, input);          // e^-input
    input = 1.0 / (1.0 + input); // 1 / (1 + e^-input)
}

// Function to apply Canny edge detection with Gaussian blur
static void applyCanny(int, void*){
    int kernelSize = 2 * blurAmount + 1;
    Mat blurred;
    GaussianBlur(gray, blurred, Size(kernelSize, kernelSize), 0);
    Mat output;
    Canny(blurred, output, threshold1, threshold2);
    imshow("Output", output);
}

// Load Model
static void loadModel(const string modelPath, int backend, int target, Net &net){
    net = readNetFromONNX(modelPath);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
}

static void setupCannyWindow(const Mat &image){
    destroyWindow("Output");
    namedWindow("Output", WINDOW_AUTOSIZE);
    moveWindow("Output", 200, 50);
    cvtColor(image, gray, COLOR_BGR2GRAY);

    createTrackbar("thrs1", "Output", &threshold1, 255, applyCanny);
    createTrackbar("thrs2", "Output", &threshold2, 255, applyCanny);
    createTrackbar("blur", "Output", &blurAmount, 20, applyCanny);
}

// Function to process the neural network output to generate edge maps
static pair<Mat, Mat> postProcess(const vector<Mat>& output, int height, int width) {
    vector<Mat> preds;
    preds.reserve(output.size());
    for (const Mat &p : output) {
        Mat img;
        // Correctly handle 4D tensor assuming it's always in the format [1, 1, height, width]
        Mat processed;
        if (p.dims == 4 && p.size[0] == 1 && p.size[1] == 1) {
            // Use only the spatial dimensions
            processed = p.reshape(0, {p.size[2], p.size[3]});
        } else {
            processed = p.clone();
        }
        sigmoid(processed);
        normalize(processed, img, 0, 255, NORM_MINMAX, CV_8U);
        resize(img, img, Size(width, height)); // Resize to the original size
        preds.push_back(img);
    }
    Mat fuse = preds.back(); // Last element as the fused result
    // Calculate the average of the predictions
    Mat ave = Mat::zeros(height, width, CV_32F);
    for (Mat &pred : preds) {
        Mat temp;
        pred.convertTo(temp, CV_32F);
        ave += temp;
    }
    ave /= static_cast<float>(preds.size());
    ave.convertTo(ave, CV_8U);
    return {fuse, ave}; // Return both fused and average edge maps
}

int main(int argc, char** argv) {
    const string about =
        "This sample demonstrates edge detection with dexined and canny edge detection techniques.\n\n"
        "For switching between deep learning based model(dexined) and canny edge detector, press 'd' (for dexined) or 'c' (for canny) respectively in case of video. For image pass the argument --method for switching between dexined and canny.\n\n";

    const string param_keys =
        "{ help h          |            | Print help message. }"
        "{ @alias          |            | An alias name of model to extract preprocessing parameters from models.yml file. }"
        "{ zoo             | models.yml | An optional path to file with preprocessing parameters }"
        "{ input i         |            | Path to input image or video file. Skip this argument to capture frames from a camera.}"
        "{ method          |   canny    | Choose method: dexined or canny. }";

    const string backend_keys = format(
        "{ backend         | 0 | Choose one of computation backends: "
        "%d: automatically (by default), "
        "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
        "%d: OpenCV implementation, "
        "%d: VKCOM, "
        "%d: CUDA, "
        "%d: WebNN }",
        DNN_BACKEND_DEFAULT, DNN_BACKEND_INFERENCE_ENGINE, DNN_BACKEND_OPENCV,
        DNN_BACKEND_VKCOM, DNN_BACKEND_CUDA, DNN_BACKEND_WEBNN);

    const string target_keys = format(
        "{ target          | 0 | Choose one of target computation devices: "
        "%d: CPU target (by default), "
        "%d: OpenCL, "
        "%d: OpenCL fp16 (half-float precision), "
        "%d: VPU, "
        "%d: Vulkan, "
        "%d: CUDA, "
        "%d: CUDA fp16 (half-float preprocess) }",
        DNN_TARGET_CPU, DNN_TARGET_OPENCL, DNN_TARGET_OPENCL_FP16,
        DNN_TARGET_MYRIAD, DNN_TARGET_VULKAN, DNN_TARGET_CUDA,
        DNN_TARGET_CUDA_FP16);

    string keys = param_keys + backend_keys + target_keys;

    CommandLineParser parser(argc, argv, keys);

    const string modelName = parser.get<String>("@alias");
    const string zooFile = parser.get<String>("zoo");

    keys += genPreprocArguments(modelName, zooFile);

    parser = CommandLineParser(argc, argv, keys);
    int width = parser.get<int>("width");
    int height = parser.get<int>("height");
    float scale = parser.get<float>("scale");
    Scalar mean = parser.get<Scalar>("mean");
    bool swapRB = parser.get<bool>("rgb");
    int backend = parser.get<int>("backend");
    int target = parser.get<int>("target");
    string method = parser.get<String>("method");
    string model = findFile(parser.get<String>("model"));
    parser.about(about);

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    VideoCapture cap;
    if (parser.has("input"))
        cap.open(samples::findFile(parser.get<String>("input")));
    else
        cap.open(0);

    namedWindow("Input", WINDOW_AUTOSIZE);
    namedWindow("Output", WINDOW_AUTOSIZE);
    moveWindow("Output", 200, 0);
    Net net;
    Mat image;

    if (model.empty()) {
        cout << "[WARN] Model file not provided, cannot use dexined." << endl;
        method = "canny";
    }

    if (method == "dexined") {
        loadModel(model, backend, target, net);
    }
    else{
        Mat dummy = Mat::zeros(512, 512, CV_8UC3);
        setupCannyWindow(dummy);
    }

    for (;;){
        cap >> image;
        if (image.empty())
        {
            cout << "Press any key to exit" << endl;
            waitKey();
            break;
        }
        if (method == "dexined")
        {
            Mat blob = blobFromImage(image, scale, Size(width, height), mean, swapRB, false, CV_32F);
            net.setInput(blob);
            vector<Mat> outputs;
            net.forward(outputs);
            int originalWidth = image.cols;
            int originalHeight = image.rows;
            pair<Mat, Mat> res = postProcess(outputs, originalHeight, originalWidth);
            Mat fusedOutput = res.first;
            Mat averageOutput = res.second;
            imshow("Output", fusedOutput);
        }
        else if (method == "canny")
        {
            cvtColor(image, gray, COLOR_BGR2GRAY);
            applyCanny(0, 0);
        }
        imshow("Input", image);
        int key = waitKey(30);

        if (key == 'd' || key == 'D')
        {
            if (!model.empty()){
                method = "dexined";
                if (net.empty())
                    loadModel(model, backend, target, net);
                namedWindow("Input", WINDOW_AUTOSIZE);
                namedWindow("Output", WINDOW_AUTOSIZE);
                moveWindow("Output", 200, 0);
            } else {
                cout << "[ERROR] Model file not provided, cannot use dexined." << endl;
            }
        }
        else if (key == 'c' || key == 'C')
        {
            method = "canny";
            setupCannyWindow(image);
        }
        else if (key == 27 || key == 'q')
        { // Escape key to exit
            break;
        }
    }
    destroyAllWindows();
    return 0;
}