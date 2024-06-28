#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>

// Define namespace to simplify code
using namespace cv;
using namespace cv::dnn;
using namespace std;

struct UserData
{
    Mat gray;
    int thrs1 = 100;
    int thrs2 = 200;
};
// Function to apply sigmoid activation
static void sigmoid(Mat& input) {
    exp(-input, input);          // e^-input
    input = 1.0 / (1.0 + input); // 1 / (1 + e^-input)
}
// Callback for the first threshold adjustment
static void cannyDetectionThresh1(int position, void* userdata) {
    UserData* data = reinterpret_cast<UserData*>(userdata);
    Mat output;
    Canny(data->gray, output, position, data->thrs2);
    data->thrs1 = position;
    imshow("Output", output);
}
// Callback for the second threshold adjustment
static void cannyDetectionThresh2(int position, void* userdata) {
    UserData* data = reinterpret_cast<UserData*>(userdata);
    Mat output;
    Canny(data->gray, output, data->thrs1, position);
    data->thrs2 = position;
    imshow("Output", output);
}

// Load Model
static void loadModel(const string modelPath, int backend, int target, Net &net){
    net = readNetFromONNX(modelPath);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
}

static void setupCannyWindow(const Mat &image, UserData &user_data){
    destroyWindow("Output");
    namedWindow("Output", WINDOW_NORMAL);
    moveWindow("Output", 200, 50);
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    user_data.gray = gray;
    // Create trackbars
    createTrackbar("thrs1", "Output", 0, 255, cannyDetectionThresh1, &user_data);
    createTrackbar("thrs2", "Output", 0, 255, cannyDetectionThresh2, &user_data);

    // Set initial positions of trackbars
    setTrackbarPos("thrs1", "Output", 100);
    setTrackbarPos("thrs2", "Output", 200);

}
static pair<Mat, Mat> postProcess(const vector<Mat>& output, int height, int width);

int main(int argc, char** argv) {

    const string about =
        "This sample demonstrates edge detection with dexined and canny edge detection techniques.\n\n"
        "For switching between deep learning based model(dexined) and canny edge detector, press 'd' (for dexined) or 'c' (for canny) respectively.\n\n"
        "Script is based on https://github.com/axinc-ai/ailia-models/blob/master/line_segment_detection/dexined/dexined.py\n"
        "To download the onnx model, see: https://storage.googleapis.com/ailia-models/dexined/model.onnx"
        "\n\nOpenCV onnx importer does not process dynamic shape. These need to be substituted with values using:\n\n"
        "python3 -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param w --dim_value 512 model.onnx model.sim1.onnx\n"
        "python3 -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param h --dim_value 512 model.sim1.onnx model.sim.onnx\n";

    const string param_keys =
        "{ help h          |        | Print help message. }"
        "{ input i         |        | Path to input image or video file. Skip this argument to capture frames from a camera.}"
        "{ model           |        | Path to the ONNX model. Required. }"
        "{ imageSize       |   512  | Image Size}";

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

    const string keys = param_keys + backend_keys + target_keys;

    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    const string modelPath = parser.get<string>("model");
    int backend = parser.get<int>("backend");
    int target = parser.get<int>("target");
    int imageSize = parser.get<int>("imageSize");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    VideoCapture cap;
    if (parser.has("input"))
        cap.open(samples::findFile(parser.get<String>("input")));
    else
        cap.open(0);

    namedWindow("Input", WINDOW_NORMAL);
    namedWindow("Output", WINDOW_NORMAL);
    moveWindow("Output", 200, 0);
    // Check if the 'modelPath' string is empty and set the 'method' accordingly
    string method;
    Net net;
    Mat image, gray;
    UserData user_data;

    if (modelPath.empty()) {
        cout << "[WARN] Model file not provided, cannot use dexined." << endl;
        method = "canny";
        Mat dummy = Mat::zeros(imageSize, imageSize, CV_8UC3);
        setupCannyWindow(dummy, user_data);
    } else {
        method = "dexined";
        loadModel(modelPath, backend, target, net);
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
            Mat blob = blobFromImage(image, 1.0, Size(imageSize, imageSize), Scalar(103.939, 116.779, 123.68), false, false, CV_32F);
            net.setInput(blob);
            vector<Mat> outputs;
            net.forward(outputs); // Get all output layers
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
            user_data.gray = gray;
            cannyDetectionThresh1(user_data.thrs1, &user_data);
        }
        imshow("Input", image);
        int key = waitKey(30);

        if (key == 'd' || key == 'D')
        {
            if (!modelPath.empty())
                method = "dexined";
            if (net.empty())
                loadModel(modelPath, backend, target, net);
            namedWindow("Input", WINDOW_NORMAL);
            namedWindow("Output", WINDOW_NORMAL);
            moveWindow("Output", 200, 0);
        }
        else if (key == 'c' || key == 'C')
        {
            method = "canny";
            setupCannyWindow(image, user_data);
        }
        else if (key == 27 || key == 'q')
        { // Escape key to exit
            break;
        }
    }
    destroyAllWindows();
    return 0;
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