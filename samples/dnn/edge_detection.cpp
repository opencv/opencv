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

struct UserData {
            Mat gray;
            int thrs1 = 100;
            int thrs2 = 200;
        };

// Function to apply sigmoid activation
static void sigmoid(Mat& input) {
    exp(-input, input); // e^-input
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

pair<Mat, Mat> postProcess(const vector<Mat>& output, int height, int width);
Mat preprocess(const Mat& img, int imageSize);

int main(int argc, char** argv) {

    const string about =
        "This sample demonstrates edge detection with dexined and canny edge detection techniques.\n"
        "Script is based on https://github.com/axinc-ai/ailia-models/blob/master/line_segment_detection/dexined/dexined.py"
        "To download the onnx model, see: https://storage.googleapis.com/ailia-models/dexined/model.onnx"
        "\n\nOpenCV onnx importer does not process dynamic shape. These need to be substituted with values using:\n\n"
        "python3 -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param w --dim_value 640 model.onnx model.sim1.onnx"
        "python3 -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param h --dim_value 480 model.sim1.onnx model.sim.onnx";

    const string param_keys =
        "{ help h          |            | Print help message. }"
        "{ input i         | baboon.jpg | Path to the input image }"
        "{ model           |            | Path to the ONNX model. Required. }"
        "{ method          |   dexined  | Choose methd: dexined or canny}"
        "{ imageSize       |   512      | Image Size}";

    const string backend_keys = format(
        "{ backend         | 0 | Choose one of computation backends: "
                                    "%d: automatically (by default), "
                                    "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                                    "%d: OpenCV implementation, "
                                    "%d: VKCOM, "
                                    "%d: CUDA, "
                                    "%d: WebNN }",
        cv::dnn::DNN_BACKEND_DEFAULT, cv::dnn::DNN_BACKEND_INFERENCE_ENGINE, cv::dnn::DNN_BACKEND_OPENCV,
        cv::dnn::DNN_BACKEND_VKCOM, cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_BACKEND_WEBNN);

    const string target_keys = format(
        "{ target          | 0 | Choose one of target computation devices: "
                              "%d: CPU target (by default), "
                              "%d: OpenCL, "
                              "%d: OpenCL fp16 (half-float precision), "
                              "%d: VPU, "
                              "%d: Vulkan, "
                              "%d: CUDA, "
                              "%d: CUDA fp16 (half-float preprocess) }",
        cv::dnn::DNN_TARGET_CPU, cv::dnn::DNN_TARGET_OPENCL, cv::dnn::DNN_TARGET_OPENCL_FP16,
        cv::dnn::DNN_TARGET_MYRIAD, cv::dnn::DNN_TARGET_VULKAN, cv::dnn::DNN_TARGET_CUDA,
        cv::dnn::DNN_TARGET_CUDA_FP16);

    const string keys = param_keys + backend_keys + target_keys;

    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    string inputImagePath = parser.get<string>("input");
    string onnxModel = parser.get<string>("model");
    string method =  parser.get<string>("method");
    int backend = parser.get<int>("backend");
    int target = parser.get<int>("target");
    int imageSize =  parser.get<int>("imageSize");

    Mat image = imread(samples::findFile(inputImagePath));
    if (image.empty()) {
        cout << "Could not read the image: " << inputImagePath << endl;
        return 1;
    }

    if (method == "dexined") {
        Net net = readNetFromONNX(onnxModel);
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);
        Mat preprocessed = preprocess(image, imageSize);

        Mat blob = blobFromImage(preprocessed);
        net.setInput(blob);

        Mat result = net.forward();

        vector<Mat> outputs;
        net.forward(outputs); // Get all output layers
        int originalWidth = image.cols;
        int originalHeight = image.rows;
        pair<Mat, Mat> res = postProcess(outputs, originalHeight, originalWidth);
        Mat fusedOutput = res.first;
        Mat averageOutput = res.second;


        imshow("Input", image);
        imshow("Output", fusedOutput);
        waitKey(0);
    } else if (method == "canny") {
        Mat gray;
        cvtColor(image, gray, COLOR_BGR2GRAY);

        UserData user_data;
        user_data.gray = gray;

        namedWindow("Output", WINDOW_NORMAL);
        namedWindow("Input", WINDOW_NORMAL);

         // Create trackbars
        createTrackbar("thrs1", "Output",0, 255, cannyDetectionThresh1, &user_data);
        createTrackbar("thrs2", "Output",0, 255, cannyDetectionThresh2, &user_data);

        // Set initial positions of trackbars
        setTrackbarPos("thrs1", "Output", 100);
        setTrackbarPos("thrs2", "Output", 200);

        imshow("Input", image);
        waitKey(0);
    }

    return 0;
}

// Function to process the neural network output to generate edge maps
pair<Mat, Mat> postProcess(const vector<Mat>& output, int height, int width) {
    const float epsilon = 1e-12;
    vector<Mat> preds;
    preds.reserve(output.size());

    for (const auto& p : output) {
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

        double minVal, maxVal;
        minMaxLoc(processed, &minVal, &maxVal); // Find min and max values

        // Normalize the image to [0, 255]
        img = (processed - minVal) * 255.0 / (maxVal - minVal + epsilon);
        img.convertTo(img, CV_8U); // Convert to 8-bit image

        resize(img, img, Size(width, height)); // Resize to the original size
        preds.push_back(img);
    }

    Mat fuse = preds.back(); // Last element as the fused result

    // Calculate the average of the predictions
    Mat ave = Mat::zeros(height, width, CV_32F);
    for (auto& pred : preds) {
        Mat temp;
        pred.convertTo(temp, CV_32F);
        ave += temp;
    }
    ave /= preds.size();
    ave.convertTo(ave, CV_8U);

    return {fuse, ave}; // Return both fused and average edge maps
}

// Preprocess the image
Mat preprocess(const Mat& img, int imageSize) {
    Mat resizedImg;
    resize(img, resizedImg, Size(imageSize, imageSize));
    resizedImg.convertTo(resizedImg, CV_32F);
    subtract(resizedImg, Scalar(103.939, 116.779, 123.68), resizedImg);
    return resizedImg;
}