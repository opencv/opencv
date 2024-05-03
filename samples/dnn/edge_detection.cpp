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
pair<Mat, Mat> postProcess(const vector<Mat>& output, int height, int width);
Mat preprocess(const Mat& img, int imageSize);

int main(int argc, char** argv) {

    const string about =
        "This sample demonstrates edge detection with dexined and canny edge detection techniques.\n\n"
        "For switching between deep learning based model(dexined) and canny edge detector, press 'd' (for dexined) or 'c' (for canny) respectively.\n\n"
        "Script is based on https://github.com/axinc-ai/ailia-models/blob/master/line_segment_detection/dexined/dexined.py\n"
        "To download the onnx model, see: https://storage.googleapis.com/ailia-models/dexined/model.onnx"
        "\n\nOpenCV onnx importer does not process dynamic shape. These need to be substituted with values using:\n\n"
        "python3 -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param w --dim_value 640 model.onnx model.sim1.onnx\n"
        "python3 -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param h --dim_value 480 model.sim1.onnx model.sim.onnx\n";

    const string param_keys =
        "{ help h          |            | Print help message. }"
        "{ input i         | baboon.jpg | Path to the input image }"
        "{ model           |            | Path to the ONNX model. Required. }"
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
    int backend = parser.get<int>("backend");
    int target = parser.get<int>("target");
    int imageSize = parser.get<int>("imageSize");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    Mat image = imread(samples::findFile(inputImagePath));
    if (image.empty()) {
        cout << "Could not read the image: " << inputImagePath << endl;
        return 1;
    }

    string method = "dexined";
    namedWindow("Input", WINDOW_NORMAL);
    imshow("Input", image);
    namedWindow("Output", WINDOW_NORMAL);

    for (;;){
        if (method == "dexined")
        {
            destroyWindow("Output");
            namedWindow("Output", WINDOW_NORMAL);

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

            imshow("Output", fusedOutput);
        }
        else if (method == "canny")
        {
            Mat gray;
            cvtColor(image, gray, COLOR_BGR2GRAY);

            UserData user_data;
            user_data.gray = gray;

            destroyWindow("Output");
            namedWindow("Output", WINDOW_NORMAL);

            // Create trackbars
            createTrackbar("thrs1", "Output", 0, 255, cannyDetectionThresh1, &user_data);
            createTrackbar("thrs2", "Output", 0, 255, cannyDetectionThresh2, &user_data);

            // Set initial positions of trackbars
            setTrackbarPos("thrs1", "Output", 100);
            setTrackbarPos("thrs2", "Output", 200);

        }

        int key = waitKey(0);

        if (key == 'd' || key == 'D')
        {
            method = "dexined";
        }
        else if (key == 'c' || key == 'C')
        {
            method = "canny";
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
pair<Mat, Mat> postProcess(const vector<Mat>& output, int height, int width) {
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
        normalize(processed, img, 0, 255, NORM_MINMAX, CV_8U);
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
    ave /= static_cast<float>(preds.size());
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