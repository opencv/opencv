// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// To download the onnx model, see: https://storage.googleapis.com/ailia-models/colorization/colorizer.onnx

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "common.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::dnn;


int main(int argc, char** argv) {
    const string about =
        "This sample demonstrates recoloring grayscale images with dnn.\n"
        "This program is based on:\n"
        "  http://richzhang.github.io/colorization\n"
        "  https://github.com/richzhang/colorization\n"
        "To download the onnx model:\n"
        " https://storage.googleapis.com/ailia-models/colorization/colorizer.onnx\n";

    const string param_keys =
        "{ help h          |            | Print help message. }"
        "{ input i         | baboon.jpg | Path to the input image }"
        "{ onnx_model_path |            | Path to the ONNX model. Required. }";

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

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    string inputImagePath = parser.get<string>("input");
    string onnxModelPath = parser.get<string>("onnx_model_path");
    int backendId = parser.get<int>("backend");
    int targetId = parser.get<int>("target");

    if (onnxModelPath.empty()) {
        cerr << "The path to the ONNX model is required!" << endl;
        return -1;
    }

    Mat imgGray = imread(samples::findFile(inputImagePath), IMREAD_GRAYSCALE);
    if (imgGray.empty()) {
        cerr << "Could not read the image: " << inputImagePath << endl;
        return -1;
    }

    Mat imgL = imgGray;
    imgL.convertTo(imgL, CV_32F, 100.0/255.0);
    Mat imgLResized;
    resize(imgL, imgLResized, Size(256, 256), 0, 0, INTER_CUBIC);

    // Prepare the model
    EngineType engine = ENGINE_AUTO;
    dnn::Net net = dnn::readNetFromONNX(onnxModelPath, engine);
    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);
    //! [Read and initialize network]

    // Create blob from the image
    Mat blob = dnn::blobFromImage(imgLResized, 1.0, Size(256, 256), Scalar(), false, false);

    net.setInput(blob);

    // Run inference
    Mat result = net.forward();
    Size siz(result.size[2], result.size[3]);
    Mat a(siz, CV_32F, result.ptr(0,0));
    Mat b(siz, CV_32F, result.ptr(0,1));
    resize(a, a, imgGray.size());
    resize(b, b, imgGray.size());

    // merge, and convert back to BGR
    Mat color, chn[] = {imgL, a, b};

    // Proc
    Mat lab;
    merge(chn, 3, lab);
    cvtColor(lab, color, COLOR_Lab2BGR);

    imshow("input image", imgGray);
    imshow("output image", color);
    waitKey();

    return 0;
}
