/*
This file is part of OpenCV project.
It is subject to the license terms in the LICENSE file found in the top-level directory
of this distribution and at http://opencv.org/license.html.

This sample deblurs the given blurry image.

Copyright (C) 2025, Bigvision LLC.

How to use:
    Sample command to run:
        `./example_dnn_deblurring`

    You can download NAFNet deblurring model using
        `python download_models.py NAFNet`

    References:
      Github: https://github.com/megvii-research/NAFNet
      PyTorch model: https://drive.google.com/file/d/14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X/view

      PyTorch model was converted to ONNX and then ONNX model was further quantized using block quantization from [opencv_zoo](https://github.com/opencv/opencv_zoo)

    Set environment variable OPENCV_DOWNLOAD_CACHE_DIR to point to the directory where models are downloaded. Also, point OPENCV_SAMPLES_DATA_PATH to opencv/samples/data.
*/

#include <iostream>
#include <fstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "common.hpp"

using namespace cv;
using namespace dnn;
using namespace std;

const string about = "Use this script for image deblurring using OpenCV. \n\n"
        "Firstly, download required models i.e. NAFNet using `download_models.py` (if not already done). Set environment variable OPENCV_DOWNLOAD_CACHE_DIR to point to the directory where models are downloaded. Also, point OPENCV_SAMPLES_DATA_PATH to opencv/samples/data.\n"
        "To run:\n"
        "\t Example: ./example_dnn_deblurring [--input=<image_name>] \n\n"
        "Deblurring model path can also be specified using --model argument.\n\n";

const string param_keys =
    "{ help    h  |                           | show help message}"
    "{ @alias     |           NAFNet          | An alias name of model to extract preprocessing parameters from models.yml file. }"
    "{ zoo        |     ../dnn/models.yml     | An optional path to file with preprocessing parameters }"
    "{ input   i  |  licenseplate_motion.jpg  | image file path}";

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


int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv, keys);

    if (!parser.has("@alias") || parser.has("help"))
    {
        cout<<about<<endl;
        parser.printMessage();
        return 0;
    }
    string modelName = parser.get<String>("@alias");
    string zooFile = findFile(parser.get<String>("zoo"));
    keys += genPreprocArguments(modelName, zooFile);
    parser = CommandLineParser(argc, argv, keys);
    parser.about("Use this script to run image deblurring using OpenCV.");

    const string sha1 = parser.get<String>("sha1");
    const string modelPath = findModel(parser.get<String>("model"), sha1);
    string imgPath = parser.get<String>("input");
    const string backend = parser.get<String>("backend");
    const string target = parser.get<String>("target");
    float scale = parser.get<float>("scale");
    bool swapRB = parser.get<bool>("rgb");
    Scalar mean_v = parser.get<Scalar>("mean");

    EngineType engine = ENGINE_AUTO;
    if (backend != "default" || target != "cpu"){
        engine = ENGINE_CLASSIC;
    }

    Net net = readNetFromONNX(modelPath, engine);
    net.setPreferableBackend(getBackendID(backend));
    net.setPreferableTarget(getTargetID(target));

    Mat inputImage = imread(findFile(imgPath));
    if (inputImage.empty()) {
        cerr << "Error: Input image could not be loaded." << endl;
        return -1;
    }
    Mat image = inputImage.clone();

    Mat image_blob = blobFromImage(image, scale, Size(image.cols, image.rows), mean_v, swapRB, false);

    net.setInput(image_blob);
    Mat output = net.forward();

    // Post Processing
    Mat output_transposed(3, &output.size[1], CV_32F, output.ptr<float>());

    vector<Mat> channels;
    for (int i = 0; i < 3; ++i) {
        channels.push_back(Mat(output_transposed.size[1], output_transposed.size[2], CV_32F,
                                    output_transposed.ptr<float>(i)));
    }
    Mat outputImage;
    merge(channels, outputImage);
    outputImage.convertTo(outputImage, CV_8UC3, 255.0);
    cvtColor(outputImage, outputImage, COLOR_RGB2BGR);

    imshow("Input Image", inputImage);
    imshow("Output Image", outputImage);
    waitKey(0);
    return 0;
}
