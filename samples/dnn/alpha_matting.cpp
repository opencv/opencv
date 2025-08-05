/*
 * This file is part of OpenCV project.
 * It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution and at http://opencv.org/license.html.
 *
 * Copyright (C) 2025, Bigvision LLC.
 *
 * @file alpha_matting.cpp
 * @brief MODNet Alpha Matting using OpenCV DNN
 *
 * This sample demonstrates human portrait alpha matting using MODNet model.
 * MODNet is a trimap-free portrait matting method that can produce high-quality
 * alpha mattes for portrait images in real-time.
 *
 * Reference:
 *      Github: https://github.com/ZHKKKe/MODNet
 *
 * Usage:
 *   ./example_dnn_alpha_matting --input=image.jpg                      # Process image
 *
 * Requirements:
 *   - OpenCV >= 5.0.0 with DNN module
 *   - MODNet ONNX model
 */

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>

#include "common.hpp"

using namespace cv;
using namespace cv::dnn;
using namespace std;

const string about =
    "This sample demonstrates human portrait alpha matting using MODNet model.\n"
    "MODNet is a trimap-free portrait matting method that can produce high-quality\n"
    "alpha mattes for portrait images in real-time.\n\n"
    "Usage examples:\n"
    "\t./example_alpha_matting --input=image.jpg\n"
    "\t./example_alpha_matting modnet (using config alias)\n\n"
    "To download the MODNet model, run: python download_models.py modnet\n"
    "Press any key to exit \n";

const string param_keys =
    "{ help h          |                   | Print help message }"
    "{ @alias          | modnet            | An alias name of model to extract preprocessing parameters from models.yml file }"
    "{ zoo             | ../dnn/models.yml | An optional path to file with preprocessing parameters }"
    "{ input i         | messi5.jpg | Path to input image file }"
    "{ model           |                   | Path to MODNet ONNX model file }";

const string backend_keys = format(
    "{ backend         | default | Choose one of computation backends: "
    "default: automatically (by default), "
    "openvino: Intel's Deep Learning Inference Engine, "
    "opencv: OpenCV implementation, "
    "vkcom: VKCOM, "
    "cuda: CUDA, "
    "webnn: WebNN }");

const string target_keys = format(
    "{ target          | cpu | Choose one of target computation devices: "
    "cpu: CPU target (by default), "
    "opencl: OpenCL, "
    "opencl_fp16: OpenCL fp16 (half-float precision), "
    "vpu: VPU, "
    "vulkan: Vulkan, "
    "cuda: CUDA, "
    "cuda_fp16: CUDA fp16 (half-float precision) }");

string keys = param_keys + backend_keys + target_keys;

static void loadModel(const string modelPath, String backend, String target, Net &net, EngineType engine)
{
    net = readNetFromONNX(modelPath, engine);
    net.setPreferableBackend(getBackendID(backend));
    net.setPreferableTarget(getTargetID(target));
}

static void postprocess(const Mat &image, const Mat &alpha_output, Mat &alpha_mask)
{
    int h = image.rows;
    int w = image.cols;

    Mat alpha;
    if (alpha_output.dims == 4 && alpha_output.size[0] == 1 && alpha_output.size[1] == 1)
    {
        alpha = alpha_output.reshape(0, {alpha_output.size[2], alpha_output.size[3]});
    }
    else
    {
        alpha = alpha_output.clone();
    }

    resize(alpha, alpha, Size(w, h));

    alpha = cv::min(cv::max(alpha, 0.0), 1.0);
    alpha.convertTo(alpha_mask, CV_8U, 255.0);
}

static void processImage(const Mat &image, Mat &alpha_mask, Mat &composite, Net &net,
                         float scale, int width, int height, const Scalar &mean, bool swapRB)
{
    if (image.empty())
        return;

    Mat blob = blobFromImage(image, scale, Size(width, height), mean, swapRB, false, CV_32F);

    if (abs(scale - 1.0f) < 1e-6)
    {
        blob = (blob / 127.5f) - 1.0f;
    }

    net.setInput(blob);
    Mat output = net.forward();
    postprocess(image, output, alpha_mask);

    Mat alpha_3ch;
    cvtColor(alpha_mask, alpha_3ch, COLOR_GRAY2BGR);
    alpha_3ch.convertTo(alpha_3ch, CV_32F, 1.0 / 255.0);

    Mat image_f;
    image.convertTo(image_f, CV_32F);
    multiply(image_f, alpha_3ch, composite);
    composite.convertTo(composite, CV_8U);
}

static void setupWindows()
{
    namedWindow("Original", WINDOW_AUTOSIZE);
    namedWindow("Alpha Mask", WINDOW_AUTOSIZE);
    namedWindow("Composite", WINDOW_AUTOSIZE);
    moveWindow("Alpha Mask", 200, 0);
    moveWindow("Composite", 400, 0);
}

int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv, keys);

    if (parser.has("help"))
    {
        cout << about << endl;
        parser.printMessage();
        return 0;
    }

    string modelName = parser.get<String>("@alias");
    string zooFile = parser.get<String>("zoo");

    zooFile = findFile(zooFile);

    keys += genPreprocArguments(modelName, zooFile);

    parser = CommandLineParser(argc, argv, keys);

    int input_width = parser.get<int>("width");
    int input_height = parser.get<int>("height");
    float scale_factor = parser.get<float>("scale");
    Scalar mean_values = parser.get<Scalar>("mean");
    bool swapRB = parser.get<bool>("rgb");
    String backend = parser.get<String>("backend");
    String target = parser.get<String>("target");
    String sha1 = parser.get<String>("sha1");

    string model = findModel(parser.get<String>("model"), sha1);

    parser.about(about);

    EngineType engine = ENGINE_AUTO;
    if (backend != "default" || target != "cpu")
    {
        engine = ENGINE_CLASSIC;
    }

    Net net;
    loadModel(model, backend, target, net, engine);

    string input_path = samples::findFile(parser.get<String>("input"));
    Mat image = imread(input_path);
    if (image.empty())
    {
        cout << "[ERROR] Cannot load input image: " << input_path << endl;
        return -1;
    }

    setupWindows();

    cout << "Processing image: " << input_path << endl;
    cout << "Press any key to exit" << endl;

    Mat alpha_mask, composite;

    processImage(image, alpha_mask, composite, net, scale_factor, input_width, input_height, mean_values, swapRB);

    imshow("Original", image);
    imshow("Alpha Mask", alpha_mask);
    imshow("Composite", composite);

    waitKey(0);
    destroyAllWindows();
    return 0;
}
