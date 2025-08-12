/*
This file is part of OpenCV project.
It is subject to the license terms in the LICENSE file found in the top-level directory
of this distribution and at http://opencv.org/license.html.

Copyright (C) 2025, Bigvision LLC.


This sample demonstrates super-resolution using the SeeMoreDetails model.
The model upscales images by 4x while enhancing details and reducing noise.
Supports image inputs only.

SeeMoreDetails Repo: https://github.com/eduardzamfir/seemoredetails
*/
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#include "common.hpp"

using namespace cv;
using namespace cv::dnn;
using namespace std;

const int WINDOW_OFFSET_X = 50;
const int WINDOW_OFFSET_Y = 50;
const int WINDOW_SPACING = 50;

const string param_keys =
    "{ help h          |                   | Print help message }"
    "{ @alias          | seemoredetails    | Model alias from models.yml }"
    "{ zoo             | ../dnn/models.yml | Path to models.yml file }"
    "{ input i         |  chicky_512.png   | Path to input image }"
    "{ model           |                   | Path to model file }";

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

static Mat postprocessOutput(const Mat &output, const Size &originalSize)
{
    Mat squeezed;
    if (output.dims == 4 && output.size[0] == 1)
    {
        vector<int> newShape = {output.size[1], output.size[2], output.size[3]};
        squeezed = output.reshape(0, newShape);
    }
    else
    {
        squeezed = output.clone();
    }

    Mat outputImage;
    vector<Mat> channels(3);
    for (int i = 0; i < 3; i++)
    {
        channels[2-i] = Mat(squeezed.size[1], squeezed.size[2], CV_32F,
                        squeezed.ptr<float>(i));
    }
    merge(channels, outputImage);

    outputImage = max(0.0, min(1.0, outputImage));
    outputImage.convertTo(outputImage, CV_8UC3, 255.0);

    Size targetSize(originalSize.width * 4, originalSize.height * 4);
    Mat result;
    resize(outputImage, result, targetSize);

    return result;
}

static Mat applySuperResolution(Net &net, const Mat &image, float scale, const Scalar &mean, bool swapRB, int width, int height)
{
    Mat blob = blobFromImage(image, scale, Size(width, height), mean, swapRB, false, CV_32F);

    net.setInput(blob);
    Mat output;
    net.forward(output);

    return postprocessOutput(output, Size(image.cols, image.rows));
}

static double calculateFontScale(const Mat &image)
{
    double baseScale = min(image.cols, image.rows) / 800.0;
    return max(0.5, baseScale);
}

static void processFrame(Net &net, Mat &frame, float scale, const Scalar &mean, bool swapRB, int width, int height)
{
    Mat result = applySuperResolution(net, frame, scale, mean, swapRB, width, height);

    double fontScale = calculateFontScale(frame);
    int thickness = max(1, (int)(fontScale * 2));

    putText(frame, "Original", Point(10, 30),
            FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 255, 0), thickness);

    double resultFontScale = calculateFontScale(result);
    int resultThickness = max(1, (int)(resultFontScale * 2));

    putText(result, "Super-Resolution 4x", Point(20, 50),
            FONT_HERSHEY_SIMPLEX, resultFontScale, Scalar(0, 255, 0), resultThickness);

    imshow("Input", frame);
    imshow("Super-Resolution", result);
}

int main(int argc, char **argv)
{
    const string about =
        "This sample demonstrates super-resolution using the SeeMore model.\n"
        "The model upscales images by 4x while enhancing details.\n\n"
        "Usage examples:\n"
        "\t./seemore_superres\n"
        "\t./seemore_superres --input=image.jpg\n"
        "\t./seemore_superres --input=../data/chicky_512.png\n";

    string keys = param_keys + backend_keys + target_keys;

    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        cout << about << endl;
        parser.printMessage();
        return 0;
    }

    string modelName = parser.get<String>("@alias");
    string zooFile = samples::findFile(parser.get<String>("zoo"));

    keys += genPreprocArguments(modelName, zooFile);
    parser = CommandLineParser(argc, argv, keys);

    float scale = parser.get<float>("scale");
    Scalar mean = parser.get<Scalar>("mean");
    bool swapRB = parser.get<bool>("rgb");
    String backend = parser.get<String>("backend");
    String target = parser.get<String>("target");
    String sha1 = parser.get<String>("sha1");
    string model = findModel(parser.get<String>("model"), sha1);
    int width = parser.get<int>("width");
    int height = parser.get<int>("height");
    string inputPath = findFile(parser.get<String>("input"));

    if (model.empty())
    {
        cerr << "Model file not found" << endl;
        return -1;
    }

    Net net;
    try
    {
        net = readNetFromONNX(model);
        net.setPreferableBackend(getBackendID(backend));
        net.setPreferableTarget(getTargetID(target));
    }
    catch (const Exception &e)
    {
        cerr << "Error loading model: " << e.what() << endl;
        return -1;
    }

    Mat testImage = imread(inputPath);
    if (testImage.empty())
    {
        cerr << "Cannot load image: " << inputPath << endl;
        return -1;
    }

    namedWindow("Input", WINDOW_NORMAL);
    namedWindow("Super-Resolution", WINDOW_NORMAL);
    moveWindow("Input", WINDOW_OFFSET_X, WINDOW_OFFSET_Y);
    moveWindow("Super-Resolution", WINDOW_OFFSET_X + testImage.cols + WINDOW_SPACING, WINDOW_OFFSET_Y);

    processFrame(net, testImage, scale, mean, swapRB, width, height);
    waitKey(0);
    destroyAllWindows();

    return 0;
}
