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

    const string keys =
        "{ help h          |     | Print help message. }"
        "{ input i         | ansel_adams3.jpg | Path to the input image }"
        "{ onnx_model_path |     | Path to the ONNX model. Required. }"
        "{ opencl          |     | enable OpenCL }";

    CommandLineParser parser(argc, argv,keys);
    parser.about(about);

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    string inputImagePath = parser.get<string>("input");
    string onnxModelPath = parser.get<string>("onnx_model_path");

    if (onnxModelPath.empty()) {
        cerr << "The path to the ONNX model is required!" << endl;
        return -1;
    }

    Mat img = imread(samples::findFile(inputImagePath));
    if (img.empty()) {
        cerr << "Could not read the image: " << inputImagePath << endl;
        return -1;
    }
    bool useOpenCL = parser.has("opencl");

    // Convert to Lab color space
    Mat imgLab;
    cvtColor(img, imgLab, COLOR_BGR2Lab);

    // Extract L channel and resize
    vector<Mat> labChannels(3);
    split(imgLab, labChannels);

    Mat imgL = labChannels[0];
    imgL.convertTo(imgL, CV_32F);
    Mat imgLResized;
    Mat lab, L, input;
    resize(imgL, imgLResized, Size(256, 256), 0, 0, INTER_CUBIC);
    imgLResized *= (100.0 / 255.0);  // Scale the L channel to 0-100 range

    // Prepare the model
    dnn::Net net = dnn::readNetFromONNX(onnxModelPath);
    if (useOpenCL)
        net.setPreferableTarget(DNN_TARGET_OPENCL);

    // Create blob from the image
    Mat blob;
    blob = dnn::blobFromImage(imgLResized, 1.0, Size(256, 256), Scalar(), false, false); //

    net.setInput(blob);

    // Run inference
    Mat result = net.forward();
    Size siz(result.size[2], result.size[3]);
    Mat a = Mat(siz, CV_32F, result.ptr(0,0));
    Mat b = Mat(siz, CV_32F, result.ptr(0,1));
    resize(a, a, img.size());
    resize(b, b, img.size());

    // merge, and convert back to BGR
    imgL *= (100.0/255.0);
    Mat color, chn[] = {imgL, a, b};

    // Proc
    merge(chn, 3, lab);
    cvtColor(lab, color, COLOR_Lab2BGR);

    imshow("output image", color);
    waitKey();
    return 0;

}
