#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "common.hpp"

std::string param_keys =
    "{ help  h          | | Print help message. }"
    "{ @alias           | | An alias name of model to extract preprocessing parameters from models.yml file. }"
    "{ zoo              | models.yml | An optional path to file with preprocessing parameters }"
    "{ input i          | | Path to input image or video file. Skip this argument to capture frames from a camera.}"
    "{ crop             | false | Preprocess input image by center cropping.}"
    "{ needSoftmax      | false | Use Softmax to post-process the output of the net.}";
std::string backend_keys = cv::format(
    "{ backend          | 0 | Choose one of computation backends: "
                              "%d: automatically (by default), "
                              "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                              "%d: OpenCV implementation, "
                              "%d: VKCOM, "
                              "%d: CUDA, "
                              "%d: WebNN }", cv::dnn::DNN_BACKEND_DEFAULT, cv::dnn::DNN_BACKEND_INFERENCE_ENGINE, cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_BACKEND_VKCOM, cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_BACKEND_WEBNN);
std::string target_keys = cv::format(
    "{ target           | 0 | Choose one of target computation devices: "
                              "%d: CPU target (by default), "
                              "%d: OpenCL, "
                              "%d: OpenCL fp16 (half-float precision), "
                              "%d: VPU, "
                              "%d: Vulkan, "
                              "%d: CUDA, "
                              "%d: CUDA fp16 (half-float preprocess) }", cv::dnn::DNN_TARGET_CPU, cv::dnn::DNN_TARGET_OPENCL, cv::dnn::DNN_TARGET_OPENCL_FP16, cv::dnn::DNN_TARGET_MYRIAD, cv::dnn::DNN_TARGET_VULKAN, cv::dnn::DNN_TARGET_CUDA, cv::dnn::DNN_TARGET_CUDA_FP16);

std::string keys = param_keys + backend_keys + target_keys;

using namespace cv;
using namespace std;
using namespace dnn;

std::vector<std::string> classes;
vector<string> listImageFilesInDirectory(const string& directoryPath);

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);

    const std::string modelName = parser.get<String>("@alias");
    const std::string zooFile = parser.get<String>("zoo");

    keys += genPreprocArguments(modelName, zooFile);

    parser = CommandLineParser(argc, argv, keys);
    parser.about("Use this script to run classification deep learning networks using OpenCV.");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    float scale = parser.get<float>("scale");
    Scalar mean = parser.get<Scalar>("mean");
    Scalar std = parser.get<Scalar>("std");
    bool swapRB = parser.get<bool>("rgb");
    bool crop = parser.get<bool>("crop");
    int inpWidth = parser.get<int>("width");
    int inpHeight = parser.get<int>("height");
    String model = findFile(parser.get<String>("model"));
    int backendId = parser.get<int>("backend");
    int targetId = parser.get<int>("target");
    bool needSoftmax = parser.get<bool>("needSoftmax");
    std::cout<<"mean: "<<mean<<std::endl;
    std::cout<<"std: "<<std<<std::endl;

    // Open file with classes names.

    std::string file = findFile(parser.get<String>("classes"));
    std::ifstream ifs(file.c_str());
    if (!ifs.is_open())
        CV_Error(Error::StsError, "File " + file + " not found");
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }


    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }
    CV_Assert(!model.empty());

    //! [Read and initialize network]
    Net net = readNetFromONNX(model);
    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);

    // Create a window
    static const std::string kWinName = "Deep learning image classification in OpenCV";
    cv::namedWindow(kWinName, WINDOW_NORMAL);


    VideoCapture cap;
    vector<string> imageFiles;
    size_t currentImageIndex = 0;

    if (parser.has("input")) {
        string input = parser.get<String>("input");

        if (input.find('.')==string::npos) {
            // Input is a directory, list all image files
            imageFiles = listImageFilesInDirectory(input);
            if (imageFiles.empty()) {
                cout << "No images found in the directory." << endl;
                return -1;
            }
        } else {
            // Input is not a directory, try to open as video or image
            cap.open(input);
            if (!cap.isOpened()) {
                cout << "Failed to open the input." << endl;
                return -1;
            }
        }
    } else {
        cap.open(0); // Open default camera
    }


    // Process frames.
    Mat frame, blob;
    for(;;)
    {
        if (!imageFiles.empty()) {
            // Handling directory of images
            if (currentImageIndex >= imageFiles.size()) {
                waitKey();
                break; // Exit if all images are processed
            }
            frame = imread(imageFiles[currentImageIndex++]);
            if(frame.empty()){
                cout<<"Cannot open file"<<endl;
                continue;
            }
        } else {
            // Handling video or single image
            cap >> frame;
        }
        if (frame.empty())
        {
            break;
        }

        //! [Create a 4D blob from a frame]
        blobFromImage(frame, blob, scale, Size(inpWidth, inpHeight), mean, swapRB, crop);

        // Check std values.
        if (std.val[0] != 0.0 && std.val[1] != 0.0 && std.val[2] != 0.0)
        {
            // Divide blob by std.
            divide(blob, std, blob);
        }
        //! [Create a 4D blob from a frame]

        //! [Set input blob]
        net.setInput(blob);

        int classId;
        double confidence;
        cv::TickMeter timeRecorder;
        timeRecorder.reset();
        Mat prob = net.forward();
        double t1;
        timeRecorder.start();
        prob = net.forward();
        timeRecorder.stop();
        t1 = timeRecorder.getTimeMilli();

        timeRecorder.reset();
        for(int i = 0; i < 200; i++) {
            //! [Make forward pass]
            timeRecorder.start();
            prob = net.forward();
            timeRecorder.stop();

            //! [Get a class with a highest score]
            Point classIdPoint;
            minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
            classId = classIdPoint.x;
        }
        if (needSoftmax == true)
        {
            float maxProb = 0.0;
            float sum = 0.0;
            Mat softmaxProb;

            maxProb = *std::max_element(prob.begin<float>(), prob.end<float>());
            cv::exp(prob-maxProb, softmaxProb);
            sum = (float)cv::sum(softmaxProb)[0];
            softmaxProb /= sum;
            Point classIdPoint;
            minMaxLoc(softmaxProb.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
            classId = classIdPoint.x;
        }
        std::string label = format("Inference time of 1 round: %.2f ms", t1);
        std::string label2 = format("Average time of 200 rounds: %.2f ms", timeRecorder.getTimeMilli()/200);
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        putText(frame, label2, Point(0, 35), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

        // Print predicted class.
        label = format("%s: %.4f", (classes.empty() ? format("Class #%d", classId).c_str() :
                                                      classes[classId].c_str()),
                                   confidence);
        putText(frame, label, Point(0, 55), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

        imshow(kWinName, frame);
        int key = cv::waitKey(1000); // Wait for 1 second
        if (key == 'q' || key == 27) // Check if 'q' or 'ESC' is pressed
            break;
    }
    return 0;
}


std::vector<std::string> listImageFilesInDirectory(const std::string& folder_path) {
    std::vector<std::string> image_paths;

    // OpenCV object for reading the directory
    cv::Ptr<cv::String> image_paths_cv = new cv::String();
    std::vector<cv::String> fn;
    *image_paths_cv = folder_path + "/*.jpg"; // Change the extension according to the image types you want to read

    // Read the images from the directory
    cv::glob(*image_paths_cv, fn, true); // true to search in subdirectories

    // Loop through the images
    for (size_t i = 0; i < fn.size(); i++) {
        image_paths.push_back(fn[i]);
    }

    return image_paths;
}