#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "common.hpp"

using namespace cv;
using namespace std;
using namespace dnn;

const string about =
        "Use this script to run a classification model on a camera stream, video, image or in a directory\n\n"
        "Firstly, download required models using `download_models.py` (if not already done).\n"
        "To run:\n"
        "\t ./example_dnn_classification model_name --input=path/to/your/input/image/or/video (don't give --input flag if want to use device camera)\n"
        "Sample command:\n"
        "\t ./example_dnn_classification googlenet --input=path/to/image \n"
        "Model path can also be specified using --model argument";

const string param_keys =
    "{ help  h         |            | Print help message. }"
    "{ @alias          |            | An alias name of model to extract preprocessing parameters from models.yml file. }"
    "{ zoo             | models.yml | An optional path to file with preprocessing parameters }"
    "{ input i         |            | Path to input image or video file. Skip this argument to capture frames from a camera.}"
    "{ crop            |   false    | Preprocess input image by center cropping.}";

const string backend_keys = format(
    "{ backend          | 0 | Choose one of computation backends: "
                              "%d: automatically (by default), "
                              "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                              "%d: OpenCV implementation, "
                              "%d: VKCOM, "
                              "%d: CUDA, "
                              "%d: WebNN }", DNN_BACKEND_DEFAULT, DNN_BACKEND_INFERENCE_ENGINE, DNN_BACKEND_OPENCV, DNN_BACKEND_VKCOM, DNN_BACKEND_CUDA, DNN_BACKEND_WEBNN);
const string target_keys = format(
    "{ target           | 0 | Choose one of target computation devices: "
                              "%d: CPU target (by default), "
                              "%d: OpenCL, "
                              "%d: OpenCL fp16 (half-float precision), "
                              "%d: VPU, "
                              "%d: Vulkan, "
                              "%d: CUDA, "
                              "%d: CUDA fp16 (half-float preprocess) }", DNN_TARGET_CPU, DNN_TARGET_OPENCL, DNN_TARGET_OPENCL_FP16, DNN_TARGET_MYRIAD, DNN_TARGET_VULKAN, DNN_TARGET_CUDA, DNN_TARGET_CUDA_FP16);

string keys = param_keys + backend_keys + target_keys;

vector<string> classes;
vector<string> listImageFilesInDirectory(const string& directoryPath);

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);

    const string modelName = parser.get<String>("@alias");
    const string zooFile = findFile(parser.get<String>("zoo"));

    keys += genPreprocArguments(modelName, zooFile);
    parser = CommandLineParser(argc, argv, keys);
    parser.about(about);
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

    // Open file with classes names.
    string file = findFile(parser.get<String>("classes"));
    ifstream ifs(file.c_str());
    if (!ifs.is_open())
        CV_Error(Error::StsError, "File " + file + " not found");
    string line;
    while (getline(ifs, line))
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
    static const string kWinName = "Deep learning image classification in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);
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
        //! [Set input blob]
        net.setInput(blob);
        int classId;
        double confidence;
        TickMeter timeRecorder;
        timeRecorder.reset();
        Mat prob = net.forward();
        double t1;
        timeRecorder.start();
        prob = net.forward();
        Point classIdPoint;
        minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
        classId = classIdPoint.x;
        timeRecorder.stop();
        t1 = timeRecorder.getTimeMilli();
        timeRecorder.reset();
        string label = format("Inference time of 1 round: %.2f ms", t1);
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        // Print predicted class.
        label = format("%s: %.4f", (classes.empty() ? format("Class #%d", classId).c_str() :
                        classes[classId].c_str()),confidence);
        putText(frame, label, Point(0, 35), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        imshow(kWinName, frame);
        int key = waitKey(1000); // Wait for 1 second
        if (key == 'q' || key == 27) // Check if 'q' or 'ESC' is pressed
            break;
    }
    return 0;
}

vector<string> listImageFilesInDirectory(const string& folder_path) {
    vector<string> image_paths;
    // OpenCV object for reading the directory
    Ptr<String> image_paths_cv = new String();
    vector<String> fn;
    *image_paths_cv = folder_path + "/*.jpg"; // Change the extension according to the image types you want to read
    // Read the images from the directory
    glob(*image_paths_cv, fn, true); // true to search in subdirectories

    // Loop through the images
    for (size_t i = 0; i < fn.size(); i++) {
        image_paths.push_back(fn[i]);
    }
    return image_paths;
}