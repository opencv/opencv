/*
This sample detects the query person in the given video file.

Authors of samples and Youtu ReID baseline:
        Xing Sun <winfredsun@tencent.com>
        Feng Zheng <zhengf@sustech.edu.cn>
        Xinyang Jiang <sevjiang@tencent.com>
        Fufu Yu <fufuyu@tencent.com>
        Enwei Zhang <miyozhang@tencent.com>

Copyright (C) 2020-2021, Tencent.
Copyright (C) 2020-2021, SUSTech.
Copyright (C) 2024, Bigvision LLC.

How to use:
    sample command to run:

        ./example_dnn_person_reid
    The system will ask you to mark the person to be tracked

    You can download ReID model using:
       `python download_models.py reid`
    and yolo model using:
       `python download_models.py yolov8`

    Set environment variable OPENCV_DOWNLOAD_CACHE_DIR to point to the directory where models are downloaded. Also, point OPENCV_SAMPLES_DATA_PATH to opencv/samples/data.
*/

#include <iostream>
#include <fstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "common.hpp"

using namespace cv;
using namespace cv::dnn;
using namespace std;

const string about = "Use this script for Person Re-identification using OpenCV. \n\n"
        "Firstly, download required models i.e. reid and yolov8 using `download_models.py` (if not already done). Set environment variable OPENCV_DOWNLOAD_CACHE_DIR to point to the directory where models are downloaded. Also, point OPENCV_SAMPLES_DATA_PATH to opencv/samples/data.\n"
        "To run:\n"
        "\t Example: ./example_dnn_person_reid reid\n\n"
        "Re-Identification model path can also be specified using --model argument. Detection model can be set using --yolo_model argument.\n\n";

const string param_keys =
    "{help    h  |                   | show help message}"
    "{ @alias    |       reid        | An alias name of model to extract preprocessing parameters from models.yml file. }"
    "{ zoo       | ../dnn/models.yml | An optional path to file with preprocessing parameters }"
    "{query   q  |                   | Path to target image. Skip this argument to select target in the video frame.}"
    "{input   i  |                   | video file path}"
    "{yolo_model |                   | Path to yolov8n.onnx}";

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


struct MatComparator
{
    bool operator()(const Mat &a, const Mat &b) const
    {
        return a.data < b.data; // This is a simple pointer comparison, not content!
    }
};

map<Mat, Rect, MatComparator> imgDict;
int height, width, yoloHeight, yoloWidth;
float scale, yoloScale;
bool swapRB, yoloSwapRB;
Scalar mean_v, stnd;


static void extractFeatures(vector<Mat> &imglist, Net &net, vector<Mat> &features)
{
    for (size_t st = 0; st < imglist.size(); st++)
    {
        Mat blob;
        blobFromImage(imglist[st], blob, scale, Size(width, height), mean_v, swapRB, false, CV_32F);

        // Check if standard deviation values are non-zero
        if (stnd[0] != 0.0 && stnd[1] != 0.0 && stnd[2] != 0.0)
        {
            // Divide blob by std for each channel
            divide(blob, stnd, blob);
        }
        net.setInput(blob);
        Mat out=net.forward();
        vector<int> s {out.size[0], out.size[1]};
        out = out.reshape(1, s);
        for (int i = 0; i < out.rows; i++)
        {
            Mat norm_features;
            normalize(out.row(i), norm_features, 1.0, 0.0, NORM_L2);
            features.push_back(norm_features);
        }
    }
    return;
}

static int findMatching(const Mat &queryFeatures, const vector<Mat> &galleryFeatures)
{
    if (queryFeatures.empty() || galleryFeatures.empty())
        return -1; // No valid index if either feature list is empty

    int bestIndex = -1;
    float maxSimilarity = FLT_MIN;

    for (int j = 0; j < (int)galleryFeatures.size(); j++)
    {
        float currentSimilarity = static_cast<float>(queryFeatures.dot(galleryFeatures[j]));
        if (currentSimilarity > maxSimilarity)
        {
            maxSimilarity = currentSimilarity;
            bestIndex = j;
        }
    }
    return bestIndex;
}

static void yoloDetector(Mat &frame, Net &net, vector<Mat>& images)
{
    int ht = frame.rows;
    int wt = frame.cols;

    int length = max(ht, wt);

    Mat image = Mat::zeros(Size(length, length), frame.type());

    frame.copyTo(image(Rect(0, 0, wt, ht)));

    // Calculate the scale
    double norm_scale = static_cast<double>(length) / yoloWidth;

    Mat blob;
    blobFromImage(image, blob, yoloScale, Size(yoloWidth, yoloHeight), Scalar(), yoloSwapRB, false, CV_32F);
    net.setInput(blob);

    vector<Mat> outputs;
    net.forward(outputs);
    Mat reshapedMatrix = outputs[0].reshape(0, 84);  // Reshape to 2D (84 rows, 8400 columns)

    Mat outputTransposed;
    transpose(reshapedMatrix, outputTransposed);

    int rows = outputTransposed.rows;

    vector<Rect2d> boxes;
    vector<float> scores;
    vector<int> class_ids;

    for (int i = 0; i < rows; i++) {
        double minScore, maxScore;
        Point minClassLoc, maxClassLoc;
        minMaxLoc(outputTransposed.row(i).colRange(4, outputTransposed.cols), &minScore, &maxScore, &minClassLoc, &maxClassLoc);

        if (maxScore >= 0.25 && maxClassLoc.x == 0) {
            double centerX = outputTransposed.at<float>(i, 0);
            double centerY = outputTransposed.at<float>(i, 1);
            double w = outputTransposed.at<float>(i, 2);
            double h = outputTransposed.at<float>(i, 3);

            Rect2d box(
                centerX - 0.5 * w, // x
                centerY - 0.5 * h, // y
                w, // width
                h // height
            );
            boxes.push_back(box);
            scores.push_back(static_cast<float>(maxScore));
            class_ids.push_back(maxClassLoc.x); // x location gives the index
        }
    }

    // Apply Non-Maximum Suppression
    vector<int> indexes;
    NMSBoxes(boxes, scores, 0.25f, 0.45f, indexes, 0.5f, 0);

    images.resize(indexes.size());
    for (size_t i = 0; i < indexes.size(); i++) {
        int index = indexes[i];
        int x = static_cast<int>(round(boxes[index].x * norm_scale));
        int y = static_cast<int>(round(boxes[index].y * norm_scale));
        int w = static_cast<int>(round(boxes[index].width * norm_scale));
        int h = static_cast<int>(round(boxes[index].height * norm_scale));
        // Make sure the box is within the frame
        x = max(0, x);
        y = max(0, y);
        w = min(w, frame.cols - x);
        h = min(h, frame.rows - y);

        // Crop the image
        Rect roi(x, y, w, h); // Define a region of interest
        images[i] = frame(roi); // Crop the region from the frame
        imgDict[images[i]] = roi;
    }
    return;
}

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
    keys += genPreprocArguments(modelName, zooFile, "yolo_");
    parser = CommandLineParser(argc, argv, keys);
    parser.about("Use this script to run ReID networks using OpenCV.");

    const string sha1 = parser.get<String>("sha1");
    const string yoloSha1 = parser.get<String>("yolo_sha1");
    const string modelPath = findModel(parser.get<String>("model"), sha1);
    const string queryImagePath = parser.get<String>("query");
    string videoPath = parser.get<String>("input");
    const string yoloPath = findModel(parser.get<String>("yolo_model"), yoloSha1);
    const string backend = parser.get<String>("backend");
    const string target = parser.get<String>("target");
    height = parser.get<int>("height");
    width = parser.get<int>("width");
    yoloHeight = parser.get<int>("yolo_height");
    yoloWidth = parser.get<int>("yolo_width");
    scale = parser.get<float>("scale");
    yoloScale = parser.get<float>("yolo_scale");
    swapRB = parser.get<bool>("rgb");
    yoloSwapRB = parser.get<bool>("yolo_rgb");
    mean_v = parser.get<Scalar>("mean");
    stnd = parser.get<Scalar>("std");
    int stdSize = 20;
    int stdWeight = 400;
    int stdImgSize = 512;
    int imgWidth = -1; // Initialization
    int fontSize = 50;
    int fontWeight = 500;

    EngineType engine = ENGINE_AUTO;
    if (backend != "default" || target != "cpu"){
        engine = ENGINE_CLASSIC;
    }
    Net reidNet = readNetFromONNX(modelPath, engine);
    reidNet.setPreferableBackend(getBackendID(backend));
    reidNet.setPreferableTarget(getTargetID(target));

    if(yoloPath.empty()){
        cout<<"[ERROR] Please pass path to yolov8.onnx model file using --yolo_model."<<endl;
        return -1;
    }
    Net net = readNetFromONNX(yoloPath, engine);

    FontFace fontFace("sans");

    VideoCapture cap;
    if (!videoPath.empty()){
        videoPath = findFile(videoPath);
        cap.open(videoPath);
    }
    else
        cap.open(0);

    if (!cap.isOpened()) {
        cerr << "Error: Video could not be opened." << endl;
        return -1;
    }
    vector<Mat> queryImages;
    Mat queryImg;
    if (!queryImagePath.empty()) {
        queryImg = imread(queryImagePath);
        if (queryImg.empty()) {
            cerr << "Error: Query image could not be loaded." << endl;
            return -1;
        }
        queryImages.push_back(queryImg);
    } else {
        Mat image;
        for(;;) {
            cap.read(image);
            if (image.empty()) {
                cerr << "Error reading the video" << endl;
                return -1;
            }
            if (imgWidth == -1){
                imgWidth = min(image.rows, image.cols);
                fontSize = min(fontSize, (stdSize*imgWidth)/stdImgSize);
                fontWeight = min(fontWeight, (stdWeight*imgWidth)/stdImgSize);
            }

            const string label = "Press space bar to pause video to draw bounding box.";
            Rect r = getTextSize(Size(), label, Point(), fontFace, fontSize, fontWeight);
            r.height += 2 * fontSize; // padding
            r.width += 10; // padding
            rectangle(image, r, Scalar::all(255), FILLED);
            putText(image, label, Point(10, fontSize), Scalar(0,0,0), fontFace, fontSize, fontWeight);
            putText(image, "Press space bar after selecting.", Point(10, 2*fontSize), Scalar(0,0,0), fontFace, fontSize, fontWeight);
            imshow("TRACKING", image);
            int key = waitKey(200);
            if(key == ' '){
                Rect rect = selectROI("TRACKING", image);

                if (rect.width > 0 && rect.height > 0) {
                    queryImg = image(rect).clone();
                    queryImages.push_back(queryImg);
                    break;
                }
            }
            if (key == 'q' || key == 27) {
                return 0;
            }
        }
    }

    Mat frame;
    vector<Mat> queryFeatures;
    extractFeatures(queryImages, reidNet, queryFeatures);

    vector<Mat> detectedImages;
    vector<Mat> galleryFeatures;
    for(;;) {
        if (!cap.read(frame) || frame.empty()) {
            break;
        }
        if (imgWidth == -1){
            imgWidth = min(frame.rows, frame.cols);
            fontSize = min(fontSize, (stdSize*imgWidth)/stdImgSize);
            fontWeight = min(fontWeight, (stdWeight*imgWidth)/stdImgSize);
        }
        detectedImages.clear();
        galleryFeatures.clear();

        yoloDetector(frame, net, detectedImages);
        extractFeatures(detectedImages, reidNet, galleryFeatures);

        int match_idx = findMatching(queryFeatures[0], galleryFeatures);
        if (match_idx != -1 && static_cast<int>(detectedImages.size()) > match_idx) {
            Mat matchImg = detectedImages[match_idx];
            Rect bbox = imgDict[matchImg];
            rectangle(frame, bbox, Scalar(0, 0, 255), 2);
            putText(frame, "Target", Point(bbox.x, bbox.y - 10), Scalar(0,0,255), fontFace, fontSize, fontWeight);
        }
        const string label = "Tracking";
        Rect r = getTextSize(Size(), label, Point(), fontFace, fontSize, fontWeight);
        r.height += fontSize; // padding
        r.width += 10; // padding
        rectangle(frame, r, Scalar::all(255), FILLED);
        putText(frame, label, Point(10, fontSize), Scalar(0,0,0), fontFace, fontSize, fontWeight);
        imshow("TRACKING", frame);
        int key = waitKey(30);
        if (key == 'q' || key == 27) {
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
