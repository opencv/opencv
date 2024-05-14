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

How to use:
    sample command to run:

        ./person_reid --query=/path/to/query/image --video=/path/to/videofile --model=path/to/youtu_reid_baseline_medium.onnx --yolo=path/to/yolov8n.onnx

    You can download a baseline ReID model from:
        https://github.com/ReID-Team/ReID_extra_testdata

*/

#include <iostream>
#include <fstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;

std::string param_keys =
    "{help    h  |                 | show help message}"
    "{model   m  |                 | network model}"
    "{query q    |                 | path to target image}"
    "{video v    |                 | video file path}"
    "{yolo       |                 | Path to yolov8.onnx}"
    "{resize_h   | 256             | resize input to specific height}"
    "{resize_w   | 128             | resize input to specific width}";

const string backend_keys = cv::format(
    "{ backend   | 0 | Choose one of computation backends: "
    "%d: automatically (by default), "
    "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
    "%d: OpenCV implementation, "
    "%d: VKCOM, "
    "%d: CUDA }",
    DNN_BACKEND_DEFAULT, DNN_BACKEND_INFERENCE_ENGINE, DNN_BACKEND_OPENCV, DNN_BACKEND_VKCOM, DNN_BACKEND_CUDA);

const string target_keys = cv::format(
    "{ target    | 0 | Choose one of target computation devices: "
    "%d: CPU target (by default), "
    "%d: OpenCL, "
    "%d: OpenCL fp16 (half-float precision), "
    "%d: VPU, "
    "%d: Vulkan, "
    "%d: CUDA, "
    "%d: CUDA fp16 (half-float preprocess) }",
    DNN_TARGET_CPU, DNN_TARGET_OPENCL, DNN_TARGET_OPENCL_FP16, DNN_TARGET_MYRIAD, DNN_TARGET_VULKAN, DNN_TARGET_CUDA, DNN_TARGET_CUDA_FP16);

string keys = param_keys + backend_keys + target_keys;

static Mat preprocess(const Mat &img)
{
    const double mean[3] = {0.485, 0.456, 0.406};
    const double std[3] = {0.229, 0.224, 0.225};
    Mat ret = Mat(img.rows, img.cols, CV_32FC3);
    for (int y = 0; y < ret.rows; y++)
    {
        for (int x = 0; x < ret.cols; x++)
        {
            for (int c = 0; c < 3; c++)
            {
                ret.at<Vec3f>(y,x)[c] = (float)((img.at<Vec3b>(y,x)[c] / 255.0 - mean[2 - c]) / std[2 - c]);
            }
        }
    }
    return ret;
}

static std::vector<float> normalization(const std::vector<float> &feature)
{
    std::vector<float> ret;
    float sum = 0.0;
    for (int i = 0; i < (int)feature.size(); i++)
    {
        sum += feature[i] * feature[i];
    }
    sum = sqrt(sum);
    for (int i = 0; i < (int)feature.size(); i++)
    {
        ret.push_back(feature[i] / sum);
    }
    return ret;
}

static void extractFeatures(std::vector<cv::Mat> &imglist, Net *net, const int &resize_h, const int &resize_w, std::vector<std::vector<float>> &features)
{
    for (int st = 0; st < (int)imglist.size(); st += 32)
    {
        std::vector<Mat> batch;
        for (int delta = 0; delta < 32 && st + delta < (int)imglist.size(); delta++)
        {
            Mat img = imglist[st + delta];
            batch.push_back(preprocess(img));
        }
        Mat blob = dnn::blobFromImages(batch, 1.0, Size(resize_w, resize_h), Scalar(0.0,0.0,0.0), true, false, CV_32F);
        net->setInput(blob);
        Mat out = net->forward();
        for (int i = 0; i < (int)out.size().height; i++)
        {
            std::vector<float> temp_feature;
            for (int j = 0; j < (int)out.size().width; j++)
            {
                temp_feature.push_back(out.at<float>(i,j));
            }
            features.push_back(normalization(temp_feature));
        }
    }
    return;
}

static float similarity(const std::vector<float> &feature1, const std::vector<float> &feature2)
{
    float result = 0.0;
    for (int i = 0; i < (int)feature1.size(); i++)
    {
        result += feature1[i] * feature2[i];
    }
    return result;
}

static int getTopK(const std::vector<std::vector<float>> &queryFeatures, const std::vector<std::vector<float>> &galleryFeatures)
{
    if (queryFeatures.empty() || galleryFeatures.empty())
        return -1; // No valid index if either feature list is empty

    int bestIndex = -1;
    float maxSimilarity = -1.0;

    const std::vector<float> &query = queryFeatures[0];

    for (int j = 0; j < (int)galleryFeatures.size(); j++)
    {
        float currentSimilarity = similarity(query, galleryFeatures[j]);
        if (currentSimilarity > maxSimilarity)
        {
            maxSimilarity = currentSimilarity;
            bestIndex = j;
        }
    }

    return bestIndex;
}

struct MatComparator
{
    bool operator()(const cv::Mat &a, const cv::Mat &b) const
    {
        return a.data < b.data; // This is a simple pointer comparison, not content!
    }
};

std::map<cv::Mat, cv::Rect, MatComparator> imgDict;

static std::vector<cv::Mat> yoloDetector(cv::Mat &frame, cv::dnn::Net &net)
{
    int height = frame.rows;
    int width = frame.cols;

    int length = std::max(height, width);

    cv:: Mat image = cv::Mat::zeros(cv::Size(length, length), frame.type());

    frame.copyTo(image(cv::Rect(0, 0, width, height)));

    // Calculate the scale
    double scale = static_cast<double>(length) / 640.0;

    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0/255.0, cv::Size(640, 640), cv::Scalar(), true, false, CV_32F);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs);
    cv::Mat reshapedMatrix = outputs[0].reshape(0, 84);  // Reshape to 2D (84 rows, 8400 columns)

    cv::Mat outputTransposed;
    cv::transpose(reshapedMatrix, outputTransposed);

    int rows = outputTransposed.rows;

    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    for (int i = 0; i < rows; i++) {
        double minScore, maxScore;
        cv::Point minClassLoc, maxClassLoc;
        cv::minMaxLoc(outputTransposed.row(i).colRange(4, outputTransposed.cols), &minScore, &maxScore, &minClassLoc, &maxClassLoc);

        if (maxScore >= 0.25) {
            double centerX = outputTransposed.at<float>(i, 0);
            double centerY = outputTransposed.at<float>(i, 1);
            double w = outputTransposed.at<float>(i, 2);
            double h = outputTransposed.at<float>(i, 3);

            cv::Rect2d box(
                centerX - 0.5 * w, // x
                centerY - 0.5 * h, // y
                w, // width
                h // height
            );
            boxes.push_back(box);
            scores.push_back(maxScore);
            class_ids.push_back(maxClassLoc.x); // x location gives the index
        }
    }

    // Apply Non-Maximum Suppression
    std::vector<int> indexes;
    NMSBoxes(boxes, scores, 0.25, 0.45, indexes, 0.5, 1);

    std::vector<cv::Mat> images;
    for (int index : indexes) {
        int x = round(boxes[index].x * scale);
        int y = round(boxes[index].y * scale);
        int w = round(boxes[index].width * scale);
        int h = round(boxes[index].height * scale);

        // Make sure the box is within the frame
        x = std::max(0, x);
        y = std::max(0, y);
        w = std::min(w, frame.cols - x);
        h = std::min(h, frame.rows - y);

        // Crop the image
        cv::Rect roi(x, y, w, h); // Define a region of interest
        cv::Mat crop_img = frame(roi); // Crop the region from the frame
        images.push_back(crop_img);
        imgDict[crop_img] = roi;
    }

    return images;
}

static void extractFrames(const std::string &queryImgPath, const std::string &videoPath, Net *reidNet, const std::string &yoloPath, int resize_h = 384, int resize_w = 128)
{
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Video could not be opened." << std::endl;
        return;
    }

    cv::dnn::Net net = cv::dnn::readNetFromONNX(yoloPath);

    std::vector<cv::Mat> frames;

    cv::Mat queryImg = cv::imread(queryImgPath);
    if (queryImg.empty())
    {
        std::cerr << "Error: Query image could not be loaded." << std::endl;
        return;
    }
    std::vector<cv::Mat> queryImages = {queryImg};

    cv::Mat frame;
    for(;;)
    {
        if (!cap.read(frame) || frame.empty())
        {
            break;
        }

        std::vector<cv::Mat> detectedImages = yoloDetector(frame, net);
        std::vector<std::vector<float>> queryFeatures;
        extractFeatures(queryImages, reidNet, resize_h, resize_w, queryFeatures);
        std::vector<std::vector<float>> galleryFeatures;
        extractFeatures(detectedImages, reidNet, resize_h, resize_w, galleryFeatures);

        int topk_idx = getTopK(queryFeatures, galleryFeatures);
        if (topk_idx != -1 && static_cast<int>(detectedImages.size()) > topk_idx) //Check if topk_idx is valid
        {
            cv::Mat topImg = detectedImages[topk_idx];
            cv::Rect bbox = imgDict[topImg];
            cv::rectangle(frame, bbox, cv::Scalar(0, 0, 255), 2);
            cv::putText(frame, "Target", cv::Point(bbox.x, bbox.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("Image", frame);
        if (cv::waitKey(1) == 'q' || cv::waitKey(1) == 27)
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
}


int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv, keys);

    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    parser = CommandLineParser(argc, argv, keys);
    parser.about("Use this script to run ReID networks using OpenCV.");

    const std::string modelPath = parser.get<String>("model");
    const std::string queryImagePath = parser.get<String>("query");
    const std::string videoPath = parser.get<String>("video");
    const std::string yoloPath = parser.get<String>("yolo");
    const int backend = parser.get<int>("backend");
    const int target = parser.get<int>("target");
    const int resize_h = parser.get<int>("resize_h");
    const int resize_w = parser.get<int>("resize_w");

    dnn::Net net = dnn::readNet(modelPath);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    extractFrames(queryImagePath, videoPath, &net, yoloPath, resize_h, resize_w);
    return 0;
}
