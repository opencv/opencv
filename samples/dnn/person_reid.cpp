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

        ./person_reid --query=/path/to/query/image --video=/path/to/videofile --model=path/to/youtu_reid_baseline_medium.onnx --yolo=path/to/yolov3.weights --cfg=/path/to/yolov3.cfg

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

std::string param_keys =
    "{help    h  |                 | show help message}"
    "{model   m  |                 | network model}"
    "{query q    |                 | path to target image}"
    "{video v    |                 | video file path}"
    "{yolo       |                 | Path to yolov3.weights}"
    "{cfg        |                 | Path to yolov3.cfg}"
    "{batch_size | 32              | batch size of each inference}"
    "{resize_h   | 256             | resize input to specific height}"
    "{resize_w   | 128             | resize input to specific width}";

std::string backend_keys = cv::format(
    "{ backend   | 0 | Choose one of computation backends: "
    "%d: automatically (by default), "
    "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
    "%d: OpenCV implementation, "
    "%d: VKCOM, "
    "%d: CUDA }",
    cv::dnn::DNN_BACKEND_DEFAULT, cv::dnn::DNN_BACKEND_INFERENCE_ENGINE, cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_BACKEND_VKCOM, cv::dnn::DNN_BACKEND_CUDA);
std::string target_keys = cv::format(
    "{ target    | 0 | Choose one of target computation devices: "
    "%d: CPU target (by default), "
    "%d: OpenCL, "
    "%d: OpenCL fp16 (half-float precision), "
    "%d: VPU, "
    "%d: Vulkan, "
    "%d: CUDA, "
    "%d: CUDA fp16 (half-float preprocess) }",
    cv::dnn::DNN_TARGET_CPU, cv::dnn::DNN_TARGET_OPENCL, cv::dnn::DNN_TARGET_OPENCL_FP16, cv::dnn::DNN_TARGET_MYRIAD, cv::dnn::DNN_TARGET_VULKAN, cv::dnn::DNN_TARGET_CUDA, cv::dnn::DNN_TARGET_CUDA_FP16);
std::string keys = param_keys + backend_keys + target_keys;

namespace cv
{
    namespace reid
    {

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
                        ret.at<Vec3f>(y, x)[c] = (float)((img.at<Vec3b>(y, x)[c] / 255.0 - mean[2 - c]) / std[2 - c]);
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

        static void extractFeatures(std::vector<cv::Mat> &imglist, Net *net, const int &batch_size, const int &resize_h, const int &resize_w, std::vector<std::vector<float>> &features)
        {
            for (int st = 0; st < (int)imglist.size(); st += batch_size)
            {
                std::vector<Mat> batch;
                for (int delta = 0; delta < batch_size && st + delta < (int)imglist.size(); delta++)
                {
                    Mat img = imglist[st + delta];
                    batch.push_back(preprocess(img));
                }
                Mat blob = dnn::blobFromImages(batch, 1.0, Size(resize_w, resize_h), Scalar(0.0, 0.0, 0.0), true, false, CV_32F);
                net->setInput(blob);
                Mat out = net->forward();
                for (int i = 0; i < (int)out.size().height; i++)
                {
                    std::vector<float> temp_feature;
                    for (int j = 0; j < (int)out.size().width; j++)
                    {
                        temp_feature.push_back(out.at<float>(i, j));
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

        static std::vector<cv::Mat> yoloDetector(cv::Mat &frame, cv::dnn::Net &net, std::vector<cv::String> &outputLayers)
        {
            int height = frame.rows;
            int width = frame.cols;

            // Create a blob from the frame
            cv::Mat blob;
            cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false, CV_32F);

            net.setInput(blob);
            std::vector<cv::Mat> outs;
            net.forward(outs, outputLayers);

            std::vector<int> class_ids;
            std::vector<float> confidences;
            std::vector<cv::Rect> boxes;

            for (size_t i = 0; i < outs.size(); ++i)
            {
                float *data = (float *)outs[i].data;
                for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
                {
                    cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                    cv::Point class_id_point;
                    double confidence;
                    minMaxLoc(scores, 0, &confidence, 0, &class_id_point);
                    int class_id = class_id_point.x;

                    if (confidence > 0.5 && class_id == 0)
                    { // Filter to detect only 'person'
                        int centerX = (int)(data[0] * width);
                        int centerY = (int)(data[1] * height);
                        int w = (int)(data[2] * width);
                        int h = (int)(data[3] * height);
                        int x = centerX - w / 2;
                        int y = centerY - h / 2;

                        boxes.push_back(cv::Rect(x, y, w, h));
                        confidences.push_back((float)confidence);
                        class_ids.push_back(class_id);
                    }
                }
            }

            // Apply Non-Maximum Suppression to reduce overlapping bounding boxes
            std::vector<int> indices;
            cv::dnn::NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

            std::vector<cv::Mat> images;
            for (int index : indices)
            {
                cv::Rect box = boxes[index];
                box.x = std::max(0, box.x);
                box.y = std::max(0, box.y);
                box.width = std::min(box.width, width - box.x);
                box.height = std::min(box.height, height - box.y);
                cv::Mat crop_img = frame(box);
                images.push_back(crop_img);
                imgDict[crop_img] = box;
            }

            return images;
        }

        static void extractFrames(const std::string &queryImgPath, const std::string &videoPath, Net *reidNet, const std::string &yoloPath, const std::string &cfgPath, int resize_h = 384, int resize_w = 128, int batch_size = 32)
        {
            cv::VideoCapture cap(videoPath);
            if (!cap.isOpened())
            {
                std::cerr << "Error: Video could not be opened." << std::endl;
                return;
            }

            cv::dnn::Net net = cv::dnn::readNet(yoloPath, cfgPath);
            std::vector<cv::String> layerNames = net.getLayerNames();

            std::vector<int> out_layers_indices = net.getUnconnectedOutLayers();
            std::vector<cv::String> outputLayers;
            for (int index : out_layers_indices)
            {
                outputLayers.push_back(layerNames[index - 1]);
            }

            std::vector<cv::Mat> frames;

            cv::Mat queryImg = cv::imread(queryImgPath);
            if (queryImg.empty())
            {
                std::cerr << "Error: Query image could not be loaded." << std::endl;
                return;
            }
            std::vector<cv::Mat> queryImages = {queryImg};

            cv::Mat frame;
            while (true)
            {
                if (!cap.read(frame) || frame.empty())
                {
                    break;
                }

                std::vector<cv::Mat> detectedImages = yoloDetector(frame, net, outputLayers);
                std::vector<std::vector<float>> queryFeatures;
                extractFeatures(queryImages, reidNet, batch_size, resize_h, resize_w, queryFeatures);
                std::vector<std::vector<float>> galleryFeatures;
                extractFeatures(detectedImages, reidNet, batch_size, resize_h, resize_w, galleryFeatures);

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

    };
};

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
    const std::string cfgPath = parser.get<String>("cfg");
    const int backend = parser.get<int>("backend");
    const int target = parser.get<int>("target");
    const int batch_size = parser.get<int>("batch_size");
    const int resize_h = parser.get<int>("resize_h");
    const int resize_w = parser.get<int>("resize_w");

    dnn::Net net = dnn::readNet(modelPath);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    reid::extractFrames(queryImagePath, videoPath, &net, yoloPath, cfgPath, resize_h, resize_w, batch_size);
    return 0;
}
