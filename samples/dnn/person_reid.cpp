//
// You can download a baseline ReID model and sample input from:
// https://github.com/ReID-Team/ReID_extra_testdata
//
// Authors of samples and Youtu ReID baseline:
//         Xing Sun <winfredsun@tencent.com>
//         Feng Zheng <zhengf@sustech.edu.cn>
//         Xinyang Jiang <sevjiang@tencent.com>
//         Fufu Yu <fufuyu@tencent.com>
//         Enwei Zhang <miyozhang@tencent.com>
//
// Copyright (C) 2020-2021, Tencent.
// Copyright (C) 2020-2021, SUSTech.
//
#include <iostream>
#include <fstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace cv::dnn;

const char* keys =
"{help    h  |                 | show help message}"
"{model   m  |                 | network model}"
"{query_list q |               | list of query images}"
"{gallery_list g |             | list of gallery images}"
"{batch_size | 32              | batch size of each inference}"
"{resize_h   | 256             | resize input to specific height.}"
"{resize_w   | 128             | resize input to specific width.}"
"{topk k     | 5               | number of gallery images showed in visualization}"
"{output_dir |                 | path for visualization(it should be existed)}"
"{backend b  | 0               | choose one of computation backends: "
"0: automatically (by default), "
"1: Halide language (http://halide-lang.org/), "
"2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
"3: OpenCV implementation ,"
"5: CUDA }"
"{target  t  | 0                | choose one of target computation devices: "
"0: CPU target (by default), "
"1: OpenCL, "
"2: OpenCL fp16 (half-float precision), "
"6: CUDA ,"
"7: CUDA fp16 (half-float preprocess) }";

namespace cv{
namespace reid{

static Mat preprocess(const Mat& img)
{
    const double mean[3] = {0.485, 0.456, 0.406};
    const double std[3] = {0.229, 0.224, 0.225};
    Mat ret = Mat(img.rows, img.cols, CV_32FC3);
    for (int y = 0; y < ret.rows; y ++)
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

static std::vector<float> normalization(const std::vector<float>& feature)
{
    std::vector<float> ret;
    float sum = 0.0;
    for(int i = 0; i < (int)feature.size(); i++)
    {
        sum += feature[i] * feature[i];
    }
    sum = sqrt(sum);
    for(int i = 0; i < (int)feature.size(); i++)
    {
        ret.push_back(feature[i] / sum);
    }
    return ret;
}

static void extractFeatures(const std::vector<std::string>& imglist, Net* net, const int& batch_size, const int& resize_h, const int& resize_w, std::vector<std::vector<float>>& features)
{
    for(int st = 0; st < (int)imglist.size(); st += batch_size)
    {
        std::vector<Mat> batch;
        for(int delta = 0; delta < batch_size && st + delta < (int)imglist.size(); delta++)
        {
            Mat img = imread(imglist[st + delta]);
            batch.push_back(preprocess(img));
        }
        Mat blob = dnn::blobFromImages(batch, 1.0, Size(resize_w, resize_h), Scalar(0.0,0.0,0.0), true, false, CV_32F);
        net->setInput(blob);
        Mat out = net->forward();
        for(int i = 0; i < (int)out.size().height; i++)
        {
            std::vector<float> temp_feature;
            for(int j = 0; j < (int)out.size().width; j++)
            {
                temp_feature.push_back(out.at<float>(i,j));
            }
            features.push_back(normalization(temp_feature));
        }
    }
    return ;
}

static void getNames(const std::string& ImageList, std::vector<std::string>& result)
{
    std::ifstream img_in(ImageList);
    std::string img_name;
    while(img_in >> img_name)
    {
        result.push_back(img_name);
    }
    return ;
}

static float similarity(const std::vector<float>& feature1, const std::vector<float>& feature2)
{
    float result = 0.0;
    for(int i = 0; i < (int)feature1.size(); i++)
    {
        result += feature1[i] * feature2[i];
    }
    return result;
}

static void getTopK(const std::vector<std::vector<float>>& queryFeatures, const std::vector<std::vector<float>>& galleryFeatures, const int& topk, std::vector<std::vector<int>>& result)
{
    for(int i = 0; i < (int)queryFeatures.size(); i++)
    {
        std::vector<float> similarityList;
        std::vector<int> index;
        for(int j = 0; j < (int)galleryFeatures.size(); j++)
        {
            similarityList.push_back(similarity(queryFeatures[i], galleryFeatures[j]));
            index.push_back(j);
        }
        sort(index.begin(), index.end(), [&](int x,int y){return similarityList[x] > similarityList[y];});
        std::vector<int> topk_result;
        for(int j = 0; j < min(topk, (int)index.size()); j++)
        {
            topk_result.push_back(index[j]);
        }
        result.push_back(topk_result);
    }
    return ;
}

static void addBorder(const Mat& img, const Scalar& color, Mat& result)
{
    const int bordersize = 5;
    copyMakeBorder(img, result, bordersize, bordersize, bordersize, bordersize, cv::BORDER_CONSTANT, color);
    return ;
}

static void drawRankList(const std::string& queryName, const std::vector<std::string>& galleryImageNames, const std::vector<int>& topk_index, const int& resize_h, const int& resize_w, Mat& result)
{
    const Size outputSize = Size(resize_w, resize_h);
    Mat q_img = imread(queryName), temp_img;
    resize(q_img, temp_img, outputSize);
    addBorder(temp_img, Scalar(0,0,0), q_img);
    putText(q_img, "Query", Point(10, 30), FONT_HERSHEY_COMPLEX, 1.0, Scalar(0,255,0), 2);
    std::vector<Mat> Images;
    Images.push_back(q_img);
    for(int i = 0; i < (int)topk_index.size(); i++)
    {
        Mat g_img = imread(galleryImageNames[topk_index[i]]);
        resize(g_img, temp_img, outputSize);
        addBorder(temp_img, Scalar(255,255,255), g_img);
        putText(g_img, "G" + std::to_string(i), Point(10, 30), FONT_HERSHEY_COMPLEX, 1.0, Scalar(0,255,0), 2);
        Images.push_back(g_img);
    }
    hconcat(Images, result);
    return ;
}

static void visualization(const std::vector<std::vector<int>>& topk, const std::vector<std::string>& queryImageNames, const std::vector<std::string>& galleryImageNames, const std::string& output_dir, const int& resize_h, const int& resize_w)
{
    for(int i = 0; i < (int)queryImageNames.size(); i++)
    {
        Mat img;
        drawRankList(queryImageNames[i], galleryImageNames, topk[i], resize_h, resize_w, img);
        std::string output_path = output_dir + "/" + queryImageNames[i].substr(queryImageNames[i].rfind("/")+1);
        imwrite(output_path, img);
    }
    return ;
}

};
};

int main(int argc, char** argv)
{
    // Parse command line arguments.
    CommandLineParser parser(argc, argv, keys);

    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    parser = CommandLineParser(argc, argv, keys);
    parser.about("Use this script to run ReID networks using OpenCV.");

    const std::string modelPath = parser.get<String>("model");
    const std::string queryImageList = parser.get<String>("query_list");
    const std::string galleryImageList = parser.get<String>("gallery_list");
    const int backend = parser.get<int>("backend");
    const int target = parser.get<int>("target");
    const int batch_size = parser.get<int>("batch_size");
    const int resize_h = parser.get<int>("resize_h");
    const int resize_w = parser.get<int>("resize_w");
    const int topk = parser.get<int>("topk");
    const std::string output_dir= parser.get<String>("output_dir");

    std::vector<std::string> queryImageNames;
    reid::getNames(queryImageList, queryImageNames);
    std::vector<std::string> galleryImageNames;
    reid::getNames(galleryImageList, galleryImageNames);

    dnn::Net net = dnn::readNet(modelPath);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    std::vector<std::vector<float>> queryFeatures;
    reid::extractFeatures(queryImageNames, &net, batch_size, resize_h, resize_w, queryFeatures);
    std::vector<std::vector<float>> galleryFeatures;
    reid::extractFeatures(galleryImageNames, &net, batch_size, resize_h, resize_w, galleryFeatures);

    std::vector<std::vector<int>> topkResult;
    reid::getTopK(queryFeatures, galleryFeatures, topk, topkResult);
    reid::visualization(topkResult, queryImageNames, galleryImageNames, output_dir, resize_h, resize_w);

    return 0;
}
