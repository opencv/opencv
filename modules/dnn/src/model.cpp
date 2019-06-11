// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2018-2019, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.


#include "precomp.hpp"
#include <algorithm>
#include <iostream>
#include <utility>
#include <iterator>

#include <opencv2/imgproc.hpp>

namespace cv {
namespace dnn {

Model::Model(const std::string& model, const std::string& config, int width,
      int height, Scalar mean, float scale, bool swapRB, bool crop) : Net(readNet(model, config)),
      width_(width), height_(height), mean_(mean), scale_(scale), swapRB_(swapRB), crop_(crop) {}

void Model::setInputWidth(int width) {
    width_ = width;
}

void Model::setInputHeight(int height) {
    height_ = height;
}

void Model::setInputMean(Scalar mean) {
    mean_ = mean;
}

void Model::setInputScale(float scale) {
    scale_ = scale;
}

void Model::setInputCrop(bool crop) {
    crop_ = crop;
}

void Model::setInputSwapRB(bool swapRB) {
    swapRB_ = swapRB;
}

std::pair<int, float> Model::classify(const Mat& frame) {
    Mat blob = blobFromImage(frame, scale_, Size(width_, height_), mean_, swapRB_, crop_);
    setInput(blob);
    Mat out = forward();
    std::vector<float> result = out.reshape(1, 1);
    std::vector<float>::iterator iter = std::max_element(result.begin(), result.end());

    std::pair<int, float> prediction(iter - result.begin(), *iter);
    return prediction;
}

void Model::detect(Mat& frame, std::vector<int>& classIds,
                   std::vector<float>& confidences, std::vector<Rect2d>& boxes,
                   float confThreshold, float nmsThreshold)
{
    Mat blob = blobFromImage(frame, scale_, Size(width_, height_), mean_, swapRB_, crop_);
    setInput(blob);
    if (getLayer(0)->outputNameToIndex("im_info") != -1) {  // Faster-RCNN or R-FCN
        resize(frame, frame, Size(width_, height_));
        Mat imInfo = (Mat_<float>(1, 3) << frame.rows, frame.cols, 1.6f);
        setInput(imInfo, "im_info");
    }

    std::vector<Mat> detections;
    forward(detections);
    CV_Assert(detections.size() > 0);

    std::vector<String> layerNames = getLayerNames();
    int lastLayerId = getLayerId(layerNames.back());
    Ptr<Layer> lastLayer = getLayer(lastLayerId);

    std::vector<int> predClassIds;
    std::vector<Rect2d> predBoxes;
    std::vector<float> predConf;
    if (lastLayer->type == "DetectionOutput") {
        for (int i = 0; i < detections.size(); ++i)
        {
            for (int j = 0; j < detections[i].size[2]; j++)
            {
                std::vector<Range> coord = { Range::all(), Range::all(), Range(j, j + 1), Range::all() };
                Mat detect;
                detections[i](coord).copyTo(detect);
                std::vector<float> detection = detect.reshape(1, 1);
                float conf = detection[2];
                if (conf > confThreshold)
                {
                    float left = detection[3];
                    float top = detection[4];
                    float right = detection[5];
                    float bottom = detection[6];
                    float width = right - left + 1;
                    float height = bottom - top + 1;

                    classIds.push_back(static_cast<int>(detection[1]));
                    boxes.push_back(Rect2d(left, top, width, height));
                    confidences.push_back(conf);
                }
            }
        }
    } else if (lastLayer->type == "Region") {
        for (int i = 0; i < detections.size(); ++i) {
            for (int j = 0; j < detections[i].rows; ++j) {

                Mat slice = detections[i].row(j).colRange(5, detections[i].cols);
                std::vector<float> scores = slice.reshape(1, 1);
                std::vector<float>::iterator classId = std::max_element(scores.begin(), scores.end());
                float conf = *classId;

                if (conf > confThreshold)
                {
                    std::vector<Range> coord = {Range(j, j + 1), Range(0, 5)};
                    Mat coords;
                    detections[i](coord).copyTo(coords);
                    std::vector<float> bboxes = coords.reshape(1, 1);

                    float centerX = bboxes[0];
                    float centerY = bboxes[1];
                    float width = bboxes[2];
                    float height = bboxes[3];
                    float left = centerX - width / 2;
                    float top = centerY - height / 2;

                    predClassIds.push_back(classId - scores.begin());
                    predConf.push_back(*classId);
                    predBoxes.push_back(Rect2d(left, top, width, height));

                }
            }
        }
    } else {
        CV_Error(Error::StsError, "Unknown output layer type: \"" + lastLayer->type + "\"");
    }

    std::vector<int> indices;
    NMSBoxes(predBoxes, predConf, confThreshold, nmsThreshold, indices);

    boxes.reserve(indices.size());
    confidences.reserve(indices.size());
    classIds.reserve(indices.size());

    for (int idx : indices)
    {
        boxes.push_back(predBoxes[idx]);
        confidences.push_back(predConf[idx]);
        classIds.push_back(predClassIds[idx]);
    }
}


void Model::detect(Mat& frame, std::vector<int>& classIds,
                   std::vector<float>& confidences, std::vector<Rect2i>& boxes,
                   float confThreshold, float nmsThreshold)
{
    Mat blob = blobFromImage(frame, scale_, Size(width_, height_), mean_, swapRB_, crop_);
    setInput(blob);
    if (getLayer(0)->outputNameToIndex("im_info") != -1) {  // Faster-RCNN or R-FCN
        resize(frame, frame, Size(width_, height_));
        Mat imInfo = (Mat_<float>(1, 3) << frame.rows, frame.cols, 1.6f);
        setInput(imInfo, "im_info");
    }

    std::vector<Mat> detections;
    forward(detections);
    CV_Assert(detections.size() > 0);

    std::vector<String> layerNames = getLayerNames();
    int lastLayerId = getLayerId(layerNames.back());
    Ptr<Layer> lastLayer = getLayer(lastLayerId);

    int frameWidth = frame.cols;
    int frameHeight = frame.rows;
    std::vector<int> predClassIds;
    std::vector<Rect2i> predBoxes;
    std::vector<float> predConf;
    if (lastLayer->type == "DetectionOutput") {
        CV_Assert(detections.size() > 0);
        for (int i = 0; i < detections.size(); ++i)
        {
            for (int j = 0; j < detections[i].size[2]; j++)
            {
                std::vector<Range> coord = { Range::all(), Range::all(), Range(j, j + 1), Range::all() };
                Mat detect;
                detections[i](coord).copyTo(detect);
                std::vector<float> detection = detect.reshape(1, 1);
                float conf = detection[2];
                if (conf > confThreshold)
                {
                    int left = static_cast<int>(detection[3]);
                    int top = static_cast<int>(detection[4]);
                    int right = static_cast<int>(detection[5]);
                    int bottom = static_cast<int>(detection[6]);
                    int width = right - left + 1;
                    int height = bottom - top + 1;

                    if (width * height <= 1) {
                        left = static_cast<int>(detection[3] * frameWidth);
                        top = static_cast<int>(detection[4] * frameHeight);
                        right = static_cast<int>(detection[5] * frameWidth);
                        bottom = static_cast<int>(detection[6] * frameHeight);
                        width = right - left + 1;
                        height = bottom - top + 1;
                    }
                    classIds.push_back(static_cast<int>(detection[1]) );
                    boxes.push_back(Rect2i(left, top, width, height));
                    confidences.push_back(conf);
                }
            }
        }
    } else if (lastLayer->type == "Region") {
        for (int i = 0; i < detections.size(); ++i) {
        for (int j = 0; j < detections[i].rows; ++j) {

            Mat slice = detections[i].row(j).colRange(5, detections[i].cols);
            std::vector<float> scores = slice.reshape(1, 1);
            std::vector<float>::iterator classId = std::max_element(scores.begin(), scores.end());
            float conf = *classId;

            if (conf > confThreshold)
            {
                std::vector<Range> coord = {Range(j, j + 1), Range(0, 5)};
                Mat coords;
                detections[i](coord).copyTo(coords);
                std::vector<float> bboxes = coords.reshape(1, 1);

                int centerX = static_cast<int>(bboxes[0] * frameWidth);
                int centerY = static_cast<int>(bboxes[1] * frameHeight);
                int width = static_cast<int>(bboxes[2] * frameWidth);
                int height = static_cast<int>(bboxes[3] * frameHeight);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                predClassIds.push_back(classId - scores.begin());
                predConf.push_back(*classId);
                predBoxes.push_back(Rect2i(left, top, width, height));

            }
        }
    }
    } else {
        CV_Error(Error::StsError, "Unknown output layer type: \"" + lastLayer->type + "\"");
    }

    std::vector<int> indices;
    NMSBoxes(predBoxes, predConf, confThreshold, nmsThreshold, indices);

    boxes.reserve(indices.size());
    confidences.reserve(indices.size());
    classIds.reserve(indices.size());

    for (int idx : indices)
    {
        boxes.push_back(predBoxes[idx]);
        confidences.push_back(predConf[idx]);
        classIds.push_back(predClassIds[idx]);
    }
}

}} // namespace
