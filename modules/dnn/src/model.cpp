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

struct Model::Impl {
    Size   size;
    Scalar mean;
    float  scale;
    bool   swapRB;
    bool   crop;
};

Model::Model(const std::string& model, const std::string& config, const Size& size,
             const Scalar& mean, float scale, bool swapRB, bool crop)
    : Net(readNet(model, config)), impl_(new Impl{size, mean, scale, swapRB, crop}) {};

Model::Model(const Net& network, const Size& size, const Scalar& mean, float scale, bool swapRB, bool crop)
    : Net(network), impl_(new Impl{size, mean, scale, swapRB, crop}) {};

Model& Model::setInputSize(const Size& size) {
    impl_->size = size;
    return *this;
}

Model& Model::setInputMean(const Scalar& mean) {
    impl_->mean = mean;
    return *this;
}

Model& Model::setInputScale(float scale) {
    impl_->scale = scale;
    return *this;
}

Model& Model::setInputCrop(bool crop) {
    impl_->crop = crop;
    return *this;
}

Model& Model::setInputSwapRB(bool swapRB) {
    impl_->swapRB = swapRB;
    return *this;
}

std::pair<int, float> Model::classify(InputArray frame) {
    Mat blob;
    blobFromImage(frame, blob, 1.0, impl_->size, Scalar(), impl_->swapRB, impl_->crop, CV_8U);
    setInput(blob, "", impl_->scale, impl_->mean);
    Mat out = forward();

    double conf;
    cv::Point maxLoc;
    minMaxLoc(out, nullptr, &conf, nullptr, &maxLoc);

    return {maxLoc.x, static_cast<float>(conf)};
}

void Model::detect(InputArray frame, CV_OUT std::vector<int>& classIds,
                   CV_OUT std::vector<float>& confidences, CV_OUT std::vector<Rect2d>& boxes,
                   float confThreshold, float nmsThreshold,  bool absoluteCoords)
{
    int frameWidth  = frame.cols();
    int frameHeight = frame.rows();

    if (impl_->size.width <= 0) impl_->size.width = frame.cols();
    if (impl_->size.height <= 0) impl_->size.height = frame.rows();

    Mat blob;
    blobFromImage(frame, blob, 1.0, impl_->size, Scalar(), impl_->swapRB, impl_->crop, CV_8U);
    setInput(blob, "", impl_->scale, impl_->mean);

    if (getLayer(0)->outputNameToIndex("im_info") != -1) {  // Faster-RCNN or R-FCN
        frameWidth  = impl_->size.width;
        frameHeight = impl_->size.height;
        Mat imInfo = (Mat_<float>(1, 3) << impl_->size.height, impl_->size.width, 1.6f);
        setInput(imInfo, "im_info");
    }

    std::vector<Mat> detections;
    std::vector<String> outNames = getUnconnectedOutLayersNames();
    forward(detections, outNames);
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
                    float left   = detection[3];
                    float top    = detection[4];
                    float right  = detection[5];
                    float bottom = detection[6];
                    float width  = right  - left + 1;
                    float height = bottom - top  + 1;

                    if (absoluteCoords && width * height <= 1)
                        boxes.emplace_back(left * frameWidth, top * frameHeight,
                                           width * frameWidth, height * frameHeight);
                    else
                        boxes.emplace_back(left, top, width, height);

                    classIds.push_back(static_cast<int>(detection[1]));
                    confidences.push_back(conf);
                }
            }
        }
    } else if (lastLayer->type == "Region") {
        for (int i = 0; i < detections.size(); ++i) {
            for (int j = 0; j < detections[i].rows; ++j) {

                Mat slice = detections[i].row(j).colRange(5, detections[i].cols);

                double max;
                cv::Point maxLoc;
                minMaxLoc(slice, nullptr, &max, nullptr, &maxLoc);

                float conf  = static_cast<float>(max);
                int classId = maxLoc.x;

                if (conf > confThreshold)
                {
                    std::vector<Range> coord = {Range(j, j + 1), Range(0, 5)};
                    Mat coords;
                    detections[i](coord).copyTo(coords);
                    std::vector<float> bboxes = coords.reshape(1, 1);

                    float centerX = bboxes[0];
                    float centerY = bboxes[1];
                    float width   = bboxes[2];
                    float height  = bboxes[3];
                    float left    = centerX - width  / 2;
                    float top     = centerY - height / 2;

                    predClassIds.push_back(classId);
                    predConf.push_back(conf);

                    if (absoluteCoords)
                        predBoxes.emplace_back(left * frameWidth, top * frameHeight,
                                               width * frameWidth, height * frameHeight);
                    else
                        predBoxes.emplace_back(left, top, width, height);
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
