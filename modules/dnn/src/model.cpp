// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

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
    float  scale = 1.0;
    bool   swapRB = false;
    bool   crop = false;
    Mat    blob;
    std::vector<String> outNames;

    void predict(Net& net, const Mat& frame, std::vector<Mat>& outs) {
        if (size.width <= 0) size.width = frame.cols;
        if (size.height <= 0) size.height = frame.rows;

        blobFromImage(frame, blob, 1.0, size, Scalar(), swapRB, crop, CV_8U);
        net.setInput(blob, "", scale, mean);

        // Faster-RCNN or R-FCN
        if (net.getLayer(0)->outputNameToIndex("im_info") != -1) {
            Mat imInfo = (Mat_<float>(1, 3) << size.height, size.width, 1.6f);
            net.setInput(imInfo, "im_info");
        }
        net.forward(outs, outNames);
    }
};

Model::Model(const String& model, const String& config)
    : Net(readNet(model, config)), impl(new Impl) {
        impl->outNames = getUnconnectedOutLayersNames();
    };

Model::Model(const Net& network) : Net(network), impl(new Impl) {
    impl->outNames = getUnconnectedOutLayersNames();
};

Ptr<Model> Model::create(const String& model, const String& config) {
    return makePtr<Model>(model, config);
}

Model& Model::setInputSize(const Size& size) {
    impl->size = size;
    return *this;
}

Model& Model::setInputSize(int width, int height) {
    impl->size = Size(width, height);
    return *this;
}

Model& Model::setInputMean(const Scalar& mean) {
    impl->mean = mean;
    return *this;
}

Model& Model::setInputScale(float scale) {
    impl->scale = scale;
    return *this;
}

Model& Model::setInputCrop(bool crop) {
    impl->crop = crop;
    return *this;
}

Model& Model::setInputSwapRB(bool swapRB) {
    impl->swapRB = swapRB;
    return *this;
}

void Model::setParams(int width, int height, Scalar mean,
                      float scale, bool swapRB, bool crop) {
    impl->size.width = width;
    impl->size.height = height;
    impl->mean = mean;
    impl->scale = scale;
    impl->crop = crop;
    impl->swapRB = swapRB;
}

void Model::predict(InputArray frame, OutputArrayOfArrays outs) {
    std::vector<Mat> outputs;
    outs.getMatVector(outputs);
    impl->predict(*this, frame.getMat(), outputs);
}

ClassificationModel::ClassificationModel(const String& model, const String& config)
    : Model(model, config) {};

ClassificationModel::ClassificationModel(const Net& network) : Model(network) {};

Ptr<ClassificationModel> ClassificationModel::create(const String& model, const String& config) {
    return makePtr<ClassificationModel>(model, config);
}

std::pair<int, float> ClassificationModel::classify(InputArray frame) {
    std::vector<Mat> outs;
    impl->predict(*this, frame.getMat(), outs);
    CV_Assert(outs.size() == 1);

    double conf;
    cv::Point maxLoc;
    minMaxLoc(outs[0], nullptr, &conf, nullptr, &maxLoc);
    return {maxLoc.x, static_cast<float>(conf)};
}

void ClassificationModel::classify(InputArray frame, int& classId, float& conf) {
    std::tie(classId, conf) = classify(frame);
}

DetectionModel::DetectionModel(const String& model, const String& config)
    : Model(model, config) {};

DetectionModel::DetectionModel(const Net& network) : Model(network) {};

Ptr<DetectionModel> DetectionModel::create(const String& model, const String& config) {
    return makePtr<DetectionModel>(model, config);
}

void DetectionModel::detect(InputArray frame, CV_OUT std::vector<int>& classIds,
                   CV_OUT std::vector<float>& confidences, CV_OUT std::vector<Rect2d>& boxes,
                   float confThreshold, float nmsThreshold, bool absoluteCoords)
{
    std::vector<Mat> detections;
    impl->predict(*this, frame.getMat(), detections);

    boxes.clear();
    confidences.clear();
    classIds.clear();

    int frameWidth  = frame.cols();
    int frameHeight = frame.rows();
    std::vector<String> layerNames = getLayerNames();
    int lastLayerId = getLayerId(layerNames.back());
    Ptr<Layer> lastLayer = getLayer(lastLayerId);

    std::vector<int> predClassIds;
    std::vector<Rect2d> predBoxes;
    std::vector<float> predConf;

    if (lastLayer->type == "DetectionOutput") {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        for (int i = 0; i < detections.size(); ++i)
        {
            float* data = (float*)detections[i].data;
            for (int j = 0; j < detections[i].total(); j += 7)
            {
                float conf = data[j + 2];
                if (conf < confThreshold)
                    continue;

                float left   = data[j + 3];
                float top    = data[j + 4];
                float right  = data[j + 5];
                float bottom = data[j + 6];
                float width  = right  - left + 1;
                float height = bottom - top  + 1;

                if (absoluteCoords && width * height <= 1)
                    boxes.emplace_back(left * frameWidth, top * frameHeight,
                                       width * frameWidth, height * frameHeight);
                else
                    boxes.emplace_back(left, top, width, height);

                classIds.push_back(static_cast<int>(data[j + 1]));
                confidences.push_back(conf);
            }
        }
    } else if (lastLayer->type == "Region") {
        for (int i = 0; i < detections.size(); ++i) {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*)detections[i].data;
            for (int j = 0; j < detections[i].rows; ++j, data += detections[i].cols) {

                Mat scores = detections[i].row(j).colRange(5, detections[i].cols);
                Point classIdPoint;
                double conf;
                minMaxLoc(scores, nullptr, &conf, nullptr, &classIdPoint);

                if (conf < confThreshold)
                    continue;

                float centerX = data[0];
                float centerY = data[1];
                float width   = data[2];
                float height  = data[3];
                float left    = centerX - width  / 2;
                float top     = centerY - height / 2;

                predClassIds.push_back(classIdPoint.x);
                predConf.push_back(conf);

                if (absoluteCoords)
                    predBoxes.emplace_back(left * frameWidth, top * frameHeight,
                                           width * frameWidth, height * frameHeight);
                else
                    predBoxes.emplace_back(left, top, width, height);
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
