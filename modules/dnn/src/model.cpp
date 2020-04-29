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

struct Model::Impl
{
    Size   size;
    Scalar mean;
    double  scale = 1.0;
    bool   swapRB = false;
    bool   crop = false;
    Mat    blob;
    std::vector<String> outNames;

    void predict(Net& net, const Mat& frame, OutputArrayOfArrays outs)
    {
        if (size.empty())
            CV_Error(Error::StsBadSize, "Input size not specified");

        blob = blobFromImage(frame, scale, size, mean, swapRB, crop);
        net.setInput(blob);

        // Faster-RCNN or R-FCN
        if (net.getLayer(0)->outputNameToIndex("im_info") != -1)
        {
            Mat imInfo = (Mat_<float>(1, 3) << size.height, size.width, 1.6f);
            net.setInput(imInfo, "im_info");
        }
        net.forward(outs, outNames);
    }
};

Model::Model() : impl(new Impl) {}

Model::Model(const String& model, const String& config)
    : Net(readNet(model, config)), impl(new Impl)
{
    impl->outNames = getUnconnectedOutLayersNames();
    std::vector<MatShape> inLayerShapes;
    std::vector<MatShape> outLayerShapes;
    getLayerShapes(MatShape(), 0, inLayerShapes, outLayerShapes);
    if (!inLayerShapes.empty() && inLayerShapes[0].size() == 4)
        impl->size = Size(inLayerShapes[0][3], inLayerShapes[0][2]);
};

Model::Model(const Net& network) : Net(network), impl(new Impl)
{
    impl->outNames = getUnconnectedOutLayersNames();
    std::vector<MatShape> inLayerShapes;
    std::vector<MatShape> outLayerShapes;
    getLayerShapes(MatShape(), 0, inLayerShapes, outLayerShapes);
    if (!inLayerShapes.empty() && inLayerShapes[0].size() == 4)
        impl->size = Size(inLayerShapes[0][3], inLayerShapes[0][2]);
};

Model& Model::setInputSize(const Size& size)
{
    impl->size = size;
    return *this;
}

Model& Model::setInputSize(int width, int height)
{
    impl->size = Size(width, height);
    return *this;
}

Model& Model::setInputMean(const Scalar& mean)
{
    impl->mean = mean;
    return *this;
}

Model& Model::setInputScale(double scale)
{
    impl->scale = scale;
    return *this;
}

Model& Model::setInputCrop(bool crop)
{
    impl->crop = crop;
    return *this;
}

Model& Model::setInputSwapRB(bool swapRB)
{
    impl->swapRB = swapRB;
    return *this;
}

void Model::setInputParams(double scale, const Size& size, const Scalar& mean,
                           bool swapRB, bool crop)
{
    impl->size = size;
    impl->mean = mean;
    impl->scale = scale;
    impl->crop = crop;
    impl->swapRB = swapRB;
}

void Model::predict(InputArray frame, OutputArrayOfArrays outs)
{
    impl->predict(*this, frame.getMat(), outs);
}

ClassificationModel::ClassificationModel(const String& model, const String& config)
    : Model(model, config) {};

ClassificationModel::ClassificationModel(const Net& network) : Model(network) {};

std::pair<int, float> ClassificationModel::classify(InputArray frame)
{
    std::vector<Mat> outs;
    impl->predict(*this, frame.getMat(), outs);
    CV_Assert(outs.size() == 1);

    double conf;
    cv::Point maxLoc;
    minMaxLoc(outs[0].reshape(1, 1), nullptr, &conf, nullptr, &maxLoc);
    return {maxLoc.x, static_cast<float>(conf)};
}

void ClassificationModel::classify(InputArray frame, int& classId, float& conf)
{
    std::tie(classId, conf) = classify(frame);
}

KeypointsModel::KeypointsModel(const String& model, const String& config)
    : Model(model, config) {};

KeypointsModel::KeypointsModel(const Net& network) : Model(network) {};

std::vector<Point2f> KeypointsModel::estimate(InputArray frame, float thresh)
{

    int frameHeight = frame.getMat().size[0];
    int frameWidth = frame.getMat().size[1];
    std::vector<Mat> outs;

    impl->predict(*this, frame.getMat(), outs);
    CV_Assert(outs.size() == 1);
    Mat output = outs[0];

    const int nPoints = output.size[1];
    std::vector<Point2f> points;

    // If output is a map, extract the keypoints
    if (output.dims == 4)
    {
        int height = output.size[2];
        int width = output.size[3];

        // find the position of the keypoints (ignore the background)
        for (int n=0; n < nPoints - 1; n++)
        {
            // Probability map of corresponding keypoint
            Mat probMap(height, width, CV_32F, output.ptr(0, n));

            Point2f p(-1, -1);
            Point maxLoc;
            double prob;
            minMaxLoc(probMap, NULL, &prob, NULL, &maxLoc);
            if (prob > thresh)
            {
                p = maxLoc;
                p.x *= (float)frameWidth / width;
                p.y *= (float)frameHeight / height;
            }
            points.push_back(p);
        }
    }
    // Otherwise the output is a vector of keypoints and we can just return it
    else
    {
        for (int n=0; n < nPoints; n++)
        {
            Point2f p;
            p.x = *output.ptr<float>(0, n, 0);
            p.y = *output.ptr<float>(0, n, 1);
            points.push_back(p);
        }
    }
    return points;
}

SegmentationModel::SegmentationModel(const String& model, const String& config)
    : Model(model, config) {};

SegmentationModel::SegmentationModel(const Net& network) : Model(network) {};

void SegmentationModel::segment(InputArray frame, OutputArray mask)
{

    std::vector<Mat> outs;
    impl->predict(*this, frame.getMat(), outs);
    CV_Assert(outs.size() == 1);
    Mat score = outs[0];

    const int chns = score.size[1];
    const int rows = score.size[2];
    const int cols = score.size[3];

    mask.create(rows, cols, CV_8U);
    Mat classIds = mask.getMat();
    classIds.setTo(0);
    Mat maxVal(rows, cols, CV_32F, score.data);

    for (int ch = 1; ch < chns; ch++)
    {
        for (int row = 0; row < rows; row++)
        {
            const float *ptrScore = score.ptr<float>(0, ch, row);
            uint8_t *ptrMaxCl = classIds.ptr<uint8_t>(row);
            float *ptrMaxVal = maxVal.ptr<float>(row);
            for (int col = 0; col < cols; col++)
            {
                if (ptrScore[col] > ptrMaxVal[col])
                {
                    ptrMaxVal[col] = ptrScore[col];
                    ptrMaxCl[col] = ch;
                }
            }
        }
    }
}

DetectionModel::DetectionModel(const String& model, const String& config)
    : Model(model, config) {};

DetectionModel::DetectionModel(const Net& network) : Model(network) {};

void DetectionModel::detect(InputArray frame, CV_OUT std::vector<int>& classIds,
                            CV_OUT std::vector<float>& confidences, CV_OUT std::vector<Rect>& boxes,
                            float confThreshold, float nmsThreshold)
{
    std::vector<Mat> detections;
    impl->predict(*this, frame.getMat(), detections);

    boxes.clear();
    confidences.clear();
    classIds.clear();

    int frameWidth  = frame.cols();
    int frameHeight = frame.rows();
    if (getLayer(0)->outputNameToIndex("im_info") != -1)
    {
        frameWidth = impl->size.width;
        frameHeight = impl->size.height;
    }

    std::vector<String> layerNames = getLayerNames();
    int lastLayerId = getLayerId(layerNames.back());
    Ptr<Layer> lastLayer = getLayer(lastLayerId);

    std::vector<int> predClassIds;
    std::vector<Rect> predBoxes;
    std::vector<float> predConf;
    if (lastLayer->type == "DetectionOutput")
    {
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

                int left   = data[j + 3];
                int top    = data[j + 4];
                int right  = data[j + 5];
                int bottom = data[j + 6];
                int width  = right  - left + 1;
                int height = bottom - top + 1;

                if (width <= 2 || height <= 2)
                {
                    left   = data[j + 3] * frameWidth;
                    top    = data[j + 4] * frameHeight;
                    right  = data[j + 5] * frameWidth;
                    bottom = data[j + 6] * frameHeight;
                    width  = right  - left + 1;
                    height = bottom - top + 1;
                }

                left   = std::max(0, std::min(left, frameWidth - 1));
                top    = std::max(0, std::min(top, frameHeight - 1));
                width  = std::max(1, std::min(width, frameWidth - left));
                height = std::max(1, std::min(height, frameHeight - top));
                predBoxes.emplace_back(left, top, width, height);

                predClassIds.push_back(static_cast<int>(data[j + 1]));
                predConf.push_back(conf);
            }
        }
    }
    else if (lastLayer->type == "Region")
    {
        for (int i = 0; i < detections.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*)detections[i].data;
            for (int j = 0; j < detections[i].rows; ++j, data += detections[i].cols)
            {

                Mat scores = detections[i].row(j).colRange(5, detections[i].cols);
                Point classIdPoint;
                double conf;
                minMaxLoc(scores, nullptr, &conf, nullptr, &classIdPoint);

                if (static_cast<float>(conf) < confThreshold)
                    continue;

                int centerX = data[0] * frameWidth;
                int centerY = data[1] * frameHeight;
                int width   = data[2] * frameWidth;
                int height  = data[3] * frameHeight;

                int left = std::max(0, std::min(centerX - width / 2, frameWidth - 1));
                int top  = std::max(0, std::min(centerY - height / 2, frameHeight - 1));
                width    = std::max(1, std::min(width, frameWidth - left));
                height   = std::max(1, std::min(height, frameHeight - top));

                predClassIds.push_back(classIdPoint.x);
                predConf.push_back(static_cast<float>(conf));
                predBoxes.emplace_back(left, top, width, height);
            }
        }
    }
    else
        CV_Error(Error::StsNotImplemented, "Unknown output layer type: \"" + lastLayer->type + "\"");

    if (nmsThreshold)
    {
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
    else
    {
        boxes       = std::move(predBoxes);
        classIds    = std::move(predClassIds);
        confidences = std::move(predConf);
    }



}

}} // namespace
