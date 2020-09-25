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

void disableRegionNMS(Net& net)
{
    for (String& name : net.getUnconnectedOutLayersNames())
    {
        int layerId = net.getLayerId(name);
        Ptr<RegionLayer> layer = net.getLayer(layerId).dynamicCast<RegionLayer>();
        if (!layer.empty())
        {
            layer->nmsThreshold = 0;
        }
    }
}

DetectionModel::DetectionModel(const String& model, const String& config)
    : Model(model, config) {
      disableRegionNMS(*this);
}

DetectionModel::DetectionModel(const Net& network) : Model(network) {
    disableRegionNMS(*this);
}

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
                boxes.emplace_back(left, top, width, height);

                classIds.push_back(static_cast<int>(data[j + 1]));
                confidences.push_back(conf);
            }
        }
    }
    else if (lastLayer->type == "Region")
    {
        std::vector<int> predClassIds;
        std::vector<Rect> predBoxes;
        std::vector<float> predConf;
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

        if (nmsThreshold)
        {
            std::map<int, std::vector<size_t> > class2indices;
            for (size_t i = 0; i < predClassIds.size(); i++)
            {
                if (predConf[i] >= confThreshold)
                {
                    class2indices[predClassIds[i]].push_back(i);
                }
            }
            for (const auto& it : class2indices)
            {
                std::vector<Rect> localBoxes;
                std::vector<float> localConfidences;
                for (size_t idx : it.second)
                {
                    localBoxes.push_back(predBoxes[idx]);
                    localConfidences.push_back(predConf[idx]);
                }
                std::vector<int> indices;
                NMSBoxes(localBoxes, localConfidences, confThreshold, nmsThreshold, indices);
                classIds.resize(classIds.size() + indices.size(), it.first);
                for (int idx : indices)
                {
                    boxes.push_back(localBoxes[idx]);
                    confidences.push_back(localConfidences[idx]);
                }
            }
        }
        else
        {
            boxes       = std::move(predBoxes);
            classIds    = std::move(predClassIds);
            confidences = std::move(predConf);
        }
    }
    else
        CV_Error(Error::StsNotImplemented, "Unknown output layer type: \"" + lastLayer->type + "\"");
}

struct TextRecognitionModel::Voc
{
    std::vector<String> vocabulary;

    void setVocabulary(const std::vector<String>& inputVoc)
    {
        vocabulary.assign(inputVoc.begin(), inputVoc.end());
    }

    String decode(const Mat& prediction, const String& decodeType)
    {
        String decodeSeq = "";
        if (decodeType == "CTC-greedy") {
            bool ctcFlag = true;
            int lastLoc = 0;
            int vocLength = (int)(vocabulary.size());
            for (int i = 0; i < prediction.size[0]; i++) {
                const float* pred = prediction.ptr<float>(i);
                int maxLoc = 0;
                float maxScore = pred[0];
                for (int j = 0; j < vocLength + 1; j++) {
                    float score = pred[j];
                    if (maxScore < score) {
                        maxScore = score;
                        maxLoc = j;
                    }
                }

                if (maxLoc > 0) {
                    String currentChar = vocabulary.at(maxLoc - 1);
                    if (maxLoc != lastLoc || ctcFlag) {
                        lastLoc = maxLoc;
                        decodeSeq += currentChar;
                        ctcFlag = false;
                    }
                } else {
                    ctcFlag = true;
                }
            }
        } else {
            CV_Error(Error::StsBadArg, "Unsupported decodeType");
        }

        return decodeSeq;
    }
};

TextRecognitionModel::TextRecognitionModel(const String& model, const String& config)
    : Model(model, config), voc(new Voc) {}

TextRecognitionModel::TextRecognitionModel(const Net& network) : Model(network), voc(new Voc) {}

void TextRecognitionModel::setVocabulary(const std::vector<String>& inputVoc)
{
    voc->setVocabulary(inputVoc);
}

void TextRecognitionModel::recognize(InputArray frame, const String& decodeType, std::vector<String>& results,
                                     const std::vector<std::vector<Point>>& roiPolygons)
{
    results.clear();

    std::vector<Mat> outs;
    uint roiSize = roiPolygons.size();
    if (roiSize == 0) {
        impl->predict(*this, frame.getMat(), outs);
        CV_Assert(outs.size() == 1);
        results.push_back(voc->decode(outs[0], decodeType));
    } else {
        Mat input = frame.getMat();

        // Predict for each RoI
        for (uint i = 0; i < roiSize; i++) {
            int xmin = input.cols, xmax = 0, ymin = input.rows, ymax = 0;
            for (uint j = 0; j < roiPolygons[i].size(); j++) {
                xmin = std::min(roiPolygons[i][j].x, xmin);
                xmax = std::max(roiPolygons[i][j].x, xmax);
                ymin = std::min(roiPolygons[i][j].y, ymin);
                ymax = std::max(roiPolygons[i][j].y, ymax);
            }
            xmin = std::max(xmin, 0);
            ymin = std::max(ymin, 0);
            xmax = std::min(input.cols - 1, xmax);
            ymax = std::min(input.rows - 1, ymax);
            Rect roiRect = Rect(xmin, ymin, xmax - xmin, ymax - ymin);
            Mat roi = input(roiRect);
            impl->predict(*this, roi, outs);
            CV_Assert(outs.size() == 1);
            results.push_back(voc->decode(outs[0], decodeType));
        }
    }
}

static double contourScore(const Mat& binary, const std::vector<Point>& contour)
{
    int rows = binary.rows;
    int cols = binary.cols;

    int xmin = cols - 1;
    int xmax = 0;
    int ymin = rows - 1;
    int ymax = 0;
    for (size_t i = 0; i < contour.size(); i++) {
        Point pt = contour[i];
        xmin = std::min(pt.x, xmin);
        xmax = std::max(pt.x, xmax);
        ymin = std::min(pt.y, ymin);
        ymax = std::max(pt.y, ymax);
    }

    xmin = std::max(xmin, 0);
    xmax = std::min(cols - 1, xmax);
    ymin = std::max(ymin, 0);
    ymax = std::min(rows - 1, ymax);

    Mat binROI = binary(Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));

    Mat mask = Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8U);
    std::vector<Point> roiContour;
    for (size_t i = 0; i < contour.size(); i++) {
        Point pt = Point(contour[i].x - xmin, contour[i].y - ymin);
        roiContour.push_back(pt);
    }
    std::vector<std::vector<Point>> roiContours = {roiContour};
    fillPoly(mask, roiContours, Scalar(1));
    double score = mean(binROI, mask).val[0];

    return score;
}

static void unclip(const std::vector<Point>& inPoly, std::vector<Point> &outPoly, const double& unclipRatio,
                   const float& scaleWidth, const float& scaleHeight)
{
    double area = contourArea(inPoly);
    double length = arcLength(inPoly, true);
    double distance = area * unclipRatio / length;

    size_t numPoints = inPoly.size();
    std::vector<std::vector<Point2f>> newLines;
    for (size_t i = 0; i < numPoints; i++) {
        std::vector<Point2f> newLine;
        Point pt1 = inPoly[i];
        Point pt2 = inPoly[(i + 1) % numPoints];
        Point vec = pt2 - pt1;
        float unclipDis = (float)(distance / norm(vec));
        Point2f rotateVec = Point2f(-vec.y * unclipDis, vec.x * unclipDis);
        newLine.push_back(Point2f(pt1.x + rotateVec.x, pt1.y + rotateVec.y));
        newLine.push_back(Point2f(pt2.x + rotateVec.x, pt2.y + rotateVec.y));
        newLines.push_back(newLine);
    }

    size_t numLines = newLines.size();
    for (size_t i = 0; i < numLines; i++) {
        Point2f a = newLines[i][0];
        Point2f b = newLines[i][1];
        Point2f c = newLines[(i + 1) % numLines][0];
        Point2f d = newLines[(i + 1) % numLines][1];
        Point pt;
        Point2f v1 = b - a;
        Point2f v2 = d - c;
        double cosAngle = (v1.x * v2.x + v1.y * v2.y) / (norm(v1) * norm(v2));

        if( fabs(cosAngle) > 0.7 ) {
            pt.x = (int)((b.x + c.x) / 2);
            pt.y = (int)((b.y + c.y) / 2);
        } else {
            double denom = a.x * (double)(d.y - c.y) + b.x * (double)(c.y - d.y) +
                           d.x * (double)(b.y - a.y) + c.x * (double)(a.y - b.y);
            double num = a.x * (double)(d.y - c.y) + c.x * (double)(a.y - d.y) + d.x * (double)(c.y - a.y);
            double s = num / denom;

            pt.x = (int)((a.x + s*(b.x - a.x)) * scaleWidth);
            pt.y = (int)((a.y + s*(b.y - a.y)) * scaleHeight);
        }

        outPoly.push_back(pt);
    }
}

TextDetectionModel::TextDetectionModel(const String& model, const String& config)
    : Model(model, config) {}

TextDetectionModel::TextDetectionModel(const Net& network) : Model(network) {}

void TextDetectionModel::detect(InputArray frame, std::vector<std::vector<Point>>& results, const int& outputType,
            const float& binThresh, const float& polyThresh, const double& unclipRatio, const uint& maxCandidates)
{
    results.clear();

    std::vector<Mat> outs;
    impl->predict(*this, frame.getMat(), outs);
    CV_Assert(outs.size() == 1);
    Mat binary = outs[0];

    // Threshold
    Mat bitmap;
    threshold(binary, bitmap, binThresh, 255, THRESH_BINARY);

    // Scale ratio
    float scaleHeight = (float)(frame.rows()) / (float)(binary.size[0]);
    float scaleWidth = (float)(frame.cols()) / (float)(binary.size[1]);

    // Find contours
    std::vector<std::vector<Point>> contours;
    bitmap.convertTo(bitmap, CV_8UC1);
    findContours(bitmap, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    // Candidate number limitation
    size_t numCandidate = 0;
    if (contours.size() < maxCandidates) {
        numCandidate = contours.size();
    } else {
        numCandidate = maxCandidates;
    }

    for (size_t i = 0; i < numCandidate; i++) {
        std::vector<Point> contour = contours[i];

        // Calculate text contour score
        if (contourScore(binary, contour) < polyThresh) continue;

        // Unclip and Rescale
        std::vector<Point> approx;
        if (outputType == 0) {
            RotatedRect box = minAreaRect(contour);
            Point2f vertex[4];
            box.points(vertex);
            for (int j = 0; j < 4; j++) {
                approx.push_back(Point((int)(vertex[3-j].x), (int)(vertex[3-j].y)));
            }
        } else {
            double epsilon = arcLength(contour, true) * 0.01;
            approxPolyDP(contour, approx, epsilon, true);
            if (approx.size() < 4) continue;
        }
        std::vector<Point> polygon;
        unclip(approx, polygon, unclipRatio, scaleWidth, scaleHeight);
        results.push_back(polygon);
    }
}

void TextDetectionModel::detect(InputArray frame, std::vector<std::vector<Point>>& results,
                                const float& confThreshold, const float& nmsThreshold)
{
    results.clear();

    std::vector<Mat> outs;
    impl->predict(*this, frame.getMat(), outs);
    CV_Assert(outs.size() == 2);
    Mat geometry = outs[0];
    Mat scoreMap = outs[1];

    CV_Assert(scoreMap.dims == 4);
    CV_Assert(geometry.dims == 4);
    CV_Assert(scoreMap.size[0] == 1);
    CV_Assert(geometry.size[0] == 1);
    CV_Assert(scoreMap.size[1] == 1);
    CV_Assert(geometry.size[1] == 5);
    CV_Assert(scoreMap.size[2] == geometry.size[2]);
    CV_Assert(scoreMap.size[3] == geometry.size[3]);

    std::vector<RotatedRect> boxes;
    std::vector<float> scores;
    const int height = scoreMap.size[2];
    const int width = scoreMap.size[3];
    for (int y = 0; y < height; ++y)
    {
        const float* scoresData = scoreMap.ptr<float>(0, 0, y);
        const float* x0_data = geometry.ptr<float>(0, 0, y);
        const float* x1_data = geometry.ptr<float>(0, 1, y);
        const float* x2_data = geometry.ptr<float>(0, 2, y);
        const float* x3_data = geometry.ptr<float>(0, 3, y);
        const float* anglesData = geometry.ptr<float>(0, 4, y);
        for (int x = 0; x < width; ++x)
        {
            float score = scoresData[x];
            if (score < confThreshold)
                continue;

            float offsetX = x * 4.0f, offsetY = y * 4.0f;
            float angle = anglesData[x];
            float cosA = std::cos(angle);
            float sinA = std::sin(angle);
            float h = x0_data[x] + x2_data[x];
            float w = x1_data[x] + x3_data[x];

            Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                           offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
            Point2f p1 = Point2f(-sinA * h, -cosA * h) + offset;
            Point2f p3 = Point2f(-cosA * w, sinA * w) + offset;
            boxes.push_back(RotatedRect(0.5f * (p1 + p3), Size2f(w, h), -angle * 180.0f / (float)CV_PI));
            scores.push_back(score);
        }
    }

    // Apply non-maximum suppression procedure.
    std::vector<int> indices;
    NMSBoxes(boxes, scores, confThreshold, nmsThreshold, indices);

    // Re-scale
    Point2f ratio((float)frame.cols() / impl->size.width, (float)frame.rows() / impl->size.height);
    for (uint i = 0; i < indices.size(); i++) {
        RotatedRect& box = boxes[indices[i]];

        Point2f vertices[4];
        box.points(vertices);

        std::vector<Point> result;
        for (int j = 0; j < 4; ++j)
        {
            int x = (int)(vertices[j].x * ratio.x);
            int y = (int)(vertices[j].y * ratio.y);
            result.push_back(Point(x, y));
        }
        results.push_back(result);
    }
}

}} // namespace
