// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"

#ifdef HAVE_OPENCV_DNN
#include "opencv2/dnn.hpp"
#endif

#include <algorithm>

// IoU https://stackoverflow.com/questions/61758075/intersection-over-union-iou-ground-truth-in-yolo

namespace cv
{

#ifdef HAVE_OPENCV_DNN
std::vector<std::string> labelYolox = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant",
        "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };

class ObjectDetectorYXImpl : public ObjectDetectorYX
    {
    public:
        ObjectDetectorYXImpl(const std::string& modelFilePath,
            float confThresh,
            float nmsThresh,
            dnn::Backend backId,
            dnn::Target tgtId):modelPath(modelFilePath), confThreshold(confThresh), nmsThreshold(nmsThresh), backendId(backId), targetId(tgtId)
        {
            this->num_classes = int(labelYolox.size());
            this->net = dnn::readNet(samples::findFile(modelPath));
            this->inputSize = Size(640, 640);
            this->strides = std::vector<int>{ 8, 16, 32 };
            this->net.setPreferableBackend(this->backendId);
            this->net.setPreferableTarget(this->targetId);
            this->generateAnchors();
        }

        Size getInputSize() override
        {
            return inputSize;
        }

        void setConfThreshold(float confThresh) override
        {
            confThreshold = confThresh;
        }

        float getConfThreshold() override
        {
            return confThreshold;
        }

        void setNMSThreshold(float nms_threshold) override
        {
            nmsThreshold = nms_threshold;
        }

        float getNMSThreshold() override
        {
            return nmsThreshold;
        }

        Mat preprocess(Mat img)
        {
            Mat blob;
            dnn::Image2BlobParams paramYolox;
            paramYolox.datalayout = dnn::DNN_LAYOUT_NCHW;
            paramYolox.ddepth = CV_32F;
            paramYolox.mean = Scalar::all(0);
            paramYolox.scalefactor = Scalar::all(1);
            paramYolox.size = Size(img.cols, img.rows);
            paramYolox.swapRB = true;

            blob = blobFromImageWithParams(img, paramYolox);
            return blob;
        }

        Mat infer(Mat srcimg)
        {
            Mat inputBlob = this->preprocess(srcimg);

            this->net.setInput(inputBlob);
            std::vector<Mat> outs;
            this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

            Mat predictions = this->postprocess(outs[0]);
            return predictions;
        }

        Mat postprocess(Mat outputs)
        {
            Mat dets = outputs.reshape(0, outputs.size[1]);
            Mat col01;
            add(dets.colRange(0, 2), this->grids, col01);
            Mat col23;
            exp(dets.colRange(2, 4), col23);
            std::vector<Mat> col = { col01, col23 };
            Mat boxes;
            hconcat(col, boxes);
            float* ptr = this->expandedStrides.ptr<float>(0);
            for (int r = 0; r < boxes.rows; r++, ptr++)
            {
                boxes.rowRange(r, r + 1) = *ptr * boxes.rowRange(r, r + 1);
            }
            // get boxes
            Mat boxes_xyxy(boxes.rows, boxes.cols, CV_32FC1, Scalar(1));
            Mat scores = dets.colRange(5, dets.cols).clone();
            std::vector<float> maxScores(dets.rows);
            std::vector<int> maxScoreIdx(dets.rows);
            std::vector<Rect2d> boxesXYXY(dets.rows);

            for (int r = 0; r < boxes_xyxy.rows; r++, ptr++)
            {
                boxes_xyxy.at<float>(r, 0) = boxes.at<float>(r, 0) - boxes.at<float>(r, 2) / 2.f;
                boxes_xyxy.at<float>(r, 1) = boxes.at<float>(r, 1) - boxes.at<float>(r, 3) / 2.f;
                boxes_xyxy.at<float>(r, 2) = boxes.at<float>(r, 0) + boxes.at<float>(r, 2) / 2.f;
                boxes_xyxy.at<float>(r, 3) = boxes.at<float>(r, 1) + boxes.at<float>(r, 3) / 2.f;
                // get scores and class indices
                scores.rowRange(r, r + 1) = scores.rowRange(r, r + 1) * dets.at<float>(r, 4);
                double minVal, maxVal;
                Point maxIdx;
                minMaxLoc(scores.rowRange(r, r + 1), &minVal, &maxVal, nullptr, &maxIdx);
                maxScoreIdx[r] = maxIdx.x;
                maxScores[r] = float(maxVal);
                boxesXYXY[r].x = boxes_xyxy.at<float>(r, 0);
                boxesXYXY[r].y = boxes_xyxy.at<float>(r, 1);
                boxesXYXY[r].width = boxes_xyxy.at<float>(r, 2);
                boxesXYXY[r].height = boxes_xyxy.at<float>(r, 3);
            }

            std::vector< int > keep;
            dnn::NMSBoxesBatched(boxesXYXY, maxScores, maxScoreIdx, this->confThreshold, this->nmsThreshold, keep);
            Mat candidates(int(keep.size()), 6, CV_32FC1);
            int row = 0;
            for (auto idx : keep)
            {
                boxes_xyxy.rowRange(idx, idx + 1).copyTo(candidates(Rect(0, row, 4, 1)));
                candidates.at<float>(row, 4) = maxScores[idx];
                candidates.at<float>(row, 5) = float(maxScoreIdx[idx]);
                row++;
            }
            if (keep.size() == 0)
                return Mat();
            return candidates;

        }

        int detect(InputArray inputImage, OutputArray object) override
        {
            if (inputImage.empty())
            {
                return 0;
            }
            Mat srcimg = inputImage.getMat();
            Mat inputBlob = this->preprocess(srcimg);

            this->net.setInput(inputBlob);
            std::vector<Mat> outs;
            this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

            Mat predictions = this->postprocess(outs[0]);
            predictions.convertTo(object, CV_32FC1);
            return 1;
        }

        void generateAnchors()
        {
            std::vector<std::tuple<int, int, int> > nb;
            int total = 0;

            for (auto v : this->strides)
            {
                int w = this->inputSize.width / v;
                int h = this->inputSize.height / v;
                nb.push_back(std::tuple<int, int, int>(w * h, w, v));
                total += w * h;
            }
            this->grids = Mat(total, 2, CV_32FC1);
            this->expandedStrides = Mat(total, 1, CV_32FC1);
            float* ptrGrids = this->grids.ptr<float>(0);
            float* ptrStrides = this->expandedStrides.ptr<float>(0);
            int pos = 0;
            for (auto le : nb)
            {
                int r = std::get<1>(le);
                for (int i = 0; i < std::get<0>(le); i++, pos++)
                {
                    *ptrGrids++ = float(i % r);
                    *ptrGrids++ = float(i / r);
                    *ptrStrides++ = float((std::get<2>(le)));
                }
            }
        }

    private:
        dnn::Net net;
        std::string modelPath;
        Size inputSize;
        float confThreshold;
        float nmsThreshold;
        dnn::Backend backendId;
        dnn::Target targetId;
        int num_classes;
        std::vector<int> strides;
        Mat expandedStrides;
        Mat grids;
    };
#endif

Ptr<ObjectDetectorYX> ObjectDetectorYX::create(std::string modelPath, float confThresh, float nmsThresh, dnn::Backend bId , dnn::Target tId)
{
#ifdef HAVE_OPENCV_DNN
    return makePtr<ObjectDetectorYXImpl>(modelPath, confThresh, nmsThresh, bId, tId);
#else
    CV_UNUSED(modelPath); CV_UNUSED(confThresh); CV_UNUSED(nmsThresh); CV_UNUSED(bId); CV_UNUSED(tId);
    CV_Error(cv::Error::StsNotImplemented, "cv::ObjectDetectorYX requires enabled 'dnn' module.");
#endif
}

} // namespace cv
