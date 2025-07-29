// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// This file is modified from the https://github.com/HonglinChu/NanoTrack/blob/master/ncnn_macos_nanotrack/nanotrack.cpp
// Author, HongLinChu, 1628464345@qq.com
// Adapt to OpenCV, ZihaoMu: zihaomu@outlook.com

// Link to original inference code: https://github.com/HonglinChu/NanoTrack
// Link to original training repo: https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack

#include "../precomp.hpp"
#ifdef HAVE_OPENCV_DNN
#include "opencv2/dnn.hpp"
#endif

namespace cv {

TrackerNano::TrackerNano()
{
    // nothing
}

TrackerNano::~TrackerNano()
{
    // nothing
}

TrackerNano::Params::Params()
{
    backbone = "backbone.onnx";
    neckhead = "neckhead.onnx";
#ifdef HAVE_OPENCV_DNN
    backend = dnn::DNN_BACKEND_DEFAULT;
    target = dnn::DNN_TARGET_CPU;
#else
    backend = -1;  // invalid value
    target = -1;  // invalid value
#endif
}

#ifdef HAVE_OPENCV_DNN
static void softmax(const Mat& src, Mat& dst)
{
    Mat maxVal;
    cv::max(src.row(1), src.row(0), maxVal);

    src.row(1) -= maxVal;
    src.row(0) -= maxVal;

    exp(src, dst);

    Mat sumVal = dst.row(0) + dst.row(1);
    dst.row(0) = dst.row(0) / sumVal;
    dst.row(1) = dst.row(1) / sumVal;
}

static float sizeCal(float w, float h)
{
    float pad = (w + h) * 0.5f;
    float sz2 = (w + pad) * (h + pad);
    return sqrt(sz2);
}

static Mat sizeCal(const Mat& w, const Mat& h)
{
    Mat pad = (w + h) * 0.5;
    Mat sz2 = (w + pad).mul((h + pad));

    cv::sqrt(sz2, sz2);
    return sz2;
}

// Similar python code: r = np.maximum(r, 1. / r) # r is matrix
static void elementReciprocalMax(Mat& srcDst)
{
    size_t totalV = srcDst.total();
    float* ptr = srcDst.ptr<float>(0);
    for (size_t i = 0; i < totalV; i++)
    {
        float val = *(ptr + i);
        *(ptr + i) = std::max(val, 1.0f/val);
    }
}

class TrackerNanoImpl : public TrackerNano
{
public:
    TrackerNanoImpl(const TrackerNano::Params& parameters)
    {
        backbone = dnn::readNet(parameters.backbone);
        neckhead = dnn::readNet(parameters.neckhead);

        CV_Assert(!backbone.empty());
        CV_Assert(!neckhead.empty());

        backbone.setPreferableBackend(parameters.backend);
        backbone.setPreferableTarget(parameters.target);
        neckhead.setPreferableBackend(parameters.backend);
        neckhead.setPreferableTarget(parameters.target);
    }

    TrackerNanoImpl(const dnn::Net& _backbone, const dnn::Net& _neckhead)
    {
        CV_Assert(!_backbone.empty());
        CV_Assert(!_neckhead.empty());

        backbone = _backbone;
        neckhead = _neckhead;
    }

    void init(InputArray image, const Rect& boundingBox) CV_OVERRIDE;
    bool update(InputArray image, Rect& boundingBox) CV_OVERRIDE;
    float getTrackingScore() CV_OVERRIDE;

    // Save the target bounding box for each frame.
    std::vector<float> targetSz = {0, 0};  // H and W of bounding box
    std::vector<float> targetPos = {0, 0}; // center point of bounding box (x, y)
    float tracking_score;

    struct trackerConfig
    {
        float windowInfluence = 0.455f;
        float lr = 0.37f;
        float contextAmount = 0.5;
        bool swapRB = true;
        int totalStride = 16;
        float penaltyK = 0.055f;
    };

protected:
    const int exemplarSize = 127;
    const int instanceSize = 255;

    trackerConfig trackState;
    int scoreSize;
    Size imgSize = {0, 0};
    Mat hanningWindow;
    Mat grid2searchX, grid2searchY;

    dnn::Net backbone, neckhead;
    Mat image;

    void getSubwindow(Mat& dstCrop, Mat& srcImg, int originalSz, int resizeSz);
    void generateGrids();
};

void TrackerNanoImpl::generateGrids()
{
    int sz = scoreSize;
    const int sz2 = sz / 2;

    std::vector<float> x1Vec(sz, 0);

    for (int i = 0; i < sz; i++)
    {
        x1Vec[i] = (float)(i - sz2);
    }

    Mat x1M(1, sz, CV_32FC1, x1Vec.data());

    cv::repeat(x1M, sz, 1, grid2searchX);
    cv::repeat(x1M.t(), 1, sz, grid2searchY);

    grid2searchX *= trackState.totalStride;
    grid2searchY *= trackState.totalStride;

    grid2searchX += instanceSize/2;
    grid2searchY += instanceSize/2;
}

void TrackerNanoImpl::init(InputArray image_, const Rect &boundingBox_)
{
    scoreSize = (instanceSize - exemplarSize) / trackState.totalStride + 8;
    trackState = trackerConfig();
    image = image_.getMat().clone();

    // convert Rect2d from left-up to center.
    targetPos[0] = float(boundingBox_.x) + float(boundingBox_.width) * 0.5f;
    targetPos[1] = float(boundingBox_.y) + float(boundingBox_.height) * 0.5f;

    targetSz[0] = float(boundingBox_.width);
    targetSz[1] = float(boundingBox_.height);

    imgSize = image.size();

    // Extent the bounding box.
    float sumSz = targetSz[0] + targetSz[1];
    float wExtent = targetSz[0] + trackState.contextAmount * (sumSz);
    float hExtent = targetSz[1] + trackState.contextAmount * (sumSz);
    int sz = int(cv::sqrt(wExtent * hExtent));

    Mat crop;
    getSubwindow(crop, image, sz, exemplarSize);
    Mat blob = dnn::blobFromImage(crop, 1.0, Size(), Scalar(), trackState.swapRB);

    backbone.setInput(blob);
    Mat out = backbone.forward(); // Feature extraction.
    neckhead.setInput(out, "input1");

    createHanningWindow(hanningWindow, Size(scoreSize, scoreSize), CV_32F);
    generateGrids();
}

void TrackerNanoImpl::getSubwindow(Mat& dstCrop, Mat& srcImg, int originalSz, int resizeSz)
{
    Scalar avgChans = mean(srcImg);
    Size imgSz = srcImg.size();
    int c = (originalSz + 1) / 2;

    int context_xmin = (int)(targetPos[0]) - c;
    int context_xmax = context_xmin + originalSz - 1;
    int context_ymin = (int)(targetPos[1]) - c;
    int context_ymax = context_ymin + originalSz - 1;

    int left_pad = std::max(0, -context_xmin);
    int top_pad = std::max(0, -context_ymin);
    int right_pad = std::max(0, context_xmax - imgSz.width + 1);
    int bottom_pad = std::max(0, context_ymax - imgSz.height + 1);

    context_xmin += left_pad;
    context_xmax += left_pad;
    context_ymin += top_pad;
    context_ymax += top_pad;

    Mat cropImg;
    if (left_pad == 0 && top_pad == 0 && right_pad == 0 && bottom_pad == 0)
    {
        // Crop image without padding.
        cropImg = srcImg(cv::Rect(context_xmin, context_ymin,
                                  context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    }
    else // Crop image with padding, and the padding value is avgChans
    {
        cv::Mat tmpMat;
        cv::copyMakeBorder(srcImg, tmpMat, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT, avgChans);
        cropImg = tmpMat(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    }
    resize(cropImg, dstCrop, Size(resizeSz, resizeSz));
}

bool TrackerNanoImpl::update(InputArray image_, Rect &boundingBoxRes)
{
    image = image_.getMat().clone();
    int targetSzSum = (int)(targetSz[0] + targetSz[1]);

    float wc = targetSz[0] + trackState.contextAmount * targetSzSum;
    float hc = targetSz[1] + trackState.contextAmount * targetSzSum;
    float sz = cv::sqrt(wc * hc);
    float scale_z = exemplarSize / sz;
    float sx = sz * (instanceSize / exemplarSize);
    targetSz[0] *= scale_z;
    targetSz[1] *= scale_z;

    Mat crop;
    getSubwindow(crop, image, int(sx), instanceSize);

    Mat blob = dnn::blobFromImage(crop, 1.0, Size(), Scalar(), trackState.swapRB);
    backbone.setInput(blob);
    Mat xf = backbone.forward();
    neckhead.setInput(xf, "input2");
    std::vector<String> outputName = {"output1", "output2"};
    std::vector<Mat> outs;
    neckhead.forward(outs, outputName);

    CV_Assert(outs.size() == 2);

    Mat clsScore = outs[0]; // 1x2x16x16
    Mat bboxPred = outs[1]; // 1x4x16x16

    clsScore = clsScore.reshape(0, {2, scoreSize, scoreSize});
    bboxPred = bboxPred.reshape(0, {4, scoreSize, scoreSize});

    Mat scoreSoftmax; // 2x16x16
    softmax(clsScore, scoreSoftmax);

    Mat score = scoreSoftmax.row(1);
    score = score.reshape(0, {scoreSize, scoreSize});

    Mat predX1 = grid2searchX - bboxPred.row(0).reshape(0, {scoreSize, scoreSize});
    Mat predY1 = grid2searchY - bboxPred.row(1).reshape(0, {scoreSize, scoreSize});
    Mat predX2 = grid2searchX + bboxPred.row(2).reshape(0, {scoreSize, scoreSize});
    Mat predY2 = grid2searchY + bboxPred.row(3).reshape(0, {scoreSize, scoreSize});

    // size penalty
    // scale penalty
    Mat sc = sizeCal(predX2 - predX1, predY2 - predY1)/sizeCal(targetPos[0], targetPos[1]);
    elementReciprocalMax(sc);

    // ratio penalty
    float ratioVal = targetSz[0] / targetSz[1];

    Mat ratioM(scoreSize, scoreSize, CV_32FC1, Scalar::all(ratioVal));
    Mat rc = ratioM / ((predX2 - predX1) / (predY2 - predY1));
    elementReciprocalMax(rc);

    Mat penalty;
    exp(((rc.mul(sc) - 1) * trackState.penaltyK * (-1)), penalty);
    Mat pscore = penalty.mul(score);

    // Window penalty
    pscore = pscore * (1.0 - trackState.windowInfluence) + hanningWindow * trackState.windowInfluence;

    // get Max
    int bestID[2] = { 0, 0 };
    minMaxIdx(pscore, 0, 0, 0, bestID);

    tracking_score = pscore.at<float>(bestID);

    float x1Val = predX1.at<float>(bestID);
    float x2Val = predX2.at<float>(bestID);
    float y1Val = predY1.at<float>(bestID);
    float y2Val = predY2.at<float>(bestID);

    float predXs = (x1Val + x2Val)/2;
    float predYs = (y1Val + y2Val)/2;
    float predW = (x2Val - x1Val)/scale_z;
    float predH = (y2Val - y1Val)/scale_z;

    float diffXs = (predXs - instanceSize / 2) / scale_z;
    float diffYs = (predYs - instanceSize / 2) / scale_z;

    targetSz[0] /= scale_z;
    targetSz[1] /= scale_z;

    float lr = penalty.at<float>(bestID) * score.at<float>(bestID) * trackState.lr;

    float resX = targetPos[0] + diffXs;
    float resY = targetPos[1] + diffYs;
    float resW = predW * lr + (1 - lr) * targetSz[0];
    float resH = predH * lr + (1 - lr) * targetSz[1];

    resX = std::max(0.f, std::min((float)imgSize.width, resX));
    resY = std::max(0.f, std::min((float)imgSize.height, resY));
    resW = std::max(10.f, std::min((float)imgSize.width, resW));
    resH = std::max(10.f, std::min((float)imgSize.height, resH));

    targetPos[0] = resX;
    targetPos[1] = resY;
    targetSz[0] = resW;
    targetSz[1] = resH;

    // convert center to Rect.
    boundingBoxRes = { int(resX - resW/2), int(resY - resH/2), int(resW), int(resH)};
    return true;
}

float TrackerNanoImpl::getTrackingScore()
{
    return tracking_score;
}

Ptr<TrackerNano> TrackerNano::create(const TrackerNano::Params& parameters)
{
    return makePtr<TrackerNanoImpl>(parameters);
}

Ptr<TrackerNano> TrackerNano::create(const dnn::Net& backbone, const dnn::Net& neckhead)
{
    return makePtr<TrackerNanoImpl>(backbone, neckhead);
}

#else  // OPENCV_HAVE_DNN
Ptr<TrackerNano> TrackerNano::create(const TrackerNano::Params& parameters)
{
    CV_UNUSED(parameters);
    CV_Error(cv::Error::StsNotImplemented, "to use NanoTrack, the tracking module needs to be built with opencv_dnn !");
}
#endif  // OPENCV_HAVE_DNN
}
