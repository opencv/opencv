// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"

#ifdef HAVE_OPENCV_DNN
#include "opencv2/dnn.hpp"
#endif

namespace cv {

TrackerDaSiamRPN::TrackerDaSiamRPN()
{
    // nothing
}

TrackerDaSiamRPN::~TrackerDaSiamRPN()
{
    // nothing
}

TrackerDaSiamRPN::Params::Params()
{
    model = "dasiamrpn_model.onnx";
    kernel_cls1 = "dasiamrpn_kernel_cls1.onnx";
    kernel_r1 = "dasiamrpn_kernel_r1.onnx";
#ifdef HAVE_OPENCV_DNN
    backend = dnn::DNN_BACKEND_DEFAULT;
    target = dnn::DNN_TARGET_CPU;
#else
    backend = -1;  // invalid value
    target = -1;  // invalid value
#endif
}

#ifdef HAVE_OPENCV_DNN

template <typename T> static
T sizeCal(const T& w, const T& h)
{
    T pad = (w + h) * T(0.5);
    T sz2 = (w + pad) * (h + pad);
    return sqrt(sz2);
}

template <>
Mat sizeCal(const Mat& w, const Mat& h)
{
    Mat pad = (w + h) * 0.5;
    Mat sz2 = (w + pad).mul((h + pad));

    cv::sqrt(sz2, sz2);
    return sz2;
}

class TrackerDaSiamRPNImpl : public TrackerDaSiamRPN
{
public:
    TrackerDaSiamRPNImpl(const TrackerDaSiamRPN::Params& params)
    {
        siamRPN = dnn::readNet(params.model);
        siamKernelCL1 = dnn::readNet(params.kernel_cls1);
        siamKernelR1 = dnn::readNet(params.kernel_r1);

        CV_Assert(!siamRPN.empty());
        CV_Assert(!siamKernelCL1.empty());
        CV_Assert(!siamKernelR1.empty());

        siamRPN.setPreferableBackend(params.backend);
        siamRPN.setPreferableTarget(params.target);
        siamKernelR1.setPreferableBackend(params.backend);
        siamKernelR1.setPreferableTarget(params.target);
        siamKernelCL1.setPreferableBackend(params.backend);
        siamKernelCL1.setPreferableTarget(params.target);
    }

    TrackerDaSiamRPNImpl(const dnn::Net& siam_rpn, const dnn::Net& kernel_cls1, const dnn::Net& kernel_r1)
    {
        CV_Assert(!siam_rpn.empty());
        CV_Assert(!kernel_cls1.empty());
        CV_Assert(!kernel_r1.empty());

        siamRPN = siam_rpn;
        siamKernelCL1 = kernel_cls1;
        siamKernelR1 = kernel_r1;
    }

    void init(InputArray image, const Rect& boundingBox) CV_OVERRIDE;
    bool update(InputArray image, Rect& boundingBox) CV_OVERRIDE;
    float getTrackingScore() CV_OVERRIDE;

protected:
    dnn::Net siamRPN, siamKernelR1, siamKernelCL1;
    Rect boundingBox_;
    Mat image_;
    struct trackerConfig
    {
        float windowInfluence = 0.43f;
        float lr = 0.4f;
        int scale = 8;
        bool swapRB = false;
        int totalStride = 8;
        float penaltyK = 0.055f;
        int exemplarSize = 127;
        int instanceSize = 271;
        float contextAmount = 0.5f;
        std::vector<float> ratios = { 0.33f, 0.5f, 1.0f, 2.0f, 3.0f };
        int anchorNum = int(ratios.size());
        Mat anchors;
        Mat windows;
        Scalar avgChans;
        Size imgSize = { 0, 0 };
        Rect2f targetBox = { 0, 0, 0, 0 };
        int scoreSize = (instanceSize - exemplarSize) / totalStride + 1;
        float tracking_score;

        void update_scoreSize()
        {
            scoreSize = int((instanceSize - exemplarSize) / totalStride + 1);
        }
    };
    trackerConfig trackState;

    void softmax(const Mat& src, Mat& dst);
    void elementMax(Mat& src);
    Mat generateHanningWindow();
    Mat generateAnchors();
    Mat getSubwindow(Mat& img, const Rect2f& targetBox, float originalSize, Scalar avgChans);
    void trackerInit(Mat img);
    void trackerEval(Mat img);
};

void TrackerDaSiamRPNImpl::init(InputArray image, const Rect& boundingBox)
{
    image_ = image.getMat().clone();

    trackState.update_scoreSize();
    trackState.targetBox = Rect2f(
        float(boundingBox.x) + float(boundingBox.width) * 0.5f,  // FIXIT don't use center in Rect structures, it is confusing
        float(boundingBox.y) + float(boundingBox.height) * 0.5f,
        float(boundingBox.width),
        float(boundingBox.height)
    );
    trackerInit(image_);
}

void TrackerDaSiamRPNImpl::trackerInit(Mat img)
{
    Rect2f targetBox = trackState.targetBox;
    Mat anchors = generateAnchors();
    trackState.anchors = anchors;

    Mat windows = generateHanningWindow();

    trackState.windows = windows;
    trackState.imgSize = img.size();

    trackState.avgChans = mean(img);
    float wc = targetBox.width + trackState.contextAmount * (targetBox.width + targetBox.height);
    float hc = targetBox.height + trackState.contextAmount * (targetBox.width + targetBox.height);
    float sz = (float)cvRound(sqrt(wc * hc));

    Mat zCrop = getSubwindow(img, targetBox, sz, trackState.avgChans);
    Mat blob;

    dnn::blobFromImage(zCrop, blob, 1.0, Size(trackState.exemplarSize, trackState.exemplarSize), Scalar(), trackState.swapRB, false, CV_32F);
    siamRPN.setInput(blob);
    Mat out1;
    siamRPN.forward(out1, "onnx_node_output_0!63");

    siamKernelCL1.setInput(out1);
    siamKernelR1.setInput(out1);

    Mat cls1 = siamKernelCL1.forward();
    Mat r1 = siamKernelR1.forward();
    std::vector<int> r1_shape = { 20, 256, 4, 4 }, cls1_shape = { 10, 256, 4, 4 };

    siamRPN.setParam(siamRPN.getLayerId("onnx_node_output_0!65"), 0, r1.reshape(0, r1_shape));
    siamRPN.setParam(siamRPN.getLayerId("onnx_node_output_0!68"), 0, cls1.reshape(0, cls1_shape));
}

bool TrackerDaSiamRPNImpl::update(InputArray image, Rect& boundingBox)
{
    image_ = image.getMat().clone();
    trackerEval(image_);
    boundingBox = {
        int(trackState.targetBox.x - int(trackState.targetBox.width / 2)),
        int(trackState.targetBox.y - int(trackState.targetBox.height / 2)),
        int(trackState.targetBox.width),
        int(trackState.targetBox.height)
    };
    return true;
}

void TrackerDaSiamRPNImpl::trackerEval(Mat img)
{
    Rect2f targetBox = trackState.targetBox;

    float wc = targetBox.height + trackState.contextAmount * (targetBox.width + targetBox.height);
    float hc = targetBox.width + trackState.contextAmount * (targetBox.width + targetBox.height);

    float sz = sqrt(wc * hc);
    float scaleZ = trackState.exemplarSize / sz;

    float searchSize = float((trackState.instanceSize - trackState.exemplarSize) / 2);
    float pad = searchSize / scaleZ;
    float sx = sz + 2 * pad;

    Mat xCrop = getSubwindow(img, targetBox, (float)cvRound(sx), trackState.avgChans);

    Mat blob;
    std::vector<Mat> outs;
    std::vector<String> outNames;
    Mat delta, score;
    Mat sc, rc, penalty, pscore;

    dnn::blobFromImage(xCrop, blob, 1.0, Size(trackState.instanceSize, trackState.instanceSize), Scalar(), trackState.swapRB, false, CV_32F);

    siamRPN.setInput(blob);

    outNames = siamRPN.getUnconnectedOutLayersNames();
    siamRPN.forward(outs, outNames);

    delta = outs[0];
    score = outs[1];

    score = score.reshape(0, { 2, trackState.anchorNum, trackState.scoreSize, trackState.scoreSize });
    delta = delta.reshape(0, { 4, trackState.anchorNum, trackState.scoreSize, trackState.scoreSize });

    softmax(score, score);

    targetBox.width *= scaleZ;
    targetBox.height *= scaleZ;

    score = score.row(1);
    score = score.reshape(0, { 5, 19, 19 });

    // Post processing
    delta.row(0) = delta.row(0).mul(trackState.anchors.row(2)) + trackState.anchors.row(0);
    delta.row(1) = delta.row(1).mul(trackState.anchors.row(3)) + trackState.anchors.row(1);
    exp(delta.row(2), delta.row(2));
    delta.row(2) = delta.row(2).mul(trackState.anchors.row(2));
    exp(delta.row(3), delta.row(3));
    delta.row(3) = delta.row(3).mul(trackState.anchors.row(3));

    sc = sizeCal(delta.row(2), delta.row(3)) / sizeCal(targetBox.width, targetBox.height);
    elementMax(sc);

    rc = delta.row(2).mul(1 / delta.row(3));
    rc = (targetBox.width / targetBox.height) / rc;
    elementMax(rc);

    // Calculating the penalty
    exp(((rc.mul(sc) - 1.) * trackState.penaltyK * (-1.0)), penalty);
    penalty = penalty.reshape(0, { trackState.anchorNum, trackState.scoreSize, trackState.scoreSize });

    pscore = penalty.mul(score);
    pscore = pscore * (1.0 - trackState.windowInfluence) + trackState.windows * trackState.windowInfluence;

    int bestID[2] = { 0, 0 };
    // Find the index of best score.
    minMaxIdx(pscore.reshape(0, { trackState.anchorNum * trackState.scoreSize * trackState.scoreSize, 1 }), 0, 0, 0, bestID);
    delta = delta.reshape(0, { 4, trackState.anchorNum * trackState.scoreSize * trackState.scoreSize });
    penalty = penalty.reshape(0, { trackState.anchorNum * trackState.scoreSize * trackState.scoreSize, 1 });
    score = score.reshape(0, { trackState.anchorNum * trackState.scoreSize * trackState.scoreSize, 1 });

    int index[2] = { 0, bestID[0] };
    Rect2f resBox = { 0, 0, 0, 0 };

    resBox.x = delta.at<float>(index) / scaleZ;
    index[0] = 1;
    resBox.y = delta.at<float>(index) / scaleZ;
    index[0] = 2;
    resBox.width = delta.at<float>(index) / scaleZ;
    index[0] = 3;
    resBox.height = delta.at<float>(index) / scaleZ;

    float lr = penalty.at<float>(bestID) * score.at<float>(bestID) * trackState.lr;

    resBox.x = resBox.x + targetBox.x;
    resBox.y = resBox.y + targetBox.y;
    targetBox.width /= scaleZ;
    targetBox.height /= scaleZ;

    resBox.width = targetBox.width * (1 - lr) + resBox.width * lr;
    resBox.height = targetBox.height * (1 - lr) + resBox.height * lr;

    resBox.x = float(fmax(0., fmin(float(trackState.imgSize.width), resBox.x)));
    resBox.y = float(fmax(0., fmin(float(trackState.imgSize.height), resBox.y)));
    resBox.width = float(fmax(10., fmin(float(trackState.imgSize.width), resBox.width)));
    resBox.height = float(fmax(10., fmin(float(trackState.imgSize.height), resBox.height)));

    trackState.targetBox = resBox;
    trackState.tracking_score = score.at<float>(bestID);
}

float TrackerDaSiamRPNImpl::getTrackingScore()
{
    return trackState.tracking_score;
}

void TrackerDaSiamRPNImpl::softmax(const Mat& src, Mat& dst)
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

void TrackerDaSiamRPNImpl::elementMax(Mat& src)
{
    int* p = src.size.p;
    int index[4] = { 0, 0, 0, 0 };
    for (int n = 0; n < *p; n++)
    {
        for (int k = 0; k < *(p + 1); k++)
        {
            for (int i = 0; i < *(p + 2); i++)
            {
                for (int j = 0; j < *(p + 3); j++)
                {
                    index[0] = n, index[1] = k, index[2] = i, index[3] = j;
                    float& v = src.at<float>(index);
                    v = fmax(v, 1.0f / v);
                }
            }
        }
    }
}

Mat TrackerDaSiamRPNImpl::generateHanningWindow()
{
    Mat baseWindows, HanningWindows;

    createHanningWindow(baseWindows, Size(trackState.scoreSize, trackState.scoreSize), CV_32F);
    baseWindows = baseWindows.reshape(0, { 1, trackState.scoreSize, trackState.scoreSize });
    HanningWindows = baseWindows.clone();
    for (int i = 1; i < trackState.anchorNum; i++)
    {
        HanningWindows.push_back(baseWindows);
    }

    return HanningWindows;
}

Mat TrackerDaSiamRPNImpl::generateAnchors()
{
    int totalStride = trackState.totalStride, scales = trackState.scale, scoreSize = trackState.scoreSize;
    std::vector<float> ratios = trackState.ratios;
    std::vector<Rect2f> baseAnchors;
    int anchorNum = int(ratios.size());
    int size = totalStride * totalStride;

    float ori = -(float(scoreSize / 2)) * float(totalStride);

    for (auto i = 0; i < anchorNum; i++)
    {
        int ws = int(sqrt(size / ratios[i]));
        int hs = int(ws * ratios[i]);

        float wws = float(ws) * scales;
        float hhs = float(hs) * scales;
        Rect2f anchor = { 0, 0, wws, hhs };
        baseAnchors.push_back(anchor);
    }

    int anchorIndex[4] = { 0, 0, 0, 0 };
    const int sizes[4] = { 4, (int)ratios.size(), scoreSize, scoreSize };
    Mat anchors(4, sizes, CV_32F);

    for (auto i = 0; i < scoreSize; i++)
    {
        for (auto j = 0; j < scoreSize; j++)
        {
            for (auto k = 0; k < anchorNum; k++)
            {
                anchorIndex[0] = 1, anchorIndex[1] = k, anchorIndex[2] = i, anchorIndex[3] = j;
                anchors.at<float>(anchorIndex) = ori + totalStride * i;

                anchorIndex[0] = 0;
                anchors.at<float>(anchorIndex) = ori + totalStride * j;

                anchorIndex[0] = 2;
                anchors.at<float>(anchorIndex) = baseAnchors[k].width;

                anchorIndex[0] = 3;
                anchors.at<float>(anchorIndex) = baseAnchors[k].height;
            }
        }
    }

    return anchors;
}

Mat TrackerDaSiamRPNImpl::getSubwindow(Mat& img, const Rect2f& targetBox, float originalSize, Scalar avgChans)
{
    Mat zCrop, dst;
    Size imgSize = img.size();
    float c = (originalSize + 1) / 2;
    float xMin = (float)cvRound(targetBox.x - c);
    float xMax = xMin + originalSize - 1;
    float yMin = (float)cvRound(targetBox.y - c);
    float yMax = yMin + originalSize - 1;

    int leftPad = (int)(fmax(0., -xMin));
    int topPad = (int)(fmax(0., -yMin));
    int rightPad = (int)(fmax(0., xMax - imgSize.width + 1));
    int bottomPad = (int)(fmax(0., yMax - imgSize.height + 1));

    xMin = xMin + leftPad;
    xMax = xMax + leftPad;
    yMax = yMax + topPad;
    yMin = yMin + topPad;

    if (topPad == 0 && bottomPad == 0 && leftPad == 0 && rightPad == 0)
    {
        img(Rect(int(xMin), int(yMin), int(xMax - xMin + 1), int(yMax - yMin + 1))).copyTo(zCrop);
    }
    else
    {
        copyMakeBorder(img, dst, topPad, bottomPad, leftPad, rightPad, BORDER_CONSTANT, avgChans);
        dst(Rect(int(xMin), int(yMin), int(xMax - xMin + 1), int(yMax - yMin + 1))).copyTo(zCrop);
    }

    return zCrop;
}

Ptr<TrackerDaSiamRPN> TrackerDaSiamRPN::create(const TrackerDaSiamRPN::Params& parameters)
{
    return makePtr<TrackerDaSiamRPNImpl>(parameters);
}

Ptr<TrackerDaSiamRPN> TrackerDaSiamRPN::create(const dnn::Net& siam_rpn, const dnn::Net& kernel_cls1, const dnn::Net& kernel_r1)
{
    return makePtr<TrackerDaSiamRPNImpl>(siam_rpn, kernel_cls1, kernel_r1);
}

#else  // OPENCV_HAVE_DNN
Ptr<TrackerDaSiamRPN> TrackerDaSiamRPN::create(const TrackerDaSiamRPN::Params& parameters)
{
    (void)(parameters);
    CV_Error(cv::Error::StsNotImplemented, "to use DaSiamRPN, the tracking module needs to be built with opencv_dnn !");
}
#endif  // OPENCV_HAVE_DNN
}
