// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Author, PengyuLiu, 1872918507@qq.com

#include "../precomp.hpp"
#ifdef HAVE_OPENCV_DNN
#include "opencv2/dnn.hpp"
#endif

namespace cv {

TrackerVit::TrackerVit()
{
    // nothing
}

TrackerVit::~TrackerVit()
{
    // nothing
}

TrackerVit::Params::Params()
{
    net = "vitTracker.onnx";
    meanvalue = Scalar{0.485, 0.456, 0.406};
    stdvalue = Scalar{0.229, 0.224, 0.225};
#ifdef HAVE_OPENCV_DNN
    backend = dnn::DNN_BACKEND_DEFAULT;
    target = dnn::DNN_TARGET_CPU;
#else
    backend = -1;  // invalid value
    target = -1;  // invalid value
#endif
}

#ifdef HAVE_OPENCV_DNN

class TrackerVitImpl : public TrackerVit
{
public:
    TrackerVitImpl(const TrackerVit::Params& parameters)
        : params(parameters)
    {
        net = dnn::readNet(params.net);
        CV_Assert(!net.empty());

        net.setPreferableBackend(params.backend);
        net.setPreferableTarget(params.target);
    }

    void init(InputArray image, const Rect& boundingBox) CV_OVERRIDE;
    bool update(InputArray image, Rect& boundingBox) CV_OVERRIDE;
    float getTrackingScore() CV_OVERRIDE;

    Rect rect_last;
    float tracking_score;

    TrackerVit::Params params;


protected:
    void preprocess(const Mat& src, Mat& dst, Size size);

    const Size searchSize{256, 256};
    const Size templateSize{128, 128};

    Mat hanningWindow;

    dnn::Net net;
    Mat image;
};

static void crop_image(const Mat& src, Mat& dst, Rect box, int factor)
{
    int x = box.x, y = box.y, w = box.width, h = box.height;
    int crop_sz = cvCeil(sqrt(w * h) * factor);

    int x1 = x + (w - crop_sz) / 2;
    int x2 = x1 + crop_sz;
    int y1 = y + (h - crop_sz) / 2;
    int y2 = y1 + crop_sz;

    int x1_pad = std::max(0, -x1);
    int y1_pad = std::max(0, -y1);
    int x2_pad = std::max(x2 - src.size[1] + 1, 0);
    int y2_pad = std::max(y2 - src.size[0] + 1, 0);

    Rect roi(x1 + x1_pad, y1 + y1_pad, x2 - x2_pad - x1 - x1_pad, y2 - y2_pad - y1 - y1_pad);
    Mat im_crop = src(roi);
    copyMakeBorder(im_crop, dst, y1_pad, y2_pad, x1_pad, x2_pad, BORDER_CONSTANT);
}

void TrackerVitImpl::preprocess(const Mat& src, Mat& dst, Size size)
{
    Mat mean = Mat(size, CV_32FC3, params.meanvalue);
    Mat std = Mat(size, CV_32FC3, params.stdvalue);
    mean = dnn::blobFromImage(mean, 1.0, Size(), Scalar(), false);
    std = dnn::blobFromImage(std, 1.0, Size(), Scalar(), false);

    Mat img;
    resize(src, img, size);

    dst = dnn::blobFromImage(img, 1.0, Size(), Scalar(), false);
    dst /= 255;
    dst = (dst - mean) / std;
}

static Mat hann1d(int sz, bool centered = true) {
    Mat hanningWindow(sz, 1, CV_32FC1);
    float* data = hanningWindow.ptr<float>(0);

    if(centered) {
        for(int i = 0; i < sz; i++) {
            float val = 0.5f * (1.f - std::cos(static_cast<float>(2 * M_PI / (sz + 1)) * (i + 1)));
            data[i] = val;
        }
    }
    else {
        int half_sz = sz / 2;
        for(int i = 0; i <= half_sz; i++) {
            float val = 0.5f * (1.f + std::cos(static_cast<float>(2 * M_PI / (sz + 2)) * i));
            data[i] = val;
            data[sz - 1 - i] = val;
        }
    }

    return hanningWindow;
}

static Mat hann2d(Size size, bool centered = true) {
    int rows = size.height;
    int cols = size.width;

    Mat hanningWindowRows = hann1d(rows, centered);
    Mat hanningWindowCols = hann1d(cols, centered);

    Mat hanningWindow = hanningWindowRows * hanningWindowCols.t();

    return hanningWindow;
}

static Rect returnfromcrop(float x, float y, float w, float h, Rect res_Last)
{
    int cropwindowwh = 4 * cvFloor(sqrt(res_Last.width * res_Last.height));
    int x0 = res_Last.x + (res_Last.width - cropwindowwh) / 2;
    int y0 = res_Last.y + (res_Last.height - cropwindowwh) / 2;
    Rect finalres;
    finalres.x = cvFloor(x * cropwindowwh + x0);
    finalres.y = cvFloor(y * cropwindowwh + y0);
    finalres.width = cvFloor(w * cropwindowwh);
    finalres.height = cvFloor(h * cropwindowwh);
    return finalres;
}

void TrackerVitImpl::init(InputArray image_, const Rect &boundingBox_)
{
    image = image_.getMat().clone();
    Mat crop;
    crop_image(image, crop, boundingBox_, 2);
    Mat blob;
    preprocess(crop, blob, templateSize);
    net.setInput(blob, "template");
    Size size(16, 16);
    hanningWindow = hann2d(size, false);
    rect_last = boundingBox_;
}

bool TrackerVitImpl::update(InputArray image_, Rect &boundingBoxRes)
{
    image = image_.getMat().clone();
    Mat crop;
    crop_image(image, crop, rect_last, 4);
    Mat blob;
    preprocess(crop, blob, searchSize);
    net.setInput(blob, "search");
    std::vector<String> outputName = {"output1", "output2", "output3"};
    std::vector<Mat> outs;
    net.forward(outs, outputName);
    CV_Assert(outs.size() == 3);

    Mat conf_map = outs[0].reshape(0, {16, 16});
    Mat size_map = outs[1].reshape(0, {2, 16, 16});
    Mat offset_map = outs[2].reshape(0, {2, 16, 16});

    multiply(conf_map, (1.0 - hanningWindow), conf_map);

    double maxVal;
    Point maxLoc;
    minMaxLoc(conf_map, nullptr, &maxVal, nullptr, &maxLoc);
    tracking_score = static_cast<float>(maxVal);

    float cx = (maxLoc.x + offset_map.at<float>(0, maxLoc.y, maxLoc.x)) / 16;
    float cy = (maxLoc.y + offset_map.at<float>(1, maxLoc.y, maxLoc.x)) / 16;
    float w = size_map.at<float>(0, maxLoc.y, maxLoc.x);
    float h = size_map.at<float>(1, maxLoc.y, maxLoc.x);

    Rect finalres = returnfromcrop(cx - w / 2, cy - h / 2, w, h, rect_last);
    rect_last = finalres;
    boundingBoxRes = finalres;
    return true;
}

float TrackerVitImpl::getTrackingScore()
{
    return tracking_score;
}

Ptr<TrackerVit> TrackerVit::create(const TrackerVit::Params& parameters)
{
    return makePtr<TrackerVitImpl>(parameters);
}

#else  // OPENCV_HAVE_DNN
Ptr<TrackerVit> TrackerVit::create(const TrackerVit::Params& parameters)
{
    CV_UNUSED(parameters);
    CV_Error(Error::StsNotImplemented, "to use vittrack, the tracking module needs to be built with opencv_dnn !");
}
#endif  // OPENCV_HAVE_DNN
}
