// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"

#ifdef HAVE_OPENCV_DNN
#include "opencv2/dnn.hpp"
#endif

namespace cv {

TrackerGOTURN::TrackerGOTURN()
{
    // nothing
}

TrackerGOTURN::~TrackerGOTURN()
{
    // nothing
}

TrackerGOTURN::Params::Params()
{
    modelTxt = "goturn.prototxt";
    modelBin = "goturn.caffemodel";
}

#ifdef HAVE_OPENCV_DNN

class TrackerGOTURNImpl : public TrackerGOTURN
{
public:
    TrackerGOTURNImpl(const dnn::Net& model)
    {
        CV_Assert(!model.empty());
        net = model;
    }

    TrackerGOTURNImpl(const TrackerGOTURN::Params& parameters)
    {
        // Load GOTURN architecture from *.prototxt and pretrained weights from *.caffemodel
        net = dnn::readNetFromCaffe(parameters.modelTxt, parameters.modelBin);
        CV_Assert(!net.empty());
    }

    void init(InputArray image, const Rect& boundingBox) CV_OVERRIDE;
    bool update(InputArray image, Rect& boundingBox) CV_OVERRIDE;

    void setBoudingBox(Rect boundingBox)
    {
        if (frameSize_.empty())
            CV_Error(Error::StsInternal, "Set image first");
        boundingBox_ = boundingBox & Rect(Point(0, 0), frameSize_);
    }

    static Rect2f computeTargetPatchRect(const Rect2d& bb, Size frameSz)
    {
        const float padTargetPatch = 2.0f;
        const float cx = (float)(bb.x + bb.width / 2.0);
        const float cy = (float)(bb.y + bb.height / 2.0);
        Rect2f r;
        r.width = std::min((float)(bb.width * padTargetPatch), (float)frameSz.width);
        r.height = std::min((float)(bb.height * padTargetPatch), (float)frameSz.height);
        r.x = (float)(cx - bb.width * padTargetPatch / 2.0 + r.width);
        r.y = (float)(cy - bb.height * padTargetPatch / 2.0 + r.height);
        r.x = std::max(-frameSz.width * 0.5f, std::min(r.x, frameSz.width * 1.5f));
        r.y = std::max(-frameSz.height * 0.5f, std::min(r.y, frameSz.height * 1.5f));
        return r;
    }

    static void extractReplicatedPatch(const Mat& frame, const Rect2f& patchRectPadded, Mat& dstPatch)
    {
        const Rect roiPadded(patchRectPadded);
        const int rw = roiPadded.width;
        const int rh = roiPadded.height;
        if (rw <= 0 || rh <= 0 || frame.empty())
        {
            dstPatch.release();
            return;
        }
        const int padLeft = (int)patchRectPadded.width;
        const int padTop = (int)patchRectPadded.height;
        const int ux = roiPadded.x - padLeft;
        const int uy = roiPadded.y - padTop;
        const int W = frame.cols;
        const int H = frame.rows;

        const int ix0 = std::max(0, std::min(ux, W - 1));
        const int ix1 = std::max(0, std::min(ux + rw - 1, W - 1));
        const int iy0 = std::max(0, std::min(uy, H - 1));
        const int iy1 = std::max(0, std::min(uy + rh - 1, H - 1));
        const Rect srcRoi(ix0, iy0, ix1 - ix0 + 1, iy1 - iy0 + 1);

        const int left = std::max(0, std::min(rw - 1, -ux));
        const int right = rw - srcRoi.width - left;
        const int top = std::max(0, std::min(rh - 1, -uy));
        const int bottom = rh - srcRoi.height - top;

        if (left == 0 && right == 0 && top == 0 && bottom == 0)
            frame(srcRoi).copyTo(dstPatch);
        else
            copyMakeBorder(frame(srcRoi), dstPatch, top, bottom, left, right, BORDER_REPLICATE);
    }

    void cacheTargetFromFrame(const Mat& frame)
    {
        const int INPUT_SIZE = 227;
        targetPatchRect_ = computeTargetPatchRect(Rect2d(boundingBox_), frameSize_);
        Mat rawTarget;
        extractReplicatedPatch(frame, targetPatchRect_, rawTarget);
        if (!rawTarget.empty())
            resize(rawTarget, targetPatch_, Size(INPUT_SIZE, INPUT_SIZE), 0, 0, INTER_LINEAR_EXACT);
        else
            targetPatch_.release();
    }

    dnn::Net net;
    Rect boundingBox_;
    Size frameSize_;
    Rect2f targetPatchRect_;
    Mat targetPatch_;
};

void TrackerGOTURNImpl::init(InputArray image, const Rect& boundingBox)
{
    Mat frame = image.getMat();
    CV_Assert(!frame.empty());
    frameSize_ = frame.size();
    setBoudingBox(boundingBox);
    cacheTargetFromFrame(frame);
}

bool TrackerGOTURNImpl::update(InputArray image, Rect& boundingBox)
{
    const int INPUT_SIZE = 227;
    Mat curFrame = image.getMat();
    CV_Assert(!curFrame.empty());
    Rect curBB;

    const Rect2f targetPatchRect = targetPatchRect_;
    Mat searchPatch;
    extractReplicatedPatch(curFrame, targetPatchRect, searchPatch);

    CV_Assert(!targetPatch_.empty() && !searchPatch.empty());
    resize(searchPatch, searchPatch, Size(INPUT_SIZE, INPUT_SIZE), 0, 0, INTER_LINEAR_EXACT);

    // Convert to Float type and subtract mean
    Mat targetBlob = dnn::blobFromImage(targetPatch_, 1.0f, Size(), Scalar::all(128), false);
    Mat searchBlob = dnn::blobFromImage(searchPatch, 1.0f, Size(), Scalar::all(128), false);
    searchPatch.release();

    net.setInput(targetBlob, "data1");
    net.setInput(searchBlob, "data2");

    Mat resMat = net.forward("scale").reshape(1, 1);

    curBB.x = cvRound(targetPatchRect.x + (resMat.at<float>(0) * targetPatchRect.width / INPUT_SIZE) - targetPatchRect.width);
    curBB.y = cvRound(targetPatchRect.y + (resMat.at<float>(1) * targetPatchRect.height / INPUT_SIZE) - targetPatchRect.height);
    curBB.width = cvRound((resMat.at<float>(2) - resMat.at<float>(0)) * targetPatchRect.width / INPUT_SIZE);
    curBB.height = cvRound((resMat.at<float>(3) - resMat.at<float>(1)) * targetPatchRect.height / INPUT_SIZE);

    // Predicted BB
    boundingBox = curBB & Rect(Point(0, 0), frameSize_);

    // Cache next target patch (227x227) instead of retaining full-resolution frame
    frameSize_ = curFrame.size();
    setBoudingBox(curBB);
    cacheTargetFromFrame(curFrame);
    return true;
}

Ptr<TrackerGOTURN> TrackerGOTURN::create(const TrackerGOTURN::Params& parameters)
{
    return makePtr<TrackerGOTURNImpl>(parameters);
}

Ptr<TrackerGOTURN> TrackerGOTURN::create(const dnn::Net& model)
{
    return makePtr<TrackerGOTURNImpl>(model);
}

#else  // OPENCV_HAVE_DNN
Ptr<TrackerGOTURN> TrackerGOTURN::create(const TrackerGOTURN::Params& parameters)
{
    (void)(parameters);
    CV_Error(cv::Error::StsNotImplemented, "to use GOTURN, the tracking module needs to be built with opencv_dnn !");
}
#endif  // OPENCV_HAVE_DNN

}  // namespace cv
