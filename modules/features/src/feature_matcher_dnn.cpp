// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "opencv2/features/lightglue.hpp"

#ifdef HAVE_OPENCV_DNN
#include "opencv2/dnn/dnn.hpp"
#endif

namespace cv {
namespace features {

LightGlue::Params::Params()
{
    modelPath = String();
    engine = 4; // dnn::ENGINE_ORT
    backend = -1;
    target = -1;
    disableWinograd = true;

    kpts0Name = "kpts0";
    desc0Name = "desc0";
    kpts1Name = "kpts1";
    desc1Name = "desc1";

    matches0Name = "matches0";
    mscores0Name = "mscores0";
}

namespace {

#ifdef HAVE_OPENCV_DNN

static Mat toKeypointMat(InputArray kpts)
{
    Mat m = kpts.getMat();
    if (m.empty())
        return Mat();

    Mat out;
    if (m.type() == CV_32FC2 || m.type() == CV_64FC2)
    {
        Mat reshaped = m.reshape(1, static_cast<int>(m.total()));
        reshaped.convertTo(out, CV_32F);
        return out;
    }

    if (m.channels() == 1 && m.cols == 2)
    {
        m.convertTo(out, CV_32F);
        return out;
    }

    if (m.channels() >= 2 && m.cols == 1)
    {
        Mat reshaped = m.reshape(1, m.rows);
        if (reshaped.cols >= 2)
        {
            out = reshaped.colRange(0, 2).clone();
            out.convertTo(out, CV_32F);
            return out;
        }
    }

    CV_Error(Error::StsBadArg, "Keypoints must be Nx2 float matrix or vector<Point2f>");
}

static Mat normalizeKeypoints(const Mat& kpts, const Size& imageSize)
{
    CV_Assert(kpts.type() == CV_32F);
    CV_Assert(kpts.cols == 2);

    CV_Assert(imageSize.width > 1 && imageSize.height > 1);

    Mat normalized = kpts.clone();
    for (int i = 0; i < normalized.rows; ++i)
    {
        normalized.at<float>(i, 0) /= static_cast<float>(imageSize.width - 1);
        normalized.at<float>(i, 1) /= static_cast<float>(imageSize.height - 1);
    }
    return normalized;
}

static Size estimateImageSize(const Mat& kpts)
{
    if (kpts.empty())
        return Size();

    double minX = 0.0, maxX = 0.0, minY = 0.0, maxY = 0.0;
    minMaxLoc(kpts.col(0), &minX, &maxX);
    minMaxLoc(kpts.col(1), &minY, &maxY);

    const int width = cvRound(maxX) + 1;
    const int height = cvRound(maxY) + 1;
    return Size(std::max(width, 2), std::max(height, 2));
}

class LightGlueImpl CV_FINAL : public LightGlue
{
public:
    explicit LightGlueImpl(const LightGlue::Params& params)
        : params_(params)
    {
        if (!params_.modelPath.empty())
            setModel(params_.modelPath);
    }

    void setModel(const String& modelPath) CV_OVERRIDE
    {
        params_.modelPath = modelPath;
        net_ = dnn::readNetFromONNX(modelPath, params_.engine);
        CV_Assert(!net_.empty());

        if (params_.disableWinograd)
            net_.enableWinograd(false);
        if (params_.backend >= 0)
            net_.setPreferableBackend(params_.backend);
        if (params_.target >= 0)
            net_.setPreferableTarget(params_.target);
    }

    String getModel() const CV_OVERRIDE
    {
        return params_.modelPath;
    }

    void match(InputArray,
               InputArray,
               std::vector<DMatch>&,
               InputArray) const CV_OVERRIDE
    {
        CV_Error(Error::StsBadArg, "LightGlue requires keypoint inputs. Use the keypoint-aware match() overload");
    }

    void match(InputArray queryKpts,
               InputArray queryDesc,
               InputArray trainKpts,
               InputArray trainDesc,
               std::vector<DMatch>& matches,
               InputArray,
               Size queryImageSize,
               Size trainImageSize) const CV_OVERRIDE
    {
        CV_Assert(!net_.empty());

        Mat qKpts = toKeypointMat(queryKpts);
        Mat tKpts = toKeypointMat(trainKpts);
        Mat qDesc = queryDesc.getMat();
        Mat tDesc = trainDesc.getMat();

        CV_Assert(!qKpts.empty() && !tKpts.empty());
        CV_Assert(!qDesc.empty() && !tDesc.empty());
        CV_Assert(qDesc.rows == qKpts.rows);
        CV_Assert(tDesc.rows == tKpts.rows);

        if (queryImageSize.empty())
            queryImageSize = estimateImageSize(qKpts);
        if (trainImageSize.empty())
            trainImageSize = estimateImageSize(tKpts);

        Mat qNorm = normalizeKeypoints(qKpts, queryImageSize);
        Mat tNorm = normalizeKeypoints(tKpts, trainImageSize);

        Mat qDesc32, tDesc32;
        qDesc.convertTo(qDesc32, CV_32F);
        tDesc.convertTo(tDesc32, CV_32F);

        int qKptsShape[] = {1, qNorm.rows, 2};
        Mat qKptsBlob(3, qKptsShape, CV_32FC1, qNorm.data);

        int tKptsShape[] = {1, tNorm.rows, 2};
        Mat tKptsBlob(3, tKptsShape, CV_32FC1, tNorm.data);

        int qDescShape[] = {1, qDesc32.rows, qDesc32.cols};
        Mat qDescBlob(3, qDescShape, CV_32FC1, qDesc32.data);

        int tDescShape[] = {1, tDesc32.rows, tDesc32.cols};
        Mat tDescBlob(3, tDescShape, CV_32FC1, tDesc32.data);

        net_.setInput(qKptsBlob, params_.kpts0Name);
        net_.setInput(qDescBlob, params_.desc0Name);
        net_.setInput(tKptsBlob, params_.kpts1Name);
        net_.setInput(tDescBlob, params_.desc1Name);

        std::vector<String> outNames;
        outNames.push_back(params_.matches0Name);
        outNames.push_back(params_.mscores0Name);

        std::vector<Mat> outs;
        net_.forward(outs, outNames);

        CV_Assert(outs.size() >= 2);

        Mat matches0 = outs[0].reshape(1, static_cast<int>(outs[0].total()));
        Mat scores0 = outs[1].reshape(1, static_cast<int>(outs[1].total()));

        Mat matches0f, scores0f;
        matches0.convertTo(matches0f, CV_32F);
        scores0.convertTo(scores0f, CV_32F);

        matches.clear();
        matches.reserve(static_cast<size_t>(matches0f.total()));

        const int trainRows = tDesc32.rows;
        const int count = static_cast<int>(matches0f.total());
        for (int i = 0; i < count; ++i)
        {
            const float trainIdxF = matches0f.at<float>(i, 0);
            if (trainIdxF < -0.5f)
                continue;

            const int trainIdx = cvRound(trainIdxF);
            if (trainIdx < 0 || trainIdx >= trainRows)
                continue;

            const float score = i < static_cast<int>(scores0f.total()) ? scores0f.at<float>(i, 0) : 0.0f;
            const float distance = 1.0f - score;
            matches.push_back(DMatch(i, trainIdx, distance));
        }
    }

private:
    LightGlue::Params params_;
    mutable dnn::Net net_;
};

#else

class LightGlueImpl CV_FINAL : public LightGlue
{
public:
    explicit LightGlueImpl(const LightGlue::Params& params)
        : params_(params)
    {
    }

    void setModel(const String& modelPath) CV_OVERRIDE
    {
        params_.modelPath = modelPath;
    }

    String getModel() const CV_OVERRIDE
    {
        return params_.modelPath;
    }

    void match(InputArray,
               InputArray,
               std::vector<DMatch>&,
               InputArray) const CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, "LightGlue requires OpenCV built with DNN support");
    }

    void match(InputArray,
               InputArray,
               InputArray,
               InputArray,
               std::vector<DMatch>&,
               InputArray,
               Size,
               Size) const CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, "LightGlue requires OpenCV built with DNN support");
    }

private:
    LightGlue::Params params_;
};

#endif

} // namespace

Ptr<LightGlue> LightGlue::create(const LightGlue::Params& params)
{
    return makePtr<LightGlueImpl>(params);
}

} // namespace features
} // namespace cv
