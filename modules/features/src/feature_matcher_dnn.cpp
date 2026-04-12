// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "opencv2/features/feature_matcher.hpp"

#ifdef HAVE_OPENCV_DNN
#include "opencv2/dnn/dnn.hpp"
#endif

namespace cv {
namespace features {

LightGlue::Params::Params()
{
    modelPath = String();
#ifdef HAVE_OPENCV_DNN
    dnnEngine = dnn::ENGINE_AUTO;
#else
    dnnEngine = 3; // dnn::ENGINE_AUTO
#endif
    dnnBackend = -1;
    dnnTarget = -1;
    disableWinograd = true;
    scoreThreshold = 0.0f;
}

namespace {

#ifdef HAVE_OPENCV_DNN

static const char* const kKpts0Name = "kpts0";
static const char* const kDesc0Name = "desc0";
static const char* const kKpts1Name = "kpts1";
static const char* const kDesc1Name = "desc1";
static const char* const kMatches0Name = "matches0";
static const char* const kMScores0Name = "mscores0";

static Mat toKeypointMat(InputArray kpts)
{
    Mat m = kpts.getMat();
    if (m.empty())
        return Mat();

    Mat out;
    if (m.type() == CV_32FC2 || m.type() == CV_64FC2)
    {
        Mat reshaped = m.reshape(1, static_cast<int>(m.total()));
        if (reshaped.type() != CV_32F)
            reshaped.convertTo(out, CV_32F);
        else
            out = reshaped.isContinuous() ? reshaped : reshaped.clone();
        return out;
    }

    if (m.channels() == 1 && m.cols == 2)
    {
        if (m.type() != CV_32F)
            m.convertTo(out, CV_32F);
        else
            out = m.isContinuous() ? m : m.clone();
        return out;
    }

    if (m.channels() >= 2 && m.cols == 1)
    {
        Mat reshaped = m.reshape(1, m.rows);
        if (reshaped.cols >= 2)
        {
            out = reshaped.colRange(0, 2).clone();
            if (out.type() != CV_32F)
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
    CV_Assert(normalized.isContinuous());

    const float invWidth = 1.0f / static_cast<float>(imageSize.width - 1);
    const float invHeight = 1.0f / static_cast<float>(imageSize.height - 1);
    float* normalizedPtr = normalized.ptr<float>();

    for (int i = 0; i < normalized.rows; ++i)
    {
        const int idx = i * 2;
        normalizedPtr[idx] *= invWidth;
        normalizedPtr[idx + 1] *= invHeight;
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
        net_ = dnn::readNetFromONNX(modelPath, params_.dnnEngine);
        CV_Assert(!net_.empty());

        if (params_.disableWinograd)
            net_.enableWinograd(false);
        if (params_.dnnBackend >= 0)
            net_.setPreferableBackend(params_.dnnBackend);
        if (params_.dnnTarget >= 0)
            net_.setPreferableTarget(params_.dnnTarget);
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

        Mat qDesc32 = qDesc;
        if (qDesc32.type() != CV_32F)
            qDesc32.convertTo(qDesc32, CV_32F);
        else if (!qDesc32.isContinuous())
            qDesc32 = qDesc32.clone();

        Mat tDesc32 = tDesc;
        if (tDesc32.type() != CV_32F)
            tDesc32.convertTo(tDesc32, CV_32F);
        else if (!tDesc32.isContinuous())
            tDesc32 = tDesc32.clone();

        int qKptsShape[] = {1, qNorm.rows, 2};
        Mat qKptsBlob(3, qKptsShape, CV_32FC1, qNorm.data);

        int tKptsShape[] = {1, tNorm.rows, 2};
        Mat tKptsBlob(3, tKptsShape, CV_32FC1, tNorm.data);

        int qDescShape[] = {1, qDesc32.rows, qDesc32.cols};
        Mat qDescBlob(3, qDescShape, CV_32FC1, qDesc32.data);

        int tDescShape[] = {1, tDesc32.rows, tDesc32.cols};
        Mat tDescBlob(3, tDescShape, CV_32FC1, tDesc32.data);

        net_.setInput(qKptsBlob, kKpts0Name);
        net_.setInput(qDescBlob, kDesc0Name);
        net_.setInput(tKptsBlob, kKpts1Name);
        net_.setInput(tDescBlob, kDesc1Name);

        std::vector<String> outNames;
        outNames.push_back(kMatches0Name);
        outNames.push_back(kMScores0Name);

        std::vector<Mat> outs;
        net_.forward(outs, outNames);

        CV_Assert(outs.size() >= 2);

        Mat matches0 = outs[0];
        if (!matches0.isContinuous())
            matches0 = matches0.clone();
        matches0 = matches0.reshape(1, static_cast<int>(matches0.total()));
        if (matches0.type() != CV_32F)
            matches0.convertTo(matches0, CV_32F);

        Mat scores0 = outs[1];
        if (!scores0.isContinuous())
            scores0 = scores0.clone();
        scores0 = scores0.reshape(1, static_cast<int>(scores0.total()));
        if (scores0.type() != CV_32F)
            scores0.convertTo(scores0, CV_32F);

        matches.clear();
        matches.reserve(static_cast<size_t>(matches0.total()));

        CV_Assert(matches0.isContinuous());
        const float* matches0Ptr = matches0.ptr<float>();

        const int scoreCount = static_cast<int>(scores0.total());
        const float* scores0Ptr = scoreCount > 0 ? scores0.ptr<float>() : NULL;

        const int trainRows = tDesc32.rows;
        const int count = static_cast<int>(matches0.total());
        for (int i = 0; i < count; ++i)
        {
            const float trainIdxF = matches0Ptr[i];
            if (trainIdxF < -0.5f)
                continue;

            const int trainIdx = cvRound(trainIdxF);
            if (trainIdx < 0 || trainIdx >= trainRows)
                continue;

            const float score = (scores0Ptr != NULL && i < scoreCount) ? scores0Ptr[i] : 0.0f;
            if (score < params_.scoreThreshold)
                continue;
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

Ptr<LightGlue> LightGlue::create()
{
    return create(LightGlue::Params());
}

Ptr<LightGlue> LightGlue::create(const LightGlue::Params& params)
{
    return makePtr<LightGlueImpl>(params);
}

} // namespace features
} // namespace cv
