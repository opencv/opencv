// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#ifdef HAVE_OPENCV_DNN
#include "opencv2/dnn.hpp"
#endif

namespace cv
{

LightGlueMatcher::LightGlueMatcher() {}
LightGlueMatcher::~LightGlueMatcher() {}

void LightGlueMatcher::setImagePairInfo(const std::vector<KeyPoint>& queryKpts, const std::vector<KeyPoint>& trainKpts,
                                        Size queryImageSize, Size trainImageSize)
{
    Mat qk((int)queryKpts.size(), 2, CV_32F);
    for (size_t i = 0; i < queryKpts.size(); ++i)
    {
        qk.at<float>((int)i, 0) = queryKpts[i].pt.x;
        qk.at<float>((int)i, 1) = queryKpts[i].pt.y;
    }

    Mat tk((int)trainKpts.size(), 2, CV_32F);
    for (size_t i = 0; i < trainKpts.size(); ++i)
    {
        tk.at<float>((int)i, 0) = trainKpts[i].pt.x;
        tk.at<float>((int)i, 1) = trainKpts[i].pt.y;
    }

    setPairInfo(qk, tk, queryImageSize, trainImageSize);
}

void normalizeLightGlueKeypoints(InputArray _keypoints, OutputArray _normalizedKeypoints,
                                 Size imageSize, int normalizationType)
{
    Mat keypoints = _keypoints.getMat();
    CV_CheckTypeEQ(keypoints.type(), CV_32F, "LightGlue keypoints must be an Nx2 CV_32F matrix");
    CV_CheckEQ(keypoints.cols, 2, "LightGlue keypoints must be an Nx2 CV_32F matrix");

    if (normalizationType == LG_KEYPOINTS_AS_IS)
    {
        keypoints.copyTo(_normalizedKeypoints);
        return;
    }

    if (normalizationType == LG_KEYPOINTS_AUTO)
        CV_Error(Error::StsBadArg,
                 "LG_KEYPOINTS_AUTO can only be resolved by LightGlueMatcher");

    CV_Check(imageSize.width, imageSize.width > 0 && imageSize.height > 0,
             "LightGlue keypoint normalization requires a valid image size");

    Mat normalized = keypoints.clone();
    if (normalizationType == LG_KEYPOINTS_ALIKED)
    {
        for (int i = 0; i < normalized.rows; i++)
        {
            normalized.at<float>(i, 0) = normalized.at<float>(i, 0) / (float)imageSize.width  * 2.0f - 1.0f;
            normalized.at<float>(i, 1) = normalized.at<float>(i, 1) / (float)imageSize.height * 2.0f - 1.0f;
        }
    }
    else if (normalizationType == LG_KEYPOINTS_DISK)
    {
        CV_Check(imageSize.width, imageSize.width > 1 && imageSize.height > 1,
                 "DISK LightGlue: image dimensions must be >= 2 for [0,1] normalization");
        float wNorm = 1.0f / (float)(imageSize.width  - 1);
        float hNorm = 1.0f / (float)(imageSize.height - 1);
        for (int i = 0; i < normalized.rows; i++)
        {
            normalized.at<float>(i, 0) *= wNorm;
            normalized.at<float>(i, 1) *= hNorm;
        }
    }
    else
    {
        CV_Error(Error::StsBadArg, "Unsupported LightGlue keypoint normalization type");
    }

    normalized.copyTo(_normalizedKeypoints);
}

#ifdef HAVE_OPENCV_DNN

static int resolveKeypointNormalization(int normalizationType, int autoNormalizationType)
{
    return normalizationType == LG_KEYPOINTS_AUTO ? autoNormalizationType : normalizationType;
}

static void normalizeMatcherKeypoints(InputArray keypoints, OutputArray normalizedKeypoints,
                                      Size imageSize, int normalizationType,
                                      int autoNormalizationType)
{
    int resolvedType = resolveKeypointNormalization(normalizationType, autoNormalizationType);
    if (normalizationType == LG_KEYPOINTS_AUTO && (imageSize.width <= 0 || imageSize.height <= 0))
        resolvedType = LG_KEYPOINTS_AS_IS;
    normalizeLightGlueKeypoints(keypoints, normalizedKeypoints, imageSize, resolvedType);
}

struct LightGluePairContext
{
    Mat queryKeypoints;   // Nx2 float (normalized or pixel)
    Mat trainKeypoints;   // Mx2 float
    Size queryImageSize;
    Size trainImageSize;
    int keypointNormalization;
    bool valid;

    LightGluePairContext() : keypointNormalization(LG_KEYPOINTS_AUTO), valid(false) {}

    void clear()
    {
        queryKeypoints.release();
        trainKeypoints.release();
        queryImageSize = Size();
        trainImageSize = Size();
        keypointNormalization = LG_KEYPOINTS_AUTO;
        valid = false;
    }
};

class LightGlueMatcherImpl : public LightGlueMatcher
{
public:
    LightGlueMatcherImpl(const String& modelPath, float _scoreThreshold, int backend, int target)
    {
        scoreThreshold = _scoreThreshold;
        net = dnn::readNet(modelPath, "", "");
        CV_Assert(!net.empty());
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);
    }

    LightGlueMatcherImpl(const std::vector<uchar>& modelData, float _scoreThreshold, int backend, int target)
    {
        scoreThreshold = _scoreThreshold;
        net = dnn::readNetFromONNX(modelData);
        CV_Assert(!net.empty());
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);
    }

    // Private constructor for clone() — shares the already-loaded network
    LightGlueMatcherImpl(const dnn::Net& _net, float _scoreThreshold)
        : net(_net), scoreThreshold(_scoreThreshold) {}

    // DescriptorMatcher interface
    bool isMaskSupported() const CV_OVERRIDE { return false; }

    // LightGlueMatcher interface
    void setPairInfo(InputArray queryKpts, InputArray trainKpts,
                     Size queryImageSize = Size(), Size trainImageSize = Size(),
                     int keypointNormalization = LG_KEYPOINTS_AUTO) CV_OVERRIDE;
    void clearPairInfo() CV_OVERRIDE;

protected:
    void knnMatchImpl(InputArray queryDescriptors,
                      std::vector<std::vector<DMatch>>& matches, int k,
                      InputArrayOfArrays masks = noArray(),
                      bool compactResult = false) CV_OVERRIDE;
    void radiusMatchImpl(InputArray queryDescriptors,
                         std::vector<std::vector<DMatch>>& matches, float maxDistance,
                         InputArrayOfArrays masks = noArray(),
                         bool compactResult = false) CV_OVERRIDE;

    virtual void lightglueMatch(const Mat& queryDesc, const Mat& trainDesc,
                                const Mat& queryKpts, const Mat& trainKpts,
                                Size queryImgSize, Size trainImgSize,
                                int keypointNormalization,
                                std::vector<DMatch>& matches) = 0;

    bool resolveContext(Mat& queryKpts, Mat& trainKpts,
                        Size& queryImgSize, Size& trainImgSize,
                        int& keypointNormalization);

    dnn::Net net;
    float scoreThreshold;
    LightGluePairContext pairContext;
};

void LightGlueMatcherImpl::setPairInfo(InputArray _queryKpts, InputArray _trainKpts,
                                        Size _queryImageSize, Size _trainImageSize,
                                        int _keypointNormalization)
{
    pairContext.queryKeypoints = _queryKpts.getMat().clone();
    pairContext.trainKeypoints = _trainKpts.getMat().clone();
    pairContext.queryImageSize = _queryImageSize;
    pairContext.trainImageSize = _trainImageSize;
    pairContext.keypointNormalization = _keypointNormalization;
    pairContext.valid = true;
}

void LightGlueMatcherImpl::clearPairInfo()
{
    pairContext.clear();
}

bool LightGlueMatcherImpl::resolveContext(Mat& queryKpts, Mat& trainKpts,
                                           Size& queryImgSize, Size& trainImgSize,
                                           int& keypointNormalization)
{
    if (pairContext.valid)
    {
        queryKpts = pairContext.queryKeypoints;
        trainKpts = pairContext.trainKeypoints;
        queryImgSize = pairContext.queryImageSize;
        trainImgSize = pairContext.trainImageSize;
        keypointNormalization = pairContext.keypointNormalization;
        return true;
    }
    return false;
}

void LightGlueMatcherImpl::knnMatchImpl(InputArray _queryDescriptors,
                                          std::vector<std::vector<DMatch>>& matches,
                                          int k, InputArrayOfArrays, bool)
{
    CV_INSTRUMENT_REGION();

    if (k != 1)
        CV_Error(cv::Error::StsBadArg, "LightGlueMatcher only supports k=1");

    Mat queryKpts, trainKpts;
    Size queryImgSize, trainImgSize;
    int keypointNormalization;
    if (!resolveContext(queryKpts, trainKpts, queryImgSize, trainImgSize, keypointNormalization))
    {
        CV_Error(cv::Error::StsBadArg,
                 "LightGlueMatcher: no valid context. Call setPairInfo() before matching.");
    }

    CV_Assert(!trainDescCollection.empty());
    const Mat& trainDesc = trainDescCollection[0];
    Mat queryDesc = _queryDescriptors.getMat();

    std::vector<DMatch> flatMatches;
    lightglueMatch(queryDesc, trainDesc, queryKpts, trainKpts,
                   queryImgSize, trainImgSize, keypointNormalization, flatMatches);

    matches.clear();
    matches.resize(queryDesc.rows);
    for (const auto& m : flatMatches)
    {
        matches[m.queryIdx].push_back(m);
    }

    clearPairInfo();
}

void LightGlueMatcherImpl::radiusMatchImpl(InputArray, std::vector<std::vector<DMatch>>&,
                                             float, InputArrayOfArrays, bool)
{
    CV_Error(cv::Error::StsNotImplemented,
             "radiusMatch is not supported by LightGlueMatcher. Use match() or knnMatch().");
}

// ==================== ALIKED variant ====================

class ALIKEDLightGlueMatcherImpl CV_FINAL : public LightGlueMatcherImpl
{
public:
    using LightGlueMatcherImpl::LightGlueMatcherImpl;
    Ptr<DescriptorMatcher> clone(bool emptyTrainData) const CV_OVERRIDE;

protected:
    void lightglueMatch(const Mat& queryDesc, const Mat& trainDesc,
                        const Mat& queryKpts, const Mat& trainKpts,
                        Size queryImgSize, Size trainImgSize,
                        int keypointNormalization,
                        std::vector<DMatch>& matches) CV_OVERRIDE;
};

Ptr<DescriptorMatcher> ALIKEDLightGlueMatcherImpl::clone(bool emptyTrainData) const
{
    Ptr<ALIKEDLightGlueMatcherImpl> matcher = makePtr<ALIKEDLightGlueMatcherImpl>(net, scoreThreshold);
    // Always copy pairContext - it's matcher state, not train data
    matcher->pairContext = pairContext;
    if (!emptyTrainData)
    {
        matcher->trainDescCollection = trainDescCollection;
        matcher->utrainDescCollection = utrainDescCollection;
    }
    return matcher;
}

void ALIKEDLightGlueMatcherImpl::lightglueMatch(const Mat& queryDesc, const Mat& trainDesc,
                                                  const Mat& queryKpts, const Mat& trainKpts,
                                                  Size queryImgSize, Size trainImgSize,
                                                  int keypointNormalization,
                                                  std::vector<DMatch>& matches)
{
    int N = queryDesc.rows;
    int M = trainDesc.rows;

    Mat kpts0, kpts1;
    normalizeMatcherKeypoints(queryKpts, kpts0, queryImgSize, keypointNormalization, LG_KEYPOINTS_ALIKED);
    normalizeMatcherKeypoints(trainKpts, kpts1, trainImgSize, keypointNormalization, LG_KEYPOINTS_ALIKED);

    // Prepare blobs: [1, N, 2] and [1, N, D]
    int descDim = queryDesc.cols;
    int szK0[] = {1, N, 2};
    int szK1[] = {1, M, 2};
    int szD0[] = {1, N, descDim};
    int szD1[] = {1, M, descDim};
    Mat kpts0blob = kpts0.reshape(0, 3, szK0);
    Mat kpts1blob = kpts1.reshape(0, 3, szK1);
    Mat desc0blob = queryDesc.reshape(0, 3, szD0);
    Mat desc1blob = trainDesc.reshape(0, 3, szD1);

    net.setInput(kpts0blob, "kpts0");
    net.setInput(kpts1blob, "kpts1");
    net.setInput(desc0blob, "desc0");
    net.setInput(desc1blob, "desc1");

    std::vector<String> outNames = {"matches0", "mscores0"};
    std::vector<Mat> outs;
    net.forward(outs, outNames);

    CV_Assert(outs.size() == 2);

    // matches0: [M, 2] int64 - pair indices (kpt0_idx, kpt1_idx)
    // mscores0: [M] float32 - confidence per pair
    Mat matchesMat = outs[0];
    Mat scoresMat = outs[1];

    matches.clear();
    int nMatches = matchesMat.rows;
    matches.reserve(nMatches);

    for (int i = 0; i < nMatches; i++)
    {
        int qIdx = (int)matchesMat.at<int64_t>(i, 0);
        int tIdx = (int)matchesMat.at<int64_t>(i, 1);
        if (qIdx >= 0 && tIdx >= 0 && qIdx < N && tIdx < M)
        {
            float score = scoresMat.at<float>(i);
            if (score >= scoreThreshold)
            {
                matches.push_back(DMatch(qIdx, tIdx, 1.0f - score));
            }
        }
    }
}

// ==================== DISK variant ====================

class DISKLightGlueMatcherImpl CV_FINAL : public LightGlueMatcherImpl
{
public:
    using LightGlueMatcherImpl::LightGlueMatcherImpl;
    Ptr<DescriptorMatcher> clone(bool emptyTrainData) const CV_OVERRIDE;

protected:
    void lightglueMatch(const Mat& queryDesc, const Mat& trainDesc,
                        const Mat& queryKpts, const Mat& trainKpts,
                        Size queryImgSize, Size trainImgSize,
                        int keypointNormalization,
                        std::vector<DMatch>& matches) CV_OVERRIDE;
};

Ptr<DescriptorMatcher> DISKLightGlueMatcherImpl::clone(bool emptyTrainData) const
{
    Ptr<DISKLightGlueMatcherImpl> matcher = makePtr<DISKLightGlueMatcherImpl>(net, scoreThreshold);
    // Always copy pairContext - it's matcher state, not train data
    matcher->pairContext = pairContext;
    if (!emptyTrainData)
    {
        matcher->trainDescCollection = trainDescCollection;
        matcher->utrainDescCollection = utrainDescCollection;
    }
    return matcher;
}

void DISKLightGlueMatcherImpl::lightglueMatch(const Mat& queryDesc, const Mat& trainDesc,
                                                const Mat& queryKpts, const Mat& trainKpts,
                                                Size queryImgSize, Size trainImgSize,
                                                int keypointNormalization,
                                                std::vector<DMatch>& matches)
{
    int N = queryDesc.rows;
    int M = trainDesc.rows;

    Mat kpts0, kpts1;
    normalizeMatcherKeypoints(queryKpts, kpts0, queryImgSize, keypointNormalization, LG_KEYPOINTS_DISK);
    normalizeMatcherKeypoints(trainKpts, kpts1, trainImgSize, keypointNormalization, LG_KEYPOINTS_DISK);

    // Prepare blobs: [1, N, 2] and [1, N, D]
    int descDim = queryDesc.cols;
    int szK0[] = {1, N, 2};
    int szK1[] = {1, M, 2};
    int szD0[] = {1, N, descDim};
    int szD1[] = {1, M, descDim};
    Mat kpts0blob = kpts0.reshape(0, 3, szK0);
    Mat kpts1blob = kpts1.reshape(0, 3, szK1);
    Mat desc0blob = queryDesc.reshape(0, 3, szD0);
    Mat desc1blob = trainDesc.reshape(0, 3, szD1);

    net.setInput(kpts0blob, "kpts0");
    net.setInput(kpts1blob, "kpts1");
    net.setInput(desc0blob, "desc0");
    net.setInput(desc1blob, "desc1");

    // DISK LightGlue has 4 outputs (bidirectional matches + scores)
    std::vector<String> outNames = {"matches0", "matches1", "mscores0", "mscores1"};
    std::vector<Mat> outs;
    net.forward(outs, outNames);

    CV_Assert(outs.size() == 4);

    // DISK LightGlue outputs:
    //   matches0: [1, N] int64 — for each query kpt i, matched train kpt j (or -1)
    //   matches1: [1, M] int64 — for each train kpt j, matched query kpt i (or -1)
    //   mscores0: [1, N] float  — confidence per query kpt
    //   mscores1: [1, M] float  — confidence per train kpt
    //
    // ORT engine may drop the batch dim, producing [N] / [M] instead of [1, N] / [1, M].
    Mat matches0 = outs[0];  // matches0
    Mat mscores0 = outs[2];  // mscores0

    // Flatten to 1D in case ORT dropped the batch dimension
    matches0 = matches0.reshape(1, (int)matches0.total());
    mscores0 = mscores0.reshape(1, (int)mscores0.total());

    CV_Assert(matches0.total() == (size_t)N);
    CV_Assert(mscores0.total() == (size_t)N);

    matches.clear();
    matches.reserve(N);

    for (int i = 0; i < N; i++)
    {
        int64_t j = matches0.at<int64_t>(i);
        if (j >= 0 && j < M)
        {
            float score = mscores0.at<float>(i);
            if (score >= scoreThreshold)
            {
                matches.push_back(DMatch(i, (int)j, 1.0f - score));
            }
        }
    }
}

Ptr<LightGlueMatcher> LightGlueMatcher::create(const String& modelPath,
                                                 float scoreThreshold, int backend, int target,
                                                 int type)
{
    if (type == LG_DISK)
        return makePtr<DISKLightGlueMatcherImpl>(modelPath, scoreThreshold, backend, target);
    else
        return makePtr<ALIKEDLightGlueMatcherImpl>(modelPath, scoreThreshold, backend, target);
}

Ptr<LightGlueMatcher> LightGlueMatcher::create(const std::vector<uchar>& modelData,
                                                 float scoreThreshold, int backend, int target,
                                                 int type)
{
    if (type == LG_DISK)
        return makePtr<DISKLightGlueMatcherImpl>(modelData, scoreThreshold, backend, target);
    else
        return makePtr<ALIKEDLightGlueMatcherImpl>(modelData, scoreThreshold, backend, target);
}

#else  // !HAVE_OPENCV_DNN

Ptr<LightGlueMatcher> LightGlueMatcher::create(const String& modelPath,
                                                 float scoreThreshold, int backend, int target,
                                                 int type)
{
    CV_UNUSED(modelPath);
    CV_UNUSED(scoreThreshold);
    CV_UNUSED(backend);
    CV_UNUSED(target);
    CV_UNUSED(type);
    CV_Error(cv::Error::StsNotImplemented,
             "LightGlueMatcher requires OpenCV built with opencv_dnn module!");
}

#endif  // HAVE_OPENCV_DNN

}  // namespace cv
