// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#ifdef HAVE_OPENCV_DNN
#include "opencv2/dnn.hpp"
#include "aliked_context.hpp"
#endif

namespace cv
{

LightGlueMatcher::LightGlueMatcher() {}
LightGlueMatcher::~LightGlueMatcher() {}

#ifdef HAVE_OPENCV_DNN

struct LightGluePairContext
{
    Mat queryKeypoints;   // Nx2 float (normalized [-1,1] or pixel)
    Mat trainKeypoints;   // Mx2 float
    Size queryImageSize;
    Size trainImageSize;
    bool valid;

    LightGluePairContext() : valid(false) {}

    void clear()
    {
        queryKeypoints.release();
        trainKeypoints.release();
        queryImageSize = Size();
        trainImageSize = Size();
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
        : net(_net), scoreThreshold(_scoreThreshold) {};

    // DescriptorMatcher interface
    bool isMaskSupported() const CV_OVERRIDE { return false; }
    Ptr<DescriptorMatcher> clone(bool emptyTrainData) const CV_OVERRIDE;

    // LightGlueMatcher interface
    void setPairInfo(InputArray queryKpts, InputArray trainKpts,
                     Size queryImageSize = Size(), Size trainImageSize = Size()) CV_OVERRIDE;
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

    void lightglueMatch(const Mat& queryDesc, const Mat& trainDesc,
                        const Mat& queryKpts, const Mat& trainKpts,
                        Size queryImgSize, Size trainImgSize,
                        std::vector<DMatch>& matches);

    bool resolveContext(Mat& queryKpts, Mat& trainKpts,
                        Size& queryImgSize, Size& trainImgSize);

    dnn::Net net;
    float scoreThreshold;
    LightGluePairContext pairContext;
};

Ptr<DescriptorMatcher> LightGlueMatcherImpl::clone(bool emptyTrainData) const
{
    Ptr<LightGlueMatcherImpl> m = makePtr<LightGlueMatcherImpl>(net, scoreThreshold);
    // Always copy pairContext - it's matcher state, not train data
    m->pairContext = pairContext;
    if (!emptyTrainData)
    {
        m->trainDescCollection = trainDescCollection;
        m->utrainDescCollection = utrainDescCollection;
    }
    return m;
}

void LightGlueMatcherImpl::setPairInfo(InputArray _queryKpts, InputArray _trainKpts,
                                        Size _queryImageSize, Size _trainImageSize)
{
    pairContext.queryKeypoints = _queryKpts.getMat().clone();
    pairContext.trainKeypoints = _trainKpts.getMat().clone();
    pairContext.queryImageSize = _queryImageSize;
    pairContext.trainImageSize = _trainImageSize;
    pairContext.valid = true;
}

void LightGlueMatcherImpl::clearPairInfo()
{
    pairContext.clear();
}

bool LightGlueMatcherImpl::resolveContext(Mat& queryKpts, Mat& trainKpts,
                                           Size& queryImgSize, Size& trainImgSize)
{
    if (pairContext.valid)
    {
        queryKpts = pairContext.queryKeypoints;
        trainKpts = pairContext.trainKeypoints;
        queryImgSize = pairContext.queryImageSize;
        trainImgSize = pairContext.trainImageSize;
        return true;
    }
    return false;
}

void LightGlueMatcherImpl::lightglueMatch(const Mat& queryDesc, const Mat& trainDesc,
                                            const Mat& queryKpts, const Mat& trainKpts,
                                            Size queryImgSize, Size trainImgSize,
                                            std::vector<DMatch>& matches)
{
    int N = queryDesc.rows;
    int M = trainDesc.rows;

    // Normalize keypoints to [-1, 1] if in pixel coordinates
    Mat kpts0 = queryKpts.clone();
    Mat kpts1 = trainKpts.clone();

    if (queryImgSize.width > 0 && queryImgSize.height > 0)
    {
        for (int i = 0; i < kpts0.rows; i++)
        {
            kpts0.at<float>(i, 0) = kpts0.at<float>(i, 0) / (float)queryImgSize.width * 2.0f - 1.0f;
            kpts0.at<float>(i, 1) = kpts0.at<float>(i, 1) / (float)queryImgSize.height * 2.0f - 1.0f;
        }
    }
    if (trainImgSize.width > 0 && trainImgSize.height > 0)
    {
        for (int i = 0; i < kpts1.rows; i++)
        {
            kpts1.at<float>(i, 0) = kpts1.at<float>(i, 0) / (float)trainImgSize.width * 2.0f - 1.0f;
            kpts1.at<float>(i, 1) = kpts1.at<float>(i, 1) / (float)trainImgSize.height * 2.0f - 1.0f;
        }
    }

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

void LightGlueMatcherImpl::knnMatchImpl(InputArray _queryDescriptors,
                                          std::vector<std::vector<DMatch>>& matches,
                                          int k, InputArrayOfArrays, bool)
{
    CV_INSTRUMENT_REGION();

    if (k != 1)
        CV_Error(cv::Error::StsBadArg, "LightGlueMatcher only supports k=1");

    Mat queryKpts, trainKpts;
    Size queryImgSize, trainImgSize;
    if (!resolveContext(queryKpts, trainKpts, queryImgSize, trainImgSize))
    {
        CV_Error(cv::Error::StsBadArg,
                 "LightGlueMatcher: no valid context. Call setPairInfo() before matching.");
    }

    CV_Assert(!trainDescCollection.empty());
    const Mat& trainDesc = trainDescCollection[0];
    Mat queryDesc = _queryDescriptors.getMat();

    std::vector<DMatch> flatMatches;
    lightglueMatch(queryDesc, trainDesc, queryKpts, trainKpts,
                   queryImgSize, trainImgSize, flatMatches);

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

Ptr<LightGlueMatcher> LightGlueMatcher::create(const String& modelPath, float scoreThreshold, int backend, int target)
{
    return makePtr<LightGlueMatcherImpl>(modelPath, scoreThreshold, backend, target);
}

Ptr<LightGlueMatcher> LightGlueMatcher::create(const std::vector<uchar>& modelData,
                                               float scoreThreshold, int backend, int target)
{
    return makePtr<LightGlueMatcherImpl>(modelData, scoreThreshold, backend, target);
}

#else  // !HAVE_OPENCV_DNN

Ptr<LightGlueMatcher> LightGlueMatcher::create(const String& modelPath,
                                                 float scoreThreshold, int backend, int target)
{
    CV_UNUSED(modelPath);
    CV_UNUSED(scoreThreshold);
    CV_UNUSED(backend);
    CV_UNUSED(target);
    CV_Error(cv::Error::StsNotImplemented,
             "LightGlueMatcher requires OpenCV built with opencv_dnn module!");
}

#endif  // HAVE_OPENCV_DNN

}  // namespace cv
