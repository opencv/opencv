// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#ifdef HAVE_OPENCV_DNN

#include <numeric>

namespace cv {

using namespace dnn;

namespace {

static const int kXFeatDescriptorSize = 64;

struct XFeatCandidate
{
    Point2f pt;
    float score;
    int x;
    int y;
};

static void validateInputSize(int inputSize)
{
    CV_Assert(inputSize > 0);
}

static Mat toGray(InputArray _image)
{
    Mat image = _image.getMat();
    if (image.channels() == 1)
        return image;

    Mat gray;
    if (image.channels() == 3)
        cvtColor(image, gray, COLOR_BGR2GRAY);
    else if (image.channels() == 4)
        cvtColor(image, gray, COLOR_BGRA2GRAY);
    else
        CV_Error(Error::StsBadArg, "XFeat expects a grayscale, BGR, or BGRA image");
    return gray;
}

static void normalizeDescriptorRow(Mat row)
{
    const double n = norm(row, NORM_L2);
    if (n > 0.0)
        row *= 1.0 / n;
}

} // namespace

class XFeat_Impl CV_FINAL : public XFeat
{
public:
    XFeat_Impl(const String& modelPath, int maxKeypoints, float scoreThreshold,
               int inputSize, int backendId, int targetId)
        : maxKeypoints_(maxKeypoints),
          scoreThreshold_(scoreThreshold),
          inputSize_(inputSize)
    {
        validateInputSize(inputSize_);
        initNet(readNetFromONNX(modelPath), backendId, targetId);
    }

    XFeat_Impl(const std::vector<uchar>& bufferModel, int maxKeypoints, float scoreThreshold,
               int inputSize, int backendId, int targetId)
        : maxKeypoints_(maxKeypoints),
          scoreThreshold_(scoreThreshold),
          inputSize_(inputSize)
    {
        validateInputSize(inputSize_);
        initNet(readNetFromONNX(bufferModel), backendId, targetId);
    }

    void detectAndCompute(InputArray _image, InputArray _mask,
                          std::vector<KeyPoint>& keypoints,
                          OutputArray _descriptors,
                          bool useProvidedKeypoints) CV_OVERRIDE
    {
        CV_Assert(!useProvidedKeypoints && "XFeat does not support providing keypoints externally");

        keypoints.clear();

        Mat image = _image.getMat();
        if (image.empty())
        {
            if (_descriptors.needed())
                _descriptors.release();
            return;
        }

        Mat mask = _mask.getMat();
        if (!mask.empty())
        {
            CV_Assert(mask.type() == CV_8UC1);
            CV_Assert(mask.size() == image.size());
        }

        Mat gray = toGray(image);
        const float scale = static_cast<float>(inputSize_) /
                            static_cast<float>(std::max(gray.cols, gray.rows));
        const int resizedW = std::max(1, static_cast<int>(gray.cols * scale));
        const int resizedH = std::max(1, static_cast<int>(gray.rows * scale));

        Mat resized;
        resize(gray, resized, Size(resizedW, resizedH));

        Mat padded = Mat::zeros(inputSize_, inputSize_, CV_8UC1);
        resized.copyTo(padded(Rect(0, 0, resizedW, resizedH)));

        Mat blob;
        blobFromImage(padded, blob, 1.0 / 255.0, Size(inputSize_, inputSize_),
                      Scalar(), false, false);
        net_.setInput(blob);

        std::vector<Mat> outs;
        net_.forward(outs, net_.getUnconnectedOutLayersNames());
        CV_Assert(outs.size() >= 3);

        Mat descMap = outs[0];
        Mat scoreMap = outs[2];
        CV_Assert(descMap.dims == 4);
        CV_Assert(scoreMap.dims == 4);
        CV_Assert(scoreMap.size[1] == 1);

        if (!descMap.isContinuous())
            descMap = descMap.clone();
        if (!scoreMap.isContinuous())
            scoreMap = scoreMap.clone();

        const int scoreH = scoreMap.size[2];
        const int scoreW = scoreMap.size[3];
        const bool descNCHW = descMap.size[1] == kXFeatDescriptorSize &&
                              descMap.size[2] == scoreH &&
                              descMap.size[3] == scoreW;
        const bool descNHWC = descMap.size[1] == scoreH &&
                              descMap.size[2] == scoreW &&
                              descMap.size[3] == kXFeatDescriptorSize;
        CV_Assert(descNCHW || descNHWC);
        const int descChannels = kXFeatDescriptorSize;
        const float strideX = static_cast<float>(inputSize_) / static_cast<float>(scoreW);
        const float strideY = static_cast<float>(inputSize_) / static_cast<float>(scoreH);

        const float* scores = scoreMap.ptr<float>();
        std::vector<XFeatCandidate> candidates;
        candidates.reserve(scoreH * scoreW);

        for (int y = 0; y < scoreH; ++y)
        {
            for (int x = 0; x < scoreW; ++x)
            {
                const float score = scores[y * scoreW + x];
                if (score <= scoreThreshold_)
                    continue;

                const float px = x * strideX / scale;
                const float py = y * strideY / scale;
                const int ix = cvFloor(px);
                const int iy = cvFloor(py);
                if (ix < 0 || iy < 0 || ix >= image.cols || iy >= image.rows)
                    continue;
                if (!mask.empty() && mask.at<uchar>(iy, ix) == 0)
                    continue;

                candidates.push_back({Point2f(px, py), score, x, y});
            }
        }

        if (maxKeypoints_ > 0 && static_cast<int>(candidates.size()) > maxKeypoints_)
        {
            std::partial_sort(candidates.begin(), candidates.begin() + maxKeypoints_, candidates.end(),
                [](const XFeatCandidate& a, const XFeatCandidate& b)
                {
                    return a.score > b.score;
                });
            candidates.resize(maxKeypoints_);
        }

        keypoints.reserve(candidates.size());
        for (const XFeatCandidate& c : candidates)
            keypoints.emplace_back(c.pt, 1.0f, -1.0f, c.score);

        if (_descriptors.needed())
        {
            if (candidates.empty())
            {
                _descriptors.release();
                return;
            }

            _descriptors.create(static_cast<int>(candidates.size()), descChannels, CV_32F);
            Mat descriptors = _descriptors.getMat();
            const float* descData = descMap.ptr<float>();

            for (int i = 0; i < static_cast<int>(candidates.size()); ++i)
            {
                float* dst = descriptors.ptr<float>(i);
                const XFeatCandidate& c = candidates[i];
                const int offset = c.y * scoreW + c.x;
                if (descNCHW)
                {
                    const int channelStep = scoreH * scoreW;
                    for (int ch = 0; ch < descChannels; ++ch)
                        dst[ch] = descData[ch * channelStep + offset];
                }
                else
                {
                    const int base = offset * descChannels;
                    for (int ch = 0; ch < descChannels; ++ch)
                        dst[ch] = descData[base + ch];
                }
                normalizeDescriptorRow(descriptors.row(i));
            }
        }
    }

    int descriptorSize() const CV_OVERRIDE { return kXFeatDescriptorSize; }
    int descriptorType() const CV_OVERRIDE { return CV_32F; }
    int defaultNorm()    const CV_OVERRIDE { return NORM_L2; }

    bool empty() const CV_OVERRIDE { return net_.empty(); }

    void setMaxKeypoints(int maxKeypoints) CV_OVERRIDE { maxKeypoints_ = maxKeypoints; }
    int  getMaxKeypoints() const CV_OVERRIDE { return maxKeypoints_; }

    void  setScoreThreshold(float threshold) CV_OVERRIDE { scoreThreshold_ = threshold; }
    float getScoreThreshold() const CV_OVERRIDE { return scoreThreshold_; }

    void setInputSize(int inputSize) CV_OVERRIDE
    {
        validateInputSize(inputSize);
        inputSize_ = inputSize;
    }
    int getInputSize() const CV_OVERRIDE { return inputSize_; }

    String getDefaultName() const CV_OVERRIDE { return Feature2D::getDefaultName() + ".XFeat"; }

private:
    void initNet(const Net& net, int backendId, int targetId)
    {
        net_ = net;
        net_.setPreferableBackend(backendId);
        net_.setPreferableTarget(targetId);
    }

    int maxKeypoints_;
    float scoreThreshold_;
    int inputSize_;
    Net net_;
};

Ptr<XFeat> XFeat::create(const String& modelPath, int maxKeypoints, float scoreThreshold,
                         int inputSize, int backendId, int targetId)
{
    CV_TRACE_FUNCTION();
    return makePtr<XFeat_Impl>(modelPath, maxKeypoints, scoreThreshold,
                               inputSize, backendId, targetId);
}

Ptr<XFeat> XFeat::create(const std::vector<uchar>& bufferModel, int maxKeypoints,
                         float scoreThreshold, int inputSize, int backendId, int targetId)
{
    CV_TRACE_FUNCTION();
    return makePtr<XFeat_Impl>(bufferModel, maxKeypoints, scoreThreshold,
                               inputSize, backendId, targetId);
}

String XFeat::getDefaultName() const
{
    return Feature2D::getDefaultName() + ".XFeat";
}

} // namespace cv

#endif // HAVE_OPENCV_DNN
