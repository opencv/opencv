// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#ifdef HAVE_OPENCV_DNN

#include <cfloat>
#include <cmath>
#include <numeric>

namespace cv {

using namespace dnn;

namespace {

static const int kXFeatDescriptorSize = 64;

struct XFeatCandidate
{
    Point2f ptPadded;
    Point2f pt;
    float score;
};

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

static Mat toNCHW(const Mat& blob, int channelsHint)
{
    CV_Assert(blob.dims == 4);
    if (blob.size[1] == channelsHint)
        return blob;

    CV_Assert(blob.size[3] == channelsHint);
    Mat out;
    Mat src = blob.isContinuous() ? blob : blob.clone();
    transposeND(src, {0, 3, 1, 2}, out);
    return out;
}

static float sampleNearest(const Mat& map, float x, float y, int normW, int normH)
{
    CV_Assert(map.type() == CV_32F);
    if (map.empty() || normW <= 1 || normH <= 1)
        return 0.0f;

    const float fx = x * static_cast<float>(map.cols - 1) / static_cast<float>(normW - 1);
    const float fy = y * static_cast<float>(map.rows - 1) / static_cast<float>(normH - 1);
    const int ix = std::max(0, std::min(map.cols - 1, cvRound(fx)));
    const int iy = std::max(0, std::min(map.rows - 1, cvRound(fy)));
    return map.at<float>(iy, ix);
}

static float sampleBilinear(const Mat& map, float x, float y, int normW, int normH)
{
    CV_Assert(map.type() == CV_32F);
    if (map.empty() || normW <= 1 || normH <= 1)
        return 0.0f;

    float fx = x * static_cast<float>(map.cols - 1) / static_cast<float>(normW - 1);
    float fy = y * static_cast<float>(map.rows - 1) / static_cast<float>(normH - 1);
    fx = std::max(0.0f, std::min(fx, static_cast<float>(map.cols - 1)));
    fy = std::max(0.0f, std::min(fy, static_cast<float>(map.rows - 1)));

    const int x0 = cvFloor(fx);
    const int y0 = cvFloor(fy);
    const int x1 = std::min(x0 + 1, map.cols - 1);
    const int y1 = std::min(y0 + 1, map.rows - 1);
    const float dx = fx - x0;
    const float dy = fy - y0;

    const float v00 = map.at<float>(y0, x0);
    const float v01 = map.at<float>(y0, x1);
    const float v10 = map.at<float>(y1, x0);
    const float v11 = map.at<float>(y1, x1);
    return (1.f - dx) * (1.f - dy) * v00 +
           dx * (1.f - dy) * v01 +
           (1.f - dx) * dy * v10 +
           dx * dy * v11;
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
        CV_Assert(inputSize_ > 0);
        initNet(readNetFromONNX(modelPath), backendId, targetId);
    }

    XFeat_Impl(const std::vector<uchar>& bufferModel, int maxKeypoints, float scoreThreshold,
               int inputSize, int backendId, int targetId)
        : maxKeypoints_(maxKeypoints),
          scoreThreshold_(scoreThreshold),
          inputSize_(inputSize)
    {
        CV_Assert(inputSize_ > 0);
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

        const std::vector<String> outNames = {"output_feats", "output_keypoints", "output_heatmap"};
        std::vector<Mat> outs;
        net_.forward(outs, outNames);
        CV_Assert(outs.size() == 3);

        Mat featBlob = toNCHW(outs[0], kXFeatDescriptorSize);
        Mat kptBlob = toNCHW(outs[1], 65);
        Mat relBlob = toNCHW(outs[2], 1);
        CV_Assert(featBlob.dims == 4 && kptBlob.dims == 4 && relBlob.dims == 4);
        if (!featBlob.isContinuous())
            featBlob = featBlob.clone();
        if (!kptBlob.isContinuous())
            kptBlob = kptBlob.clone();
        if (!relBlob.isContinuous())
            relBlob = relBlob.clone();

        const int featH = featBlob.size[2];
        const int featW = featBlob.size[3];
        const int kptC = kptBlob.size[1];
        const int kptH = kptBlob.size[2];
        const int kptW = kptBlob.size[3];
        CV_Assert(featBlob.size[1] == kXFeatDescriptorSize && kptC >= 64);

        Mat reliability(relBlob.size[2], relBlob.size[3], CV_32F, relBlob.ptr<float>());
        Mat heatmap = Mat::zeros(kptH * 8, kptW * 8, CV_32F);
        const float* kptPtr = kptBlob.ptr<float>();
        const int kptHW = kptH * kptW;

        parallel_for_(Range(0, kptH), [&](const Range& range)
        {
            for (int y = range.start; y < range.end; ++y)
            {
                for (int x = 0; x < kptW; ++x)
                {
                    const int offset = y * kptW + x;
                    float maxLogit = -FLT_MAX;
                    for (int ch = 0; ch < kptC; ++ch)
                        maxLogit = std::max(maxLogit, kptPtr[ch * kptHW + offset]);

                    float sumExp = 0.f;
                    float probs[64];
                    for (int ch = 0; ch < 64; ++ch)
                    {
                        probs[ch] = std::exp(kptPtr[ch * kptHW + offset] - maxLogit);
                        sumExp += probs[ch];
                    }
                    if (kptC > 64)
                        sumExp += std::exp(kptPtr[64 * kptHW + offset] - maxLogit);
                    if (sumExp <= 0.f)
                        continue;

                    for (int ch = 0; ch < 64; ++ch)
                    {
                        const int dy = ch / 8;
                        const int dx = ch % 8;
                        heatmap.at<float>(y * 8 + dy, x * 8 + dx) = probs[ch] / sumExp;
                    }
                }
            }
        });

        Mat localMax;
        dilate(heatmap, localMax, getStructuringElement(MORPH_RECT, Size(5, 5)));

        std::vector<XFeatCandidate> candidates;
        candidates.reserve(4096);

        for (int y = 0; y < heatmap.rows; ++y)
        {
            const float* hm = heatmap.ptr<float>(y);
            const float* mx = localMax.ptr<float>(y);
            for (int x = 0; x < heatmap.cols; ++x)
            {
                const float h = hm[x];
                if (h <= scoreThreshold_ || h != mx[x])
                    continue;

                const float xp = static_cast<float>(x);
                const float yp = static_cast<float>(y);
                const float score = sampleNearest(heatmap, xp, yp, inputSize_, inputSize_) *
                                    sampleBilinear(reliability, xp, yp, inputSize_, inputSize_);
                if (score <= 0.f)
                    continue;

                const float px = xp / scale;
                const float py = yp / scale;
                const int ix = cvFloor(px);
                const int iy = cvFloor(py);
                if (ix < 0 || iy < 0 || ix >= image.cols || iy >= image.rows)
                    continue;
                if (!mask.empty() && mask.at<uchar>(iy, ix) == 0)
                    continue;

                candidates.push_back({Point2f(xp, yp), Point2f(px, py), score});
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

            _descriptors.create(static_cast<int>(candidates.size()), kXFeatDescriptorSize, CV_32F);
            Mat descriptors = _descriptors.getMat();
            const float* featPtr = featBlob.ptr<float>();
            const int featHW = featH * featW;

            parallel_for_(Range(0, static_cast<int>(candidates.size())),
                          [&](const Range& range)
            {
                for (int i = range.start; i < range.end; ++i)
                {
                    float* dst = descriptors.ptr<float>(i);
                    const XFeatCandidate& c = candidates[i];
                    for (int ch = 0; ch < kXFeatDescriptorSize; ++ch)
                    {
                        Mat channel(featH, featW, CV_32F,
                                    const_cast<float*>(featPtr + ch * featHW));
                        dst[ch] = sampleBilinear(channel, c.ptPadded.x, c.ptPadded.y,
                                                 inputSize_, inputSize_);
                    }
                    normalize(descriptors.row(i), descriptors.row(i), 1.0, 0.0, NORM_L2);
                }
            });
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
        CV_Assert(inputSize > 0);
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
