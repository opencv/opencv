// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "opencv2/features.hpp"

#ifdef HAVE_OPENCV_DNN
#include "opencv2/dnn.hpp"

#include <algorithm>
#include <numeric>

namespace cv {

using namespace dnn;

// Default network input size used when the user does not specify one explicitly.
// Matches the fixed-shape standalone DISK ONNX export shipped in opencv_extra.
static const Size kDefaultDiskInputSize = Size(1024, 1024);

// DISK is a fully convolutional network with a 16x downsampling stride, so user-provided
// input sizes must be positive multiples of 16.
static const int kDiskStride = 16;

class DISK_Impl CV_FINAL : public DISK
{
public:
    DISK_Impl(const String& modelPath, int maxKeypoints, float scoreThreshold,
              const Size& imageSize, int backendId, int targetId)
        : maxKeypoints_(maxKeypoints),
          scoreThreshold_(scoreThreshold),
          imageSize_(imageSize)
    {
        validateImageSize(imageSize_);
        initNet(readNetFromONNX(modelPath), backendId, targetId);
    }

    DISK_Impl(const std::vector<uchar>& bufferModel, int maxKeypoints, float scoreThreshold,
              const Size& imageSize, int backendId, int targetId)
        : maxKeypoints_(maxKeypoints),
          scoreThreshold_(scoreThreshold),
          imageSize_(imageSize)
    {
        validateImageSize(imageSize_);
        initNet(readNetFromONNX(bufferModel), backendId, targetId);
    }

    void detectAndCompute(InputArray _image, InputArray _mask,
                          std::vector<KeyPoint>& keypoints,
                          OutputArray _descriptors,
                          bool useProvidedKeypoints) CV_OVERRIDE
    {
        CV_Assert(!useProvidedKeypoints && "DISK does not support providing keypoints externally");

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

        const Size netSize = (imageSize_.width > 0 && imageSize_.height > 0) ? imageSize_ : kDefaultDiskInputSize;
        const float scaleX = static_cast<float>(image.cols) / netSize.width;
        const float scaleY = static_cast<float>(image.rows) / netSize.height;

        Mat blob;
        // 1/255 normalization, swap BGR->RGB, no mean subtraction.
        blobFromImage(image, blob, 1.0 / 255.0, netSize, Scalar(), /*swapRB=*/true, /*crop=*/false);
        net_.setInput(blob, "image");

        const std::vector<String> outNames = {"keypoints", "scores", "descriptors"};
        std::vector<Mat> outs;
        net_.forward(outs, outNames);
        CV_Assert(outs.size() == 3);

        // DISK's ONNX export emits keypoints as int64 pixel coordinates, scores and
        // descriptors as float32. Reshape each output to (N, *) for row-wise indexing.
        Mat kptsBlob   = outs[0].reshape(1, outs[0].size[1]); // N x 2, int64
        Mat scoresBlob = outs[1].reshape(1, outs[1].size[1]); // N x 1, float32
        Mat descBlob   = outs[2].reshape(1, outs[2].size[1]); // N x D, float32

        CV_Assert(kptsBlob.depth()   == CV_64S);
        CV_Assert(scoresBlob.depth() == CV_32F);
        CV_Assert(descBlob.depth()   == CV_32F);

        const int numFeatures = kptsBlob.rows;
        CV_Assert(scoresBlob.rows == numFeatures);
        CV_Assert(descBlob.rows == numFeatures);

        const int64_t* kptsData   = kptsBlob.ptr<int64_t>();
        const float*   scoresData = scoresBlob.ptr<float>();

        std::vector<int> validIndices;
        validIndices.reserve(numFeatures);
        keypoints.reserve(numFeatures);

        for (int i = 0; i < numFeatures; ++i)
        {
            const float score = scoresData[i];
            if (score <= scoreThreshold_)
                continue;

            const float x = static_cast<float>(kptsData[i * 2])     * scaleX;
            const float y = static_cast<float>(kptsData[i * 2 + 1]) * scaleY;

            if (!mask.empty())
            {
                const int ix = cvFloor(x);
                const int iy = cvFloor(y);
                if (ix < 0 || iy < 0 || ix >= mask.cols || iy >= mask.rows)
                    continue;
                if (mask.at<uchar>(iy, ix) == 0)
                    continue;
            }

            keypoints.emplace_back(x, y, 1.0f, -1.0f, score);
            validIndices.push_back(i);
        }

        if (maxKeypoints_ > 0 && static_cast<int>(keypoints.size()) > maxKeypoints_)
        {
            std::vector<int> order(keypoints.size());
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + maxKeypoints_, order.end(),
                [&](int a, int b) { return keypoints[a].response > keypoints[b].response; });
            order.resize(maxKeypoints_);

            std::vector<KeyPoint> kept;
            std::vector<int> keptIdx;
            kept.reserve(maxKeypoints_);
            keptIdx.reserve(maxKeypoints_);
            for (int idx : order)
            {
                kept.push_back(keypoints[idx]);
                keptIdx.push_back(validIndices[idx]);
            }
            keypoints.swap(kept);
            validIndices.swap(keptIdx);
        }

        if (_descriptors.needed())
        {
            if (validIndices.empty())
            {
                _descriptors.release();
                return;
            }
            const int dim = descBlob.cols;
            _descriptors.create(static_cast<int>(validIndices.size()), dim, CV_32F);
            Mat descriptors = _descriptors.getMat();
            for (size_t i = 0; i < validIndices.size(); ++i)
                descBlob.row(validIndices[i]).copyTo(descriptors.row(static_cast<int>(i)));
        }
    }

    int descriptorSize() const CV_OVERRIDE { return 128; }
    int descriptorType() const CV_OVERRIDE { return CV_32F; }
    int defaultNorm()    const CV_OVERRIDE { return NORM_L2; }

    bool empty() const CV_OVERRIDE { return net_.empty(); }

    void setMaxKeypoints(int maxKeypoints) CV_OVERRIDE { maxKeypoints_ = maxKeypoints; }
    int  getMaxKeypoints() const CV_OVERRIDE { return maxKeypoints_; }

    void  setScoreThreshold(float threshold) CV_OVERRIDE { scoreThreshold_ = threshold; }
    float getScoreThreshold() const CV_OVERRIDE { return scoreThreshold_; }

    void setImageSize(const Size& size) CV_OVERRIDE
    {
        validateImageSize(size);
        imageSize_ = size;
    }
    Size getImageSize() const CV_OVERRIDE { return imageSize_; }

    String getDefaultName() const CV_OVERRIDE { return Feature2D::getDefaultName() + ".DISK"; }

private:
    void initNet(const Net& net, int backendId, int targetId)
    {
        net_ = net;
        net_.setPreferableBackend(backendId);
        net_.setPreferableTarget(targetId);
    }

    static void validateImageSize(const Size& size)
    {
        if (size.width == 0 && size.height == 0)
            return; // use default
        CV_Assert(size.width > 0 && size.height > 0);
        CV_Assert(size.width  % kDiskStride == 0);
        CV_Assert(size.height % kDiskStride == 0);
    }

    int    maxKeypoints_;
    float  scoreThreshold_;
    Size   imageSize_;
    Net    net_;
};

Ptr<DISK> DISK::create(const String& modelPath, int maxKeypoints, float scoreThreshold,
                       const Size& imageSize, int backendId, int targetId)
{
    CV_TRACE_FUNCTION();
    return makePtr<DISK_Impl>(modelPath, maxKeypoints, scoreThreshold, imageSize, backendId, targetId);
}

Ptr<DISK> DISK::create(const std::vector<uchar>& bufferModel, int maxKeypoints, float scoreThreshold,
                       const Size& imageSize, int backendId, int targetId)
{
    CV_TRACE_FUNCTION();
    return makePtr<DISK_Impl>(bufferModel, maxKeypoints, scoreThreshold, imageSize, backendId, targetId);
}

String DISK::getDefaultName() const
{
    return Feature2D::getDefaultName() + ".DISK";
}

} // namespace cv

#endif // HAVE_OPENCV_DNN
