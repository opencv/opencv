/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/*
 * This class implements an algorithm described in "Visual Tracking of Human Visitors under
 * Variable-Lighting Conditions for a Responsive Audio Art Installation," A. Godbehere,
 * A. Matsukawa, K. Goldberg, American Control Conference, Montreal, June 2012.
 *
 * Prepared and integrated by Andrew B. Godbehere.
 */

#include "precomp.hpp"

cv::BackgroundSubtractorGMG::BackgroundSubtractorGMG()
{
    /*
     * Default Parameter Values. Override with algorithm "set" method.
     */
    maxFeatures = 64;
    learningRate = 0.025;
    numInitializationFrames = 120;
    quantizationLevels = 16;
    backgroundPrior = 0.8;
    decisionThreshold = 0.8;
    smoothingRadius = 7;
    updateBackgroundModel = true;
}

cv::BackgroundSubtractorGMG::~BackgroundSubtractorGMG()
{
}

void cv::BackgroundSubtractorGMG::initialize(cv::Size frameSize, double min, double max)
{
    CV_Assert(min < max);
    CV_Assert(maxFeatures > 0);
    CV_Assert(learningRate >= 0.0 && learningRate <= 1.0);
    CV_Assert(numInitializationFrames >= 1);
    CV_Assert(quantizationLevels >= 1 && quantizationLevels <= 255);
    CV_Assert(backgroundPrior >= 0.0 && backgroundPrior <= 1.0);

    minVal_ = min;
    maxVal_ = max;

    frameSize_ = frameSize;
    frameNum_ = 0;

    nfeatures_.create(frameSize_);
    colors_.create(frameSize_.area(), maxFeatures);
    weights_.create(frameSize_.area(), maxFeatures);

    nfeatures_.setTo(cv::Scalar::all(0));
}

namespace
{
    float findFeature(int color, const int* colors, const float* weights, int nfeatures)
    {
        for (int i = 0; i < nfeatures; ++i)
        {
            if (color == colors[i])
                return weights[i];
        }

        // not in histogram, so return 0.
        return 0.0f;
    }

    void normalizeHistogram(float* weights, int nfeatures)
    {
        float total = 0.0f;
        for (int i = 0; i < nfeatures; ++i)
            total += weights[i];

        if (total != 0.0f)
        {
            for (int i = 0; i < nfeatures; ++i)
                weights[i] /= total;
        }
    }

    bool insertFeature(int color, float weight, int* colors, float* weights, int& nfeatures, int maxFeatures)
    {
        int idx = -1;
        for (int i = 0; i < nfeatures; ++i)
        {
            if (color == colors[i])
            {
                // feature in histogram
                weight += weights[i];
                idx = i;
                break;
            }
        }

        if (idx >= 0)
        {
            // move feature to beginning of list

            ::memmove(colors + 1, colors, idx * sizeof(int));
            ::memmove(weights + 1, weights, idx * sizeof(float));

            colors[0] = color;
            weights[0] = weight;
        }
        else if (nfeatures == maxFeatures)
        {
            // discard oldest feature

            ::memmove(colors + 1, colors, (nfeatures - 1) * sizeof(int));
            ::memmove(weights + 1, weights, (nfeatures - 1) * sizeof(float));

            colors[0] = color;
            weights[0] = weight;
        }
        else
        {
            colors[nfeatures] = color;
            weights[nfeatures] = weight;

            ++nfeatures;

            return true;
        }

        return false;
    }
}

namespace
{
    template <int cn> struct Quantization_
    {
        template <typename T>
        static inline int apply(T val, double minVal, double maxVal, int quantizationLevels)
        {
            int res = 0;
            res |= static_cast<int>((val[0] - minVal) * quantizationLevels / (maxVal - minVal));
            res |= static_cast<int>((val[1] - minVal) * quantizationLevels / (maxVal - minVal)) << 8;
            res |= static_cast<int>((val[2] - minVal) * quantizationLevels / (maxVal - minVal)) << 16;
            return res;
        }
    };
    template <> struct Quantization_<1>
    {
        template <typename T>
        static inline int apply(T val, double minVal, double maxVal, int quantizationLevels)
        {
            return static_cast<int>((val - minVal) * quantizationLevels / (maxVal - minVal));
        }
    };
    template <typename T> struct Quantization
    {
        static int apply(const void* src_, int x, double minVal, double maxVal, int quantizationLevels)
        {
            const T* src = static_cast<const T*>(src_);
            return Quantization_<cv::DataType<T>::channels>::apply(src[x], minVal, maxVal, quantizationLevels);
        }
    };

    class GMG_LoopBody : public cv::ParallelLoopBody
    {
    public:
        GMG_LoopBody(const cv::Mat& frame, const cv::Mat& fgmask, const cv::Mat_<int>& nfeatures, const cv::Mat_<int>& colors, const cv::Mat_<float>& weights,
                     int maxFeatures, double learningRate, int numInitializationFrames, int quantizationLevels, double backgroundPrior, double decisionThreshold,
                     double maxVal, double minVal, int frameNum, bool updateBackgroundModel) :
            frame_(frame), fgmask_(fgmask), nfeatures_(nfeatures), colors_(colors), weights_(weights),
            maxFeatures_(maxFeatures), learningRate_(learningRate), numInitializationFrames_(numInitializationFrames),
            quantizationLevels_(quantizationLevels), backgroundPrior_(backgroundPrior), decisionThreshold_(decisionThreshold),
            maxVal_(maxVal), minVal_(minVal), frameNum_(frameNum), updateBackgroundModel_(updateBackgroundModel)
        {
        }

        void operator() (const cv::Range& range) const;

    private:
        const cv::Mat frame_;

        mutable cv::Mat_<uchar> fgmask_;

        mutable cv::Mat_<int> nfeatures_;
        mutable cv::Mat_<int> colors_;
        mutable cv::Mat_<float> weights_;

        int     maxFeatures_;
        double  learningRate_;
        int     numInitializationFrames_;
        int     quantizationLevels_;
        double  backgroundPrior_;
        double  decisionThreshold_;
        bool updateBackgroundModel_;

        double maxVal_;
        double minVal_;
        int frameNum_;
    };

    void GMG_LoopBody::operator() (const cv::Range& range) const
    {
        typedef int (*func_t)(const void* src_, int x, double minVal, double maxVal, int quantizationLevels);
        static const func_t funcs[6][4] =
        {
            {Quantization<uchar>::apply, 0, Quantization<cv::Vec3b>::apply, Quantization<cv::Vec4b>::apply},
            {0,0,0,0},
            {Quantization<ushort>::apply, 0, Quantization<cv::Vec3w>::apply, Quantization<cv::Vec4w>::apply},
            {0,0,0,0},
            {0,0,0,0},
            {Quantization<float>::apply, 0, Quantization<cv::Vec3f>::apply, Quantization<cv::Vec4f>::apply},
        };

        const func_t func = funcs[frame_.depth()][frame_.channels() - 1];
        CV_Assert(func != 0);

        for (int y = range.start, featureIdx = y * frame_.cols; y < range.end; ++y)
        {
            const uchar* frame_row = frame_.ptr(y);
            int* nfeatures_row = nfeatures_[y];
            uchar* fgmask_row = fgmask_[y];

            for (int x = 0; x < frame_.cols; ++x, ++featureIdx)
            {
                int nfeatures = nfeatures_row[x];
                int* colors = colors_[featureIdx];
                float* weights = weights_[featureIdx];

                int newFeatureColor = func(frame_row, x, minVal_, maxVal_, quantizationLevels_);

                bool isForeground = false;

                if (frameNum_ >= numInitializationFrames_)
                {
                    // typical operation

                    const double weight = findFeature(newFeatureColor, colors, weights, nfeatures);

                    // see Godbehere, Matsukawa, Goldberg (2012) for reasoning behind this implementation of Bayes rule
                    const double posterior = (weight * backgroundPrior_) / (weight * backgroundPrior_ + (1.0 - weight) * (1.0 - backgroundPrior_));

                    isForeground = ((1.0 - posterior) > decisionThreshold_);

                    // update histogram.

                    if (updateBackgroundModel_)
                    {
                        for (int i = 0; i < nfeatures; ++i)
                            weights[i] *= 1.0f - learningRate_;

                        bool inserted = insertFeature(newFeatureColor, learningRate_, colors, weights, nfeatures, maxFeatures_);

                        if (inserted)
                        {
                            normalizeHistogram(weights, nfeatures);
                            nfeatures_row[x] = nfeatures;
                        }
                    }
                }
                else if (updateBackgroundModel_)
                {
                    // training-mode update

                    insertFeature(newFeatureColor, 1.0f, colors, weights, nfeatures, maxFeatures_);

                    if (frameNum_ == numInitializationFrames_ - 1)
                        normalizeHistogram(weights, nfeatures);
                }

                fgmask_row[x] = (uchar)(-isForeground);
            }
        }
    }
}

void cv::BackgroundSubtractorGMG::operator ()(InputArray _frame, OutputArray _fgmask, double newLearningRate)
{
    cv::Mat frame = _frame.getMat();

    CV_Assert(frame.depth() == CV_8U || frame.depth() == CV_16U || frame.depth() == CV_32F);
    CV_Assert(frame.channels() == 1 || frame.channels() == 3 || frame.channels() == 4);

    if (newLearningRate != -1.0)
    {
        CV_Assert(newLearningRate >= 0.0 && newLearningRate <= 1.0);
        learningRate = newLearningRate;
    }

    if (frame.size() != frameSize_)
        initialize(frame.size(), 0.0, frame.depth() == CV_8U ? 255.0 : frame.depth() == CV_16U ? std::numeric_limits<ushort>::max() : 1.0);

    _fgmask.create(frameSize_, CV_8UC1);
    cv::Mat fgmask = _fgmask.getMat();

    GMG_LoopBody body(frame, fgmask, nfeatures_, colors_, weights_,
                      maxFeatures, learningRate, numInitializationFrames, quantizationLevels, backgroundPrior, decisionThreshold,
                      maxVal_, minVal_, frameNum_, updateBackgroundModel);
    cv::parallel_for_(cv::Range(0, frame.rows), body);

    if (smoothingRadius > 0)
    {
        cv::medianBlur(fgmask, buf_, smoothingRadius);
        cv::swap(fgmask, buf_);
    }

    // keep track of how many frames we have processed
    ++frameNum_;
}

void cv::BackgroundSubtractorGMG::release()
{
    frameSize_ = cv::Size();

    nfeatures_.release();
    colors_.release();
    weights_.release();
    buf_.release();
}
