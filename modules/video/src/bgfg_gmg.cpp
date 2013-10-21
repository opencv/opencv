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
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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
#include <limits>

namespace cv
{

class BackgroundSubtractorGMGImpl : public BackgroundSubtractorGMG
{
public:
    BackgroundSubtractorGMGImpl()
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
        minVal_ = maxVal_ = 0;
        name_ = "BackgroundSubtractor.GMG";
    }

    ~BackgroundSubtractorGMGImpl()
    {
    }

    virtual AlgorithmInfo* info() const { return 0; }

    /**
     * Validate parameters and set up data structures for appropriate image size.
     * Must call before running on data.
     * @param frameSize input frame size
     * @param min       minimum value taken on by pixels in image sequence. Usually 0
     * @param max       maximum value taken on by pixels in image sequence. e.g. 1.0 or 255
     */
    void initialize(Size frameSize, double minVal, double maxVal);

    /**
     * Performs single-frame background subtraction and builds up a statistical background image
     * model.
     * @param image Input image
     * @param fgmask Output mask image representing foreground and background pixels
     */
    virtual void apply(InputArray image, OutputArray fgmask, double learningRate=-1.0);

    /**
     * Releases all inner buffers.
     */
    void release();

    virtual int getMaxFeatures() const { return maxFeatures; }
    virtual void setMaxFeatures(int _maxFeatures) { maxFeatures = _maxFeatures; }

    virtual double getDefaultLearningRate() const { return learningRate; }
    virtual void setDefaultLearningRate(double lr) { learningRate = lr; }

    virtual int getNumFrames() const { return numInitializationFrames; }
    virtual void setNumFrames(int nframes) { numInitializationFrames = nframes; }

    virtual int getQuantizationLevels() const { return quantizationLevels; }
    virtual void setQuantizationLevels(int nlevels) { quantizationLevels = nlevels; }

    virtual double getBackgroundPrior() const { return backgroundPrior; }
    virtual void setBackgroundPrior(double bgprior) { backgroundPrior = bgprior; }

    virtual int getSmoothingRadius() const { return smoothingRadius; }
    virtual void setSmoothingRadius(int radius) { smoothingRadius = radius; }

    virtual double getDecisionThreshold() const { return decisionThreshold; }
    virtual void setDecisionThreshold(double thresh) { decisionThreshold = thresh; }

    virtual bool getUpdateBackgroundModel() const { return updateBackgroundModel; }
    virtual void setUpdateBackgroundModel(bool update) { updateBackgroundModel = update; }

    virtual double getMinVal() const { return minVal_; }
    virtual void setMinVal(double val) { minVal_ = val; }

    virtual double getMaxVal() const  { return maxVal_; }
    virtual void setMaxVal(double val)  { maxVal_ = val; }

    virtual void getBackgroundImage(OutputArray) const
    {
        CV_Error( Error::StsNotImplemented, "" );
    }

    virtual void write(FileStorage& fs) const
    {
        fs << "name" << name_
        << "maxFeatures" << maxFeatures
        << "defaultLearningRate" << learningRate
        << "numFrames" << numInitializationFrames
        << "quantizationLevels" << quantizationLevels
        << "backgroundPrior" << backgroundPrior
        << "decisionThreshold" << decisionThreshold
        << "smoothingRadius" << smoothingRadius
        << "updateBackgroundModel" << (int)updateBackgroundModel;
        // we do not save minVal_ & maxVal_, since they depend on the image type.
    }

    virtual void read(const FileNode& fn)
    {
        CV_Assert( (String)fn["name"] == name_ );
        maxFeatures = (int)fn["maxFeatures"];
        learningRate = (double)fn["defaultLearningRate"];
        numInitializationFrames = (int)fn["numFrames"];
        quantizationLevels = (int)fn["quantizationLevels"];
        backgroundPrior = (double)fn["backgroundPrior"];
        smoothingRadius = (int)fn["smoothingRadius"];
        decisionThreshold = (double)fn["decisionThreshold"];
        updateBackgroundModel = (int)fn["updateBackgroundModel"] != 0;
        minVal_ = maxVal_ = 0;
        frameSize_ = Size();
    }

    //! Total number of distinct colors to maintain in histogram.
    int     maxFeatures;
    //! Set between 0.0 and 1.0, determines how quickly features are "forgotten" from histograms.
    double  learningRate;
    //! Number of frames of video to use to initialize histograms.
    int     numInitializationFrames;
    //! Number of discrete levels in each channel to be used in histograms.
    int     quantizationLevels;
    //! Prior probability that any given pixel is a background pixel. A sensitivity parameter.
    double  backgroundPrior;
    //! Value above which pixel is determined to be FG.
    double  decisionThreshold;
    //! Smoothing radius, in pixels, for cleaning up FG image.
    int     smoothingRadius;
    //! Perform background model update
    bool updateBackgroundModel;

private:
    double maxVal_;
    double minVal_;

    Size frameSize_;
    int frameNum_;

    String name_;

    Mat_<int> nfeatures_;
    Mat_<unsigned int> colors_;
    Mat_<float> weights_;

    Mat buf_;
};


void BackgroundSubtractorGMGImpl::initialize(Size frameSize, double minVal, double maxVal)
{
    CV_Assert(minVal < maxVal);
    CV_Assert(maxFeatures > 0);
    CV_Assert(learningRate >= 0.0 && learningRate <= 1.0);
    CV_Assert(numInitializationFrames >= 1);
    CV_Assert(quantizationLevels >= 1 && quantizationLevels <= 255);
    CV_Assert(backgroundPrior >= 0.0 && backgroundPrior <= 1.0);

    minVal_ = minVal;
    maxVal_ = maxVal;

    frameSize_ = frameSize;
    frameNum_ = 0;

    nfeatures_.create(frameSize_);
    colors_.create(frameSize_.area(), maxFeatures);
    weights_.create(frameSize_.area(), maxFeatures);

    nfeatures_.setTo(Scalar::all(0));
}

namespace
{
    float findFeature(unsigned int color, const unsigned int* colors, const float* weights, int nfeatures)
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

    bool insertFeature(unsigned int color, float weight, unsigned int* colors, float* weights, int& nfeatures, int maxFeatures)
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

            ::memmove(colors + 1, colors, idx * sizeof(unsigned int));
            ::memmove(weights + 1, weights, idx * sizeof(float));

            colors[0] = color;
            weights[0] = weight;
        }
        else if (nfeatures == maxFeatures)
        {
            // discard oldest feature

            ::memmove(colors + 1, colors, (nfeatures - 1) * sizeof(unsigned int));
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
    template <typename T> struct Quantization
    {
        static unsigned int apply(const void* src_, int x, int cn, double minVal, double maxVal, int quantizationLevels)
        {
            const T* src = static_cast<const T*>(src_);
            src += x * cn;

            unsigned int res = 0;
            for (int i = 0, shift = 0; i < cn; ++i, ++src, shift += 8)
                res |= static_cast<int>((*src - minVal) * quantizationLevels / (maxVal - minVal)) << shift;

            return res;
        }
    };

    class GMG_LoopBody : public ParallelLoopBody
    {
    public:
        GMG_LoopBody(const Mat& frame, const Mat& fgmask, const Mat_<int>& nfeatures, const Mat_<unsigned int>& colors, const Mat_<float>& weights,
                     int maxFeatures, double learningRate, int numInitializationFrames, int quantizationLevels, double backgroundPrior, double decisionThreshold,
                     double maxVal, double minVal, int frameNum, bool updateBackgroundModel) :
            frame_(frame), fgmask_(fgmask), nfeatures_(nfeatures), colors_(colors), weights_(weights),
            maxFeatures_(maxFeatures), learningRate_(learningRate), numInitializationFrames_(numInitializationFrames), quantizationLevels_(quantizationLevels),
            backgroundPrior_(backgroundPrior), decisionThreshold_(decisionThreshold), updateBackgroundModel_(updateBackgroundModel),
            maxVal_(maxVal), minVal_(minVal), frameNum_(frameNum)
        {
        }

        void operator() (const Range& range) const;

    private:
        Mat frame_;

        mutable Mat_<uchar> fgmask_;

        mutable Mat_<int> nfeatures_;
        mutable Mat_<unsigned int> colors_;
        mutable Mat_<float> weights_;

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

    void GMG_LoopBody::operator() (const Range& range) const
    {
        typedef unsigned int (*func_t)(const void* src_, int x, int cn, double minVal, double maxVal, int quantizationLevels);
        static const func_t funcs[] =
        {
            Quantization<uchar>::apply,
            Quantization<schar>::apply,
            Quantization<ushort>::apply,
            Quantization<short>::apply,
            Quantization<int>::apply,
            Quantization<float>::apply,
            Quantization<double>::apply
        };

        const func_t func = funcs[frame_.depth()];
        CV_Assert(func != 0);

        const int cn = frame_.channels();

        for (int y = range.start, featureIdx = y * frame_.cols; y < range.end; ++y)
        {
            const uchar* frame_row = frame_.ptr(y);
            int* nfeatures_row = nfeatures_[y];
            uchar* fgmask_row = fgmask_[y];

            for (int x = 0; x < frame_.cols; ++x, ++featureIdx)
            {
                int nfeatures = nfeatures_row[x];
                unsigned int* colors = colors_[featureIdx];
                float* weights = weights_[featureIdx];

                unsigned int newFeatureColor = func(frame_row, x, cn, minVal_, maxVal_, quantizationLevels_);

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
                            weights[i] *= (float)(1.0f - learningRate_);

                        bool inserted = insertFeature(newFeatureColor, (float)learningRate_, colors, weights, nfeatures, maxFeatures_);

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

                fgmask_row[x] = (uchar)(-(schar)isForeground);
            }
        }
    }
}

void BackgroundSubtractorGMGImpl::apply(InputArray _frame, OutputArray _fgmask, double newLearningRate)
{
    Mat frame = _frame.getMat();

    CV_Assert(frame.depth() == CV_8U || frame.depth() == CV_16U || frame.depth() == CV_32F);
    CV_Assert(frame.channels() == 1 || frame.channels() == 3 || frame.channels() == 4);

    if (newLearningRate != -1.0)
    {
        CV_Assert(newLearningRate >= 0.0 && newLearningRate <= 1.0);
        learningRate = newLearningRate;
    }

    if (frame.size() != frameSize_)
    {
        double minval = minVal_;
        double maxval = maxVal_;
        if( minVal_ == 0 && maxVal_ == 0 )
        {
            minval = 0;
            maxval = frame.depth() == CV_8U ? 255.0 : frame.depth() == CV_16U ? std::numeric_limits<ushort>::max() : 1.0;
        }
        initialize(frame.size(), minval, maxval);
    }

    _fgmask.create(frameSize_, CV_8UC1);
    Mat fgmask = _fgmask.getMat();

    GMG_LoopBody body(frame, fgmask, nfeatures_, colors_, weights_,
                      maxFeatures, learningRate, numInitializationFrames, quantizationLevels, backgroundPrior, decisionThreshold,
                      maxVal_, minVal_, frameNum_, updateBackgroundModel);
    parallel_for_(Range(0, frame.rows), body, frame.total()/(double)(1<<16));

    if (smoothingRadius > 0)
    {
        medianBlur(fgmask, buf_, smoothingRadius);
        swap(fgmask, buf_);
    }

    // keep track of how many frames we have processed
    ++frameNum_;
}

void BackgroundSubtractorGMGImpl::release()
{
    frameSize_ = Size();

    nfeatures_.release();
    colors_.release();
    weights_.release();
    buf_.release();
}


Ptr<BackgroundSubtractorGMG> createBackgroundSubtractorGMG(int initializationFrames, double decisionThreshold)
{
    Ptr<BackgroundSubtractorGMG> bgfg = makePtr<BackgroundSubtractorGMGImpl>();
    bgfg->setNumFrames(initializationFrames);
    bgfg->setDecisionThreshold(decisionThreshold);

    return bgfg;
}

/*
 ///////////////////////////////////////////////////////////////////////////////////////////////////////////

 CV_INIT_ALGORITHM(BackgroundSubtractorGMG, "BackgroundSubtractor.GMG",
 obj.info()->addParam(obj, "maxFeatures", obj.maxFeatures,false,0,0,
 "Maximum number of features to store in histogram. Harsh enforcement of sparsity constraint.");
 obj.info()->addParam(obj, "learningRate", obj.learningRate,false,0,0,
 "Adaptation rate of histogram. Close to 1, slow adaptation. Close to 0, fast adaptation, features forgotten quickly.");
 obj.info()->addParam(obj, "initializationFrames", obj.numInitializationFrames,false,0,0,
 "Number of frames to use to initialize histograms of pixels.");
 obj.info()->addParam(obj, "quantizationLevels", obj.quantizationLevels,false,0,0,
 "Number of discrete colors to be used in histograms. Up-front quantization.");
 obj.info()->addParam(obj, "backgroundPrior", obj.backgroundPrior,false,0,0,
 "Prior probability that each individual pixel is a background pixel.");
 obj.info()->addParam(obj, "smoothingRadius", obj.smoothingRadius,false,0,0,
 "Radius of smoothing kernel to filter noise from FG mask image.");
 obj.info()->addParam(obj, "decisionThreshold", obj.decisionThreshold,false,0,0,
 "Threshold for FG decision rule. Pixel is FG if posterior probability exceeds threshold.");
 obj.info()->addParam(obj, "updateBackgroundModel", obj.updateBackgroundModel,false,0,0,
 "Perform background model update.");
 obj.info()->addParam(obj, "minVal", obj.minVal_,false,0,0,
 "Minimum of the value range (mostly for regression testing)");
 obj.info()->addParam(obj, "maxVal", obj.maxVal_,false,0,0,
 "Maximum of the value range (mostly for regression testing)");
 );
*/

}
