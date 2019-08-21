/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;

#if !defined HAVE_CUDA || defined(CUDA_DISABLER)

Ptr<cuda::BackgroundSubtractorGMG> cv::cuda::createBackgroundSubtractorGMG(int, double) { throw_no_cuda(); return Ptr<cuda::BackgroundSubtractorGMG>(); }

#else

namespace cv { namespace cuda { namespace device {
    namespace gmg
    {
        void loadConstants(int width, int height, float minVal, float maxVal, int quantizationLevels, float backgroundPrior,
                           float decisionThreshold, int maxFeatures, int numInitializationFrames);

        template <typename SrcT>
        void update_gpu(PtrStepSzb frame, PtrStepb fgmask, PtrStepSzi colors, PtrStepf weights, PtrStepi nfeatures,
                        int frameNum,  float learningRate, bool updateBackgroundModel, cudaStream_t stream);
    }
}}}

namespace
{
    class GMGImpl : public cuda::BackgroundSubtractorGMG
    {
    public:
        GMGImpl(int initializationFrames, double decisionThreshold);

        void apply(InputArray image, OutputArray fgmask, double learningRate=-1);
        void apply(InputArray image, OutputArray fgmask, double learningRate, Stream& stream);

        void getBackgroundImage(OutputArray backgroundImage) const;

        int getMaxFeatures() const { return maxFeatures_; }
        void setMaxFeatures(int maxFeatures) { maxFeatures_ = maxFeatures; }

        double getDefaultLearningRate() const { return learningRate_; }
        void setDefaultLearningRate(double lr) { learningRate_ = (float) lr; }

        int getNumFrames() const { return numInitializationFrames_; }
        void setNumFrames(int nframes) { numInitializationFrames_ = nframes; }

        int getQuantizationLevels() const { return quantizationLevels_; }
        void setQuantizationLevels(int nlevels) { quantizationLevels_ = nlevels; }

        double getBackgroundPrior() const { return backgroundPrior_; }
        void setBackgroundPrior(double bgprior) { backgroundPrior_ = (float) bgprior; }

        int getSmoothingRadius() const { return smoothingRadius_; }
        void setSmoothingRadius(int radius) { smoothingRadius_ = radius; }

        double getDecisionThreshold() const { return decisionThreshold_; }
        void setDecisionThreshold(double thresh) { decisionThreshold_ = (float) thresh; }

        bool getUpdateBackgroundModel() const { return updateBackgroundModel_; }
        void setUpdateBackgroundModel(bool update) { updateBackgroundModel_ = update; }

        double getMinVal() const { return minVal_; }
        void setMinVal(double val) { minVal_ = (float) val; }

        double getMaxVal() const { return maxVal_; }
        void setMaxVal(double val) { maxVal_ = (float) val; }

    private:
        void initialize(Size frameSize, float min, float max);

        //! Total number of distinct colors to maintain in histogram.
        int maxFeatures_;

        //! Set between 0.0 and 1.0, determines how quickly features are "forgotten" from histograms.
        float learningRate_;

        //! Number of frames of video to use to initialize histograms.
        int numInitializationFrames_;

        //! Number of discrete levels in each channel to be used in histograms.
        int quantizationLevels_;

        //! Prior probability that any given pixel is a background pixel. A sensitivity parameter.
        float backgroundPrior_;

        //! Smoothing radius, in pixels, for cleaning up FG image.
        int smoothingRadius_;

        //! Value above which pixel is determined to be FG.
        float decisionThreshold_;

        //! Perform background model update.
        bool updateBackgroundModel_;

        float minVal_, maxVal_;

        Size frameSize_;
        int frameNum_;

        GpuMat nfeatures_;
        GpuMat colors_;
        GpuMat weights_;

#if defined(HAVE_OPENCV_CUDAFILTERS) && defined(HAVE_OPENCV_CUDAARITHM)
        Ptr<cuda::Filter> boxFilter_;
        GpuMat buf_;
#endif
    };

    GMGImpl::GMGImpl(int initializationFrames, double decisionThreshold)
    {
        maxFeatures_ = 64;
        learningRate_ = 0.025f;
        numInitializationFrames_ = initializationFrames;
        quantizationLevels_ = 16;
        backgroundPrior_ = 0.8f;
        decisionThreshold_ = (float) decisionThreshold;
        smoothingRadius_ = 7;
        updateBackgroundModel_ = true;
        minVal_ = maxVal_ = 0;
    }

    void GMGImpl::apply(InputArray image, OutputArray fgmask, double learningRate)
    {
        apply(image, fgmask, learningRate, Stream::Null());
    }

    void GMGImpl::apply(InputArray _frame, OutputArray _fgmask, double newLearningRate, Stream& stream)
    {
        using namespace cv::cuda::device::gmg;

        typedef void (*func_t)(PtrStepSzb frame, PtrStepb fgmask, PtrStepSzi colors, PtrStepf weights, PtrStepi nfeatures,
                               int frameNum, float learningRate, bool updateBackgroundModel, cudaStream_t stream);
        static const func_t funcs[6][4] =
        {
            {update_gpu<uchar>, 0, update_gpu<uchar3>, update_gpu<uchar4>},
            {0,0,0,0},
            {update_gpu<ushort>, 0, update_gpu<ushort3>, update_gpu<ushort4>},
            {0,0,0,0},
            {0,0,0,0},
            {update_gpu<float>, 0, update_gpu<float3>, update_gpu<float4>}
        };

        GpuMat frame = _frame.getGpuMat();

        CV_Assert( frame.depth() == CV_8U || frame.depth() == CV_16U || frame.depth() == CV_32F );
        CV_Assert( frame.channels() == 1 || frame.channels() == 3 || frame.channels() == 4 );

        if (newLearningRate != -1.0)
        {
            CV_Assert( newLearningRate >= 0.0 && newLearningRate <= 1.0 );
            learningRate_ = (float) newLearningRate;
        }

        if (frame.size() != frameSize_)
        {
            double minVal = minVal_;
            double maxVal = maxVal_;

            if (minVal_ == 0 && maxVal_ == 0)
            {
                minVal = 0;
                maxVal = frame.depth() == CV_8U ? 255.0 : frame.depth() == CV_16U ? std::numeric_limits<ushort>::max() : 1.0;
            }

            initialize(frame.size(), (float) minVal, (float) maxVal);
        }

        _fgmask.create(frameSize_, CV_8UC1);
        GpuMat fgmask = _fgmask.getGpuMat();

        fgmask.setTo(Scalar::all(0), stream);

        funcs[frame.depth()][frame.channels() - 1](frame, fgmask, colors_, weights_, nfeatures_, frameNum_,
                                                   learningRate_, updateBackgroundModel_, StreamAccessor::getStream(stream));

#if defined(HAVE_OPENCV_CUDAFILTERS) && defined(HAVE_OPENCV_CUDAARITHM)
        // medianBlur
        if (smoothingRadius_ > 0)
        {
            boxFilter_->apply(fgmask, buf_, stream);
            const int minCount = (smoothingRadius_ * smoothingRadius_ + 1) / 2;
            const double thresh = 255.0 * minCount / (smoothingRadius_ * smoothingRadius_);
            cuda::threshold(buf_, fgmask, thresh, 255.0, THRESH_BINARY, stream);
        }
#endif

        // keep track of how many frames we have processed
        ++frameNum_;
    }

    void GMGImpl::getBackgroundImage(OutputArray backgroundImage) const
    {
        CV_UNUSED(backgroundImage);
        CV_Error(Error::StsNotImplemented, "Not implemented");
    }

    void GMGImpl::initialize(Size frameSize, float min, float max)
    {
        using namespace cv::cuda::device::gmg;

        CV_Assert( maxFeatures_ > 0 );
        CV_Assert( learningRate_ >= 0.0f && learningRate_ <= 1.0f);
        CV_Assert( numInitializationFrames_ >= 1);
        CV_Assert( quantizationLevels_ >= 1 && quantizationLevels_ <= 255);
        CV_Assert( backgroundPrior_ >= 0.0f && backgroundPrior_ <= 1.0f);

        minVal_ = min;
        maxVal_ = max;
        CV_Assert( minVal_ < maxVal_ );

        frameSize_ = frameSize;

        frameNum_ = 0;

        nfeatures_.create(frameSize_, CV_32SC1);
        colors_.create(maxFeatures_ * frameSize_.height, frameSize_.width, CV_32SC1);
        weights_.create(maxFeatures_ * frameSize_.height, frameSize_.width, CV_32FC1);

        nfeatures_.setTo(Scalar::all(0));

#if defined(HAVE_OPENCV_CUDAFILTERS) && defined(HAVE_OPENCV_CUDAARITHM)
        if (smoothingRadius_ > 0)
            boxFilter_ = cuda::createBoxFilter(CV_8UC1, -1, Size(smoothingRadius_, smoothingRadius_));
#endif

        loadConstants(frameSize_.width, frameSize_.height, minVal_, maxVal_,
                      quantizationLevels_, backgroundPrior_, decisionThreshold_, maxFeatures_, numInitializationFrames_);
    }
}

Ptr<cuda::BackgroundSubtractorGMG> cv::cuda::createBackgroundSubtractorGMG(int initializationFrames, double decisionThreshold)
{
    return makePtr<GMGImpl>(initializationFrames, decisionThreshold);
}

#endif
