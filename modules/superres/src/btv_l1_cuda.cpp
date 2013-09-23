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

// S. Farsiu , D. Robinson, M. Elad, P. Milanfar. Fast and robust multiframe super resolution.
// Dennis Mitzel, Thomas Pock, Thomas Schoenemann, Daniel Cremers. Video Super Resolution using Duality Based TV-L1 Optical Flow.

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;
using namespace cv::superres;
using namespace cv::superres::detail;

#if !defined(HAVE_CUDA) || !defined(HAVE_OPENCV_CUDAARITHM) || !defined(HAVE_OPENCV_CUDAWARPING) || !defined(HAVE_OPENCV_CUDAFILTERS)

Ptr<SuperResolution> cv::superres::createSuperResolution_BTVL1_CUDA()
{
    CV_Error(Error::StsNotImplemented, "The called functionality is disabled for current build or platform");
    return Ptr<SuperResolution>();
}

#else // HAVE_CUDA

namespace btv_l1_cudev
{
    void buildMotionMaps(PtrStepSzf forwardMotionX, PtrStepSzf forwardMotionY,
                         PtrStepSzf backwardMotionX, PtrStepSzf bacwardMotionY,
                         PtrStepSzf forwardMapX, PtrStepSzf forwardMapY,
                         PtrStepSzf backwardMapX, PtrStepSzf backwardMapY);

    template <int cn>
    void upscale(const PtrStepSzb src, PtrStepSzb dst, int scale, cudaStream_t stream);

    void diffSign(PtrStepSzf src1, PtrStepSzf src2, PtrStepSzf dst, cudaStream_t stream);

    void loadBtvWeights(const float* weights, size_t count);
    template <int cn> void calcBtvRegularization(PtrStepSzb src, PtrStepSzb dst, int ksize);
}

namespace
{
    void calcRelativeMotions(const std::vector<std::pair<GpuMat, GpuMat> >& forwardMotions, const std::vector<std::pair<GpuMat, GpuMat> >& backwardMotions,
                             std::vector<std::pair<GpuMat, GpuMat> >& relForwardMotions, std::vector<std::pair<GpuMat, GpuMat> >& relBackwardMotions,
                             int baseIdx, Size size)
    {
        const int count = static_cast<int>(forwardMotions.size());

        relForwardMotions.resize(count);
        relForwardMotions[baseIdx].first.create(size, CV_32FC1);
        relForwardMotions[baseIdx].first.setTo(Scalar::all(0));
        relForwardMotions[baseIdx].second.create(size, CV_32FC1);
        relForwardMotions[baseIdx].second.setTo(Scalar::all(0));

        relBackwardMotions.resize(count);
        relBackwardMotions[baseIdx].first.create(size, CV_32FC1);
        relBackwardMotions[baseIdx].first.setTo(Scalar::all(0));
        relBackwardMotions[baseIdx].second.create(size, CV_32FC1);
        relBackwardMotions[baseIdx].second.setTo(Scalar::all(0));

        for (int i = baseIdx - 1; i >= 0; --i)
        {
            cuda::add(relForwardMotions[i + 1].first, forwardMotions[i].first, relForwardMotions[i].first);
            cuda::add(relForwardMotions[i + 1].second, forwardMotions[i].second, relForwardMotions[i].second);

            cuda::add(relBackwardMotions[i + 1].first, backwardMotions[i + 1].first, relBackwardMotions[i].first);
            cuda::add(relBackwardMotions[i + 1].second, backwardMotions[i + 1].second, relBackwardMotions[i].second);
        }

        for (int i = baseIdx + 1; i < count; ++i)
        {
            cuda::add(relForwardMotions[i - 1].first, backwardMotions[i].first, relForwardMotions[i].first);
            cuda::add(relForwardMotions[i - 1].second, backwardMotions[i].second, relForwardMotions[i].second);

            cuda::add(relBackwardMotions[i - 1].first, forwardMotions[i - 1].first, relBackwardMotions[i].first);
            cuda::add(relBackwardMotions[i - 1].second, forwardMotions[i - 1].second, relBackwardMotions[i].second);
        }
    }

    void upscaleMotions(const std::vector<std::pair<GpuMat, GpuMat> >& lowResMotions, std::vector<std::pair<GpuMat, GpuMat> >& highResMotions, int scale)
    {
        highResMotions.resize(lowResMotions.size());

        for (size_t i = 0; i < lowResMotions.size(); ++i)
        {
            cuda::resize(lowResMotions[i].first, highResMotions[i].first, Size(), scale, scale, INTER_CUBIC);
            cuda::resize(lowResMotions[i].second, highResMotions[i].second, Size(), scale, scale, INTER_CUBIC);

            cuda::multiply(highResMotions[i].first, Scalar::all(scale), highResMotions[i].first);
            cuda::multiply(highResMotions[i].second, Scalar::all(scale), highResMotions[i].second);
        }
    }

    void buildMotionMaps(const std::pair<GpuMat, GpuMat>& forwardMotion, const std::pair<GpuMat, GpuMat>& backwardMotion,
                         std::pair<GpuMat, GpuMat>& forwardMap, std::pair<GpuMat, GpuMat>& backwardMap)
    {
        forwardMap.first.create(forwardMotion.first.size(), CV_32FC1);
        forwardMap.second.create(forwardMotion.first.size(), CV_32FC1);

        backwardMap.first.create(forwardMotion.first.size(), CV_32FC1);
        backwardMap.second.create(forwardMotion.first.size(), CV_32FC1);

        btv_l1_cudev::buildMotionMaps(forwardMotion.first, forwardMotion.second,
                                       backwardMotion.first, backwardMotion.second,
                                       forwardMap.first, forwardMap.second,
                                       backwardMap.first, backwardMap.second);
    }

    void upscale(const GpuMat& src, GpuMat& dst, int scale, Stream& stream)
    {
        typedef void (*func_t)(const PtrStepSzb src, PtrStepSzb dst, int scale, cudaStream_t stream);
        static const func_t funcs[] =
        {
            0, btv_l1_cudev::upscale<1>, 0, btv_l1_cudev::upscale<3>, btv_l1_cudev::upscale<4>
        };

        CV_Assert( src.channels() == 1 || src.channels() == 3 || src.channels() == 4 );

        dst.create(src.rows * scale, src.cols * scale, src.type());
        dst.setTo(Scalar::all(0));

        const func_t func = funcs[src.channels()];

        func(src, dst, scale, StreamAccessor::getStream(stream));
    }

    void diffSign(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream)
    {
        dst.create(src1.size(), src1.type());

        btv_l1_cudev::diffSign(src1.reshape(1), src2.reshape(1), dst.reshape(1), StreamAccessor::getStream(stream));
    }

    void calcBtvWeights(int btvKernelSize, double alpha, std::vector<float>& btvWeights)
    {
        const size_t size = btvKernelSize * btvKernelSize;

        btvWeights.resize(size);

        const int ksize = (btvKernelSize - 1) / 2;
        const float alpha_f = static_cast<float>(alpha);

        for (int m = 0, ind = 0; m <= ksize; ++m)
        {
            for (int l = ksize; l + m >= 0; --l, ++ind)
                btvWeights[ind] = pow(alpha_f, std::abs(m) + std::abs(l));
        }

        btv_l1_cudev::loadBtvWeights(&btvWeights[0], size);
    }

    void calcBtvRegularization(const GpuMat& src, GpuMat& dst, int btvKernelSize)
    {
        typedef void (*func_t)(PtrStepSzb src, PtrStepSzb dst, int ksize);
        static const func_t funcs[] =
        {
            0,
            btv_l1_cudev::calcBtvRegularization<1>,
            0,
            btv_l1_cudev::calcBtvRegularization<3>,
            btv_l1_cudev::calcBtvRegularization<4>
        };

        dst.create(src.size(), src.type());
        dst.setTo(Scalar::all(0));

        const int ksize = (btvKernelSize - 1) / 2;

        funcs[src.channels()](src, dst, ksize);
    }

    class BTVL1_CUDA_Base
    {
    public:
        BTVL1_CUDA_Base();

        void process(const std::vector<GpuMat>& src, GpuMat& dst,
                     const std::vector<std::pair<GpuMat, GpuMat> >& forwardMotions, const std::vector<std::pair<GpuMat, GpuMat> >& backwardMotions,
                     int baseIdx);

        void collectGarbage();

    protected:
        int scale_;
        int iterations_;
        double lambda_;
        double tau_;
        double alpha_;
        int btvKernelSize_;
        int blurKernelSize_;
        double blurSigma_;
        Ptr<DenseOpticalFlowExt> opticalFlow_;

    private:
        std::vector<Ptr<cuda::Filter> > filters_;
        int curBlurKernelSize_;
        double curBlurSigma_;
        int curSrcType_;

        std::vector<float> btvWeights_;
        int curBtvKernelSize_;
        double curAlpha_;

        std::vector<std::pair<GpuMat, GpuMat> > lowResForwardMotions_;
        std::vector<std::pair<GpuMat, GpuMat> > lowResBackwardMotions_;

        std::vector<std::pair<GpuMat, GpuMat> > highResForwardMotions_;
        std::vector<std::pair<GpuMat, GpuMat> > highResBackwardMotions_;

        std::vector<std::pair<GpuMat, GpuMat> > forwardMaps_;
        std::vector<std::pair<GpuMat, GpuMat> > backwardMaps_;

        GpuMat highRes_;

        std::vector<Stream> streams_;
        std::vector<GpuMat> diffTerms_;
        std::vector<GpuMat> a_, b_, c_;
        GpuMat regTerm_;
    };

    BTVL1_CUDA_Base::BTVL1_CUDA_Base()
    {
        scale_ = 4;
        iterations_ = 180;
        lambda_ = 0.03;
        tau_ = 1.3;
        alpha_ = 0.7;
        btvKernelSize_ = 7;
        blurKernelSize_ = 5;
        blurSigma_ = 0.0;

#ifdef HAVE_OPENCV_CUDAOPTFLOW
        opticalFlow_ = createOptFlow_Farneback_CUDA();
#else
        opticalFlow_ = createOptFlow_Farneback();
#endif

        curBlurKernelSize_ = -1;
        curBlurSigma_ = -1.0;
        curSrcType_ = -1;

        curBtvKernelSize_ = -1;
        curAlpha_ = -1.0;
    }

    void BTVL1_CUDA_Base::process(const std::vector<GpuMat>& src, GpuMat& dst,
                                 const std::vector<std::pair<GpuMat, GpuMat> >& forwardMotions, const std::vector<std::pair<GpuMat, GpuMat> >& backwardMotions,
                                 int baseIdx)
    {
        CV_Assert( scale_ > 1 );
        CV_Assert( iterations_ > 0 );
        CV_Assert( tau_ > 0.0 );
        CV_Assert( alpha_ > 0.0 );
        CV_Assert( btvKernelSize_ > 0 && btvKernelSize_ <= 16 );
        CV_Assert( blurKernelSize_ > 0 );
        CV_Assert( blurSigma_ >= 0.0 );

        // update blur filter and btv weights

        if (filters_.size() != src.size() || blurKernelSize_ != curBlurKernelSize_ || blurSigma_ != curBlurSigma_ || src[0].type() != curSrcType_)
        {
            filters_.resize(src.size());
            for (size_t i = 0; i < src.size(); ++i)
                filters_[i] = cuda::createGaussianFilter(src[0].type(), -1, Size(blurKernelSize_, blurKernelSize_), blurSigma_);
            curBlurKernelSize_ = blurKernelSize_;
            curBlurSigma_ = blurSigma_;
            curSrcType_ = src[0].type();
        }

        if (btvWeights_.empty() || btvKernelSize_ != curBtvKernelSize_ || alpha_ != curAlpha_)
        {
            calcBtvWeights(btvKernelSize_, alpha_, btvWeights_);
            curBtvKernelSize_ = btvKernelSize_;
            curAlpha_ = alpha_;
        }

        // calc motions between input frames

        calcRelativeMotions(forwardMotions, backwardMotions, lowResForwardMotions_, lowResBackwardMotions_, baseIdx, src[0].size());

        upscaleMotions(lowResForwardMotions_, highResForwardMotions_, scale_);
        upscaleMotions(lowResBackwardMotions_, highResBackwardMotions_, scale_);

        forwardMaps_.resize(highResForwardMotions_.size());
        backwardMaps_.resize(highResForwardMotions_.size());
        for (size_t i = 0; i < highResForwardMotions_.size(); ++i)
            buildMotionMaps(highResForwardMotions_[i], highResBackwardMotions_[i], forwardMaps_[i], backwardMaps_[i]);

        // initial estimation

        const Size lowResSize = src[0].size();
        const Size highResSize(lowResSize.width * scale_, lowResSize.height * scale_);

        cuda::resize(src[baseIdx], highRes_, highResSize, 0, 0, INTER_CUBIC);

        // iterations

        streams_.resize(src.size());
        diffTerms_.resize(src.size());
        a_.resize(src.size());
        b_.resize(src.size());
        c_.resize(src.size());

        for (int i = 0; i < iterations_; ++i)
        {
            for (size_t k = 0; k < src.size(); ++k)
            {
                // a = M * Ih
                cuda::remap(highRes_, a_[k], backwardMaps_[k].first, backwardMaps_[k].second, INTER_NEAREST, BORDER_REPLICATE, Scalar(), streams_[k]);
                // b = HM * Ih
                filters_[k]->apply(a_[k], b_[k], streams_[k]);
                // c = DHF * Ih
                cuda::resize(b_[k], c_[k], lowResSize, 0, 0, INTER_NEAREST, streams_[k]);

                diffSign(src[k], c_[k], c_[k], streams_[k]);

                // a = Dt * diff
                upscale(c_[k], a_[k], scale_, streams_[k]);
                // b = HtDt * diff
                filters_[k]->apply(a_[k], b_[k], streams_[k]);
                // diffTerm = MtHtDt * diff
                cuda::remap(b_[k], diffTerms_[k], forwardMaps_[k].first, forwardMaps_[k].second, INTER_NEAREST, BORDER_REPLICATE, Scalar(), streams_[k]);
            }

            if (lambda_ > 0)
            {
                calcBtvRegularization(highRes_, regTerm_, btvKernelSize_);
                cuda::addWeighted(highRes_, 1.0, regTerm_, -tau_ * lambda_, 0.0, highRes_);
            }

            for (size_t k = 0; k < src.size(); ++k)
            {
                streams_[k].waitForCompletion();
                cuda::addWeighted(highRes_, 1.0, diffTerms_[k], tau_, 0.0, highRes_);
            }
        }

        Rect inner(btvKernelSize_, btvKernelSize_, highRes_.cols - 2 * btvKernelSize_, highRes_.rows - 2 * btvKernelSize_);
        highRes_(inner).copyTo(dst);
    }

    void BTVL1_CUDA_Base::collectGarbage()
    {
        filters_.clear();

        lowResForwardMotions_.clear();
        lowResBackwardMotions_.clear();

        highResForwardMotions_.clear();
        highResBackwardMotions_.clear();

        forwardMaps_.clear();
        backwardMaps_.clear();

        highRes_.release();

        diffTerms_.clear();
        a_.clear();
        b_.clear();
        c_.clear();
        regTerm_.release();
    }

////////////////////////////////////////////////////////////

    class BTVL1_CUDA : public SuperResolution, private BTVL1_CUDA_Base
    {
    public:
        AlgorithmInfo* info() const;

        BTVL1_CUDA();

        void collectGarbage();

    protected:
        void initImpl(Ptr<FrameSource>& frameSource);
        void processImpl(Ptr<FrameSource>& frameSource, OutputArray output);

    private:
        int temporalAreaRadius_;

        void readNextFrame(Ptr<FrameSource>& frameSource);
        void processFrame(int idx);

        GpuMat curFrame_;
        GpuMat prevFrame_;

        std::vector<GpuMat> frames_;
        std::vector<std::pair<GpuMat, GpuMat> > forwardMotions_;
        std::vector<std::pair<GpuMat, GpuMat> > backwardMotions_;
        std::vector<GpuMat> outputs_;

        int storePos_;
        int procPos_;
        int outPos_;

        std::vector<GpuMat> srcFrames_;
        std::vector<std::pair<GpuMat, GpuMat> > srcForwardMotions_;
        std::vector<std::pair<GpuMat, GpuMat> > srcBackwardMotions_;
        GpuMat finalOutput_;
    };

    CV_INIT_ALGORITHM(BTVL1_CUDA, "SuperResolution.BTVL1_CUDA",
                      obj.info()->addParam(obj, "scale", obj.scale_, false, 0, 0, "Scale factor.");
                      obj.info()->addParam(obj, "iterations", obj.iterations_, false, 0, 0, "Iteration count.");
                      obj.info()->addParam(obj, "tau", obj.tau_, false, 0, 0, "Asymptotic value of steepest descent method.");
                      obj.info()->addParam(obj, "lambda", obj.lambda_, false, 0, 0, "Weight parameter to balance data term and smoothness term.");
                      obj.info()->addParam(obj, "alpha", obj.alpha_, false, 0, 0, "Parameter of spacial distribution in Bilateral-TV.");
                      obj.info()->addParam(obj, "btvKernelSize", obj.btvKernelSize_, false, 0, 0, "Kernel size of Bilateral-TV filter.");
                      obj.info()->addParam(obj, "blurKernelSize", obj.blurKernelSize_, false, 0, 0, "Gaussian blur kernel size.");
                      obj.info()->addParam(obj, "blurSigma", obj.blurSigma_, false, 0, 0, "Gaussian blur sigma.");
                      obj.info()->addParam(obj, "temporalAreaRadius", obj.temporalAreaRadius_, false, 0, 0, "Radius of the temporal search area.");
                      obj.info()->addParam<DenseOpticalFlowExt>(obj, "opticalFlow", obj.opticalFlow_, false, 0, 0, "Dense optical flow algorithm."));

    BTVL1_CUDA::BTVL1_CUDA()
    {
        temporalAreaRadius_ = 4;
    }

    void BTVL1_CUDA::collectGarbage()
    {
        curFrame_.release();
        prevFrame_.release();

        frames_.clear();
        forwardMotions_.clear();
        backwardMotions_.clear();
        outputs_.clear();

        srcFrames_.clear();
        srcForwardMotions_.clear();
        srcBackwardMotions_.clear();
        finalOutput_.release();

        SuperResolution::collectGarbage();
        BTVL1_CUDA_Base::collectGarbage();
    }

    void BTVL1_CUDA::initImpl(Ptr<FrameSource>& frameSource)
    {
        const int cacheSize = 2 * temporalAreaRadius_ + 1;

        frames_.resize(cacheSize);
        forwardMotions_.resize(cacheSize);
        backwardMotions_.resize(cacheSize);
        outputs_.resize(cacheSize);

        storePos_ = -1;

        for (int t = -temporalAreaRadius_; t <= temporalAreaRadius_; ++t)
            readNextFrame(frameSource);

        for (int i = 0; i <= temporalAreaRadius_; ++i)
            processFrame(i);

        procPos_ = temporalAreaRadius_;
        outPos_ = -1;
    }

    void BTVL1_CUDA::processImpl(Ptr<FrameSource>& frameSource, OutputArray _output)
    {
        if (outPos_ >= storePos_)
        {
            _output.release();
            return;
        }

        readNextFrame(frameSource);

        if (procPos_ < storePos_)
        {
            ++procPos_;
            processFrame(procPos_);
        }

        ++outPos_;
        const GpuMat& curOutput = at(outPos_, outputs_);

        if (_output.kind() == _InputArray::GPU_MAT)
            curOutput.convertTo(_output.getGpuMatRef(), CV_8U);
        else
        {
            curOutput.convertTo(finalOutput_, CV_8U);
            arrCopy(finalOutput_, _output);
        }
    }

    void BTVL1_CUDA::readNextFrame(Ptr<FrameSource>& frameSource)
    {
        frameSource->nextFrame(curFrame_);

        if (curFrame_.empty())
            return;

        ++storePos_;
        curFrame_.convertTo(at(storePos_, frames_), CV_32F);

        if (storePos_ > 0)
        {
            std::pair<GpuMat, GpuMat>& forwardMotion = at(storePos_ - 1, forwardMotions_);
            std::pair<GpuMat, GpuMat>& backwardMotion = at(storePos_, backwardMotions_);

            opticalFlow_->calc(prevFrame_, curFrame_, forwardMotion.first, forwardMotion.second);
            opticalFlow_->calc(curFrame_, prevFrame_, backwardMotion.first, backwardMotion.second);
        }

        curFrame_.copyTo(prevFrame_);
    }

    void BTVL1_CUDA::processFrame(int idx)
    {
        const int startIdx = std::max(idx - temporalAreaRadius_, 0);
        const int procIdx = idx;
        const int endIdx = std::min(startIdx + 2 * temporalAreaRadius_, storePos_);

        const int count = endIdx - startIdx + 1;

        srcFrames_.resize(count);
        srcForwardMotions_.resize(count);
        srcBackwardMotions_.resize(count);

        int baseIdx = -1;

        for (int i = startIdx, k = 0; i <= endIdx; ++i, ++k)
        {
            if (i == procIdx)
                baseIdx = k;

            srcFrames_[k] = at(i, frames_);

            if (i < endIdx)
                srcForwardMotions_[k] = at(i, forwardMotions_);
            if (i > startIdx)
                srcBackwardMotions_[k] = at(i, backwardMotions_);
        }

        process(srcFrames_, at(idx, outputs_), srcForwardMotions_, srcBackwardMotions_, baseIdx);
    }
}

Ptr<SuperResolution> cv::superres::createSuperResolution_BTVL1_CUDA()
{
    return makePtr<BTVL1_CUDA>();
}

#endif // HAVE_CUDA
