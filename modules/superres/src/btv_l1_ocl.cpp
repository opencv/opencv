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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//		Jin Ma, jin@multicorewareinc.com
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

#if !defined(HAVE_OPENCL) || !defined(HAVE_OPENCV_OCL)

cv::Ptr<cv::superres::SuperResolution> cv::superres::createSuperResolution_BTVL1_OCL()
{
    CV_Error(cv::Error::StsNotImplemented, "The called functionality is disabled for current build or platform");
    return Ptr<SuperResolution>();
}

#else
#include "opencl_kernels.hpp"

using namespace std;
using namespace cv;
using namespace cv::ocl;
using namespace cv::superres;
using namespace cv::superres::detail;

namespace cv
{
    namespace ocl
    {
        float* btvWeights_ = NULL;
        size_t btvWeights_size = 0;
        oclMat c_btvRegWeights;
    }
}

namespace btv_l1_device_ocl
{
    void buildMotionMaps(const oclMat& forwardMotionX, const oclMat& forwardMotionY,
        const oclMat& backwardMotionX, const oclMat& bacwardMotionY,
        oclMat& forwardMapX, oclMat& forwardMapY,
        oclMat& backwardMapX, oclMat& backwardMapY);

    void upscale(const oclMat& src, oclMat& dst, int scale);

    void diffSign(const oclMat& src1, const oclMat& src2, oclMat& dst);

    void calcBtvRegularization(const oclMat& src, oclMat& dst, int ksize);
}

void btv_l1_device_ocl::buildMotionMaps(const oclMat& forwardMotionX, const oclMat& forwardMotionY,
    const oclMat& backwardMotionX, const oclMat& backwardMotionY,
    oclMat& forwardMapX, oclMat& forwardMapY,
    oclMat& backwardMapX, oclMat& backwardMapY)
{
    Context* clCxt = Context::getContext();

    size_t local_thread[] = {32, 8, 1};
    size_t global_thread[] = {forwardMapX.cols, forwardMapX.rows, 1};

    int forwardMotionX_step = (int)(forwardMotionX.step/forwardMotionX.elemSize());
    int forwardMotionY_step = (int)(forwardMotionY.step/forwardMotionY.elemSize());
    int backwardMotionX_step = (int)(backwardMotionX.step/backwardMotionX.elemSize());
    int backwardMotionY_step = (int)(backwardMotionY.step/backwardMotionY.elemSize());
    int forwardMapX_step = (int)(forwardMapX.step/forwardMapX.elemSize());
    int forwardMapY_step = (int)(forwardMapY.step/forwardMapY.elemSize());
    int backwardMapX_step = (int)(backwardMapX.step/backwardMapX.elemSize());
    int backwardMapY_step = (int)(backwardMapY.step/backwardMapY.elemSize());

    String kernel_name = "buildMotionMapsKernel";
    vector< pair<size_t, const void*> > args;

    args.push_back(make_pair(sizeof(cl_mem), (void*)&forwardMotionX.data));
    args.push_back(make_pair(sizeof(cl_mem), (void*)&forwardMotionY.data));
    args.push_back(make_pair(sizeof(cl_mem), (void*)&backwardMotionX.data));
    args.push_back(make_pair(sizeof(cl_mem), (void*)&backwardMotionY.data));
    args.push_back(make_pair(sizeof(cl_mem), (void*)&forwardMapX.data));
    args.push_back(make_pair(sizeof(cl_mem), (void*)&forwardMapY.data));
    args.push_back(make_pair(sizeof(cl_mem), (void*)&backwardMapX.data));
    args.push_back(make_pair(sizeof(cl_mem), (void*)&backwardMapY.data));

    args.push_back(make_pair(sizeof(cl_int), (void*)&forwardMotionX.rows));
    args.push_back(make_pair(sizeof(cl_int), (void*)&forwardMotionY.cols));

    args.push_back(make_pair(sizeof(cl_int), (void*)&forwardMotionX_step));
    args.push_back(make_pair(sizeof(cl_int), (void*)&forwardMotionY_step));
    args.push_back(make_pair(sizeof(cl_int), (void*)&backwardMotionX_step));
    args.push_back(make_pair(sizeof(cl_int), (void*)&backwardMotionY_step));
    args.push_back(make_pair(sizeof(cl_int), (void*)&forwardMapX_step));
    args.push_back(make_pair(sizeof(cl_int), (void*)&forwardMapY_step));
    args.push_back(make_pair(sizeof(cl_int), (void*)&backwardMapX_step));
    args.push_back(make_pair(sizeof(cl_int), (void*)&backwardMapY_step));

    openCLExecuteKernel(clCxt, &superres_btvl1, kernel_name, global_thread, local_thread, args, -1, -1);
}

void btv_l1_device_ocl::upscale(const oclMat& src, oclMat& dst, int scale)
{
    Context* clCxt = Context::getContext();

    size_t local_thread[] = {32, 8, 1};
    size_t global_thread[] = {src.cols, src.rows, 1};

    int src_step = (int)(src.step/src.elemSize());
    int dst_step = (int)(dst.step/dst.elemSize());

    String kernel_name = "upscaleKernel";
    vector< pair<size_t, const void*> > args;

    int cn = src.oclchannels();

    args.push_back(make_pair(sizeof(cl_mem), (void*)&src.data));
    args.push_back(make_pair(sizeof(cl_mem), (void*)&dst.data));
    args.push_back(make_pair(sizeof(cl_int), (void*)&src_step));
    args.push_back(make_pair(sizeof(cl_int), (void*)&dst_step));
    args.push_back(make_pair(sizeof(cl_int), (void*)&src.rows));
    args.push_back(make_pair(sizeof(cl_int), (void*)&src.cols));
    args.push_back(make_pair(sizeof(cl_int), (void*)&scale));
    args.push_back(make_pair(sizeof(cl_int), (void*)&cn));

    openCLExecuteKernel(clCxt, &superres_btvl1, kernel_name, global_thread, local_thread, args, -1, -1);

}

void btv_l1_device_ocl::diffSign(const oclMat& src1, const oclMat& src2, oclMat& dst)
{
    Context* clCxt = Context::getContext();

    oclMat src1_ = src1.reshape(1);
    oclMat src2_ = src2.reshape(1);
    oclMat dst_ = dst.reshape(1);

    int src1_step = (int)(src1_.step/src1_.elemSize());
    int src2_step = (int)(src2_.step/src2_.elemSize());
    int dst_step = (int)(dst_.step/dst_.elemSize());

    size_t local_thread[] = {32, 8, 1};
    size_t global_thread[] = {src1_.cols, src1_.rows, 1};

    String kernel_name = "diffSignKernel";
    vector< pair<size_t, const void*> > args;

    args.push_back(make_pair(sizeof(cl_mem), (void*)&src1_.data));
    args.push_back(make_pair(sizeof(cl_mem), (void*)&src2_.data));
    args.push_back(make_pair(sizeof(cl_mem), (void*)&dst_.data));

    args.push_back(make_pair(sizeof(cl_int), (void*)&src1_.rows));
    args.push_back(make_pair(sizeof(cl_int), (void*)&src1_.cols));
    args.push_back(make_pair(sizeof(cl_int), (void*)&dst_step));
    args.push_back(make_pair(sizeof(cl_int), (void*)&src1_step));
    args.push_back(make_pair(sizeof(cl_int), (void*)&src2_step));

    openCLExecuteKernel(clCxt, &superres_btvl1, kernel_name, global_thread, local_thread, args, -1, -1);
}

void btv_l1_device_ocl::calcBtvRegularization(const oclMat& src, oclMat& dst, int ksize)
{
    Context* clCxt = Context::getContext();

    oclMat src_ = src.reshape(1);
    oclMat dst_ = dst.reshape(1);

    size_t local_thread[] = {32, 8, 1};
    size_t global_thread[] = {src.cols, src.rows, 1};

    int src_step = (int)(src_.step/src_.elemSize());
    int dst_step = (int)(dst_.step/dst_.elemSize());

    String kernel_name = "calcBtvRegularizationKernel";
    vector< pair<size_t, const void*> > args;

    int cn = src.oclchannels();

    args.push_back(make_pair(sizeof(cl_mem), (void*)&src_.data));
    args.push_back(make_pair(sizeof(cl_mem), (void*)&dst_.data));
    args.push_back(make_pair(sizeof(cl_int), (void*)&src_step));
    args.push_back(make_pair(sizeof(cl_int), (void*)&dst_step));
    args.push_back(make_pair(sizeof(cl_int), (void*)&src.rows));
    args.push_back(make_pair(sizeof(cl_int), (void*)&src.cols));
    args.push_back(make_pair(sizeof(cl_int), (void*)&ksize));
    args.push_back(make_pair(sizeof(cl_int), (void*)&cn));
    args.push_back(make_pair(sizeof(cl_mem), (void*)&c_btvRegWeights.data));

    openCLExecuteKernel(clCxt, &superres_btvl1, kernel_name, global_thread, local_thread, args, -1, -1);
}

namespace
{
    void calcRelativeMotions(const vector<pair<oclMat, oclMat> >& forwardMotions, const vector<pair<oclMat, oclMat> >& backwardMotions,
        vector<pair<oclMat, oclMat> >& relForwardMotions, vector<pair<oclMat, oclMat> >& relBackwardMotions,
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
            ocl::add(relForwardMotions[i + 1].first, forwardMotions[i].first, relForwardMotions[i].first);
            ocl::add(relForwardMotions[i + 1].second, forwardMotions[i].second, relForwardMotions[i].second);

            ocl::add(relBackwardMotions[i + 1].first, backwardMotions[i + 1].first, relBackwardMotions[i].first);
            ocl::add(relBackwardMotions[i + 1].second, backwardMotions[i + 1].second, relBackwardMotions[i].second);
        }

        for (int i = baseIdx + 1; i < count; ++i)
        {
            ocl::add(relForwardMotions[i - 1].first, backwardMotions[i].first, relForwardMotions[i].first);
            ocl::add(relForwardMotions[i - 1].second, backwardMotions[i].second, relForwardMotions[i].second);

            ocl::add(relBackwardMotions[i - 1].first, forwardMotions[i - 1].first, relBackwardMotions[i].first);
            ocl::add(relBackwardMotions[i - 1].second, forwardMotions[i - 1].second, relBackwardMotions[i].second);
        }
    }

    void upscaleMotions(const vector<pair<oclMat, oclMat> >& lowResMotions, vector<pair<oclMat, oclMat> >& highResMotions, int scale)
    {
        highResMotions.resize(lowResMotions.size());

        for (size_t i = 0; i < lowResMotions.size(); ++i)
        {
            ocl::resize(lowResMotions[i].first, highResMotions[i].first, Size(), scale, scale, INTER_LINEAR);
            ocl::resize(lowResMotions[i].second, highResMotions[i].second, Size(), scale, scale, INTER_LINEAR);

            ocl::multiply(scale, highResMotions[i].first, highResMotions[i].first);
            ocl::multiply(scale, highResMotions[i].second, highResMotions[i].second);
        }
    }

    void buildMotionMaps(const pair<oclMat, oclMat>& forwardMotion, const pair<oclMat, oclMat>& backwardMotion,
        pair<oclMat, oclMat>& forwardMap, pair<oclMat, oclMat>& backwardMap)
    {
        forwardMap.first.create(forwardMotion.first.size(), CV_32FC1);
        forwardMap.second.create(forwardMotion.first.size(), CV_32FC1);

        backwardMap.first.create(forwardMotion.first.size(), CV_32FC1);
        backwardMap.second.create(forwardMotion.first.size(), CV_32FC1);

        btv_l1_device_ocl::buildMotionMaps(forwardMotion.first, forwardMotion.second,
            backwardMotion.first, backwardMotion.second,
            forwardMap.first, forwardMap.second,
            backwardMap.first, backwardMap.second);
    }

    void upscale(const oclMat& src, oclMat& dst, int scale)
    {
        CV_Assert( src.channels() == 1 || src.channels() == 3 || src.channels() == 4 );

        btv_l1_device_ocl::upscale(src, dst, scale);
    }

    void diffSign(const oclMat& src1, const oclMat& src2, oclMat& dst)
    {
        dst.create(src1.size(), src1.type());

        btv_l1_device_ocl::diffSign(src1, src2, dst);
    }

    void calcBtvWeights(int btvKernelSize, double alpha, vector<float>& btvWeights)
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

        btvWeights_ = &btvWeights[0];
        btvWeights_size = size;
        Mat btvWeights_mheader(1, static_cast<int>(size), CV_32FC1, btvWeights_);
        c_btvRegWeights = btvWeights_mheader;
    }

    void calcBtvRegularization(const oclMat& src, oclMat& dst, int btvKernelSize)
    {
        dst.create(src.size(), src.type());

        const int ksize = (btvKernelSize - 1) / 2;

        btv_l1_device_ocl::calcBtvRegularization(src, dst, ksize);
    }

    class BTVL1_OCL_Base
    {
    public:
        BTVL1_OCL_Base();

        void process(const vector<oclMat>& src, oclMat& dst,
            const vector<pair<oclMat, oclMat> >& forwardMotions, const vector<pair<oclMat, oclMat> >& backwardMotions,
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
        vector<Ptr<cv::ocl::FilterEngine_GPU> > filters_;
        int curBlurKernelSize_;
        double curBlurSigma_;
        int curSrcType_;

        vector<float> btvWeights_;
        int curBtvKernelSize_;
        double curAlpha_;

        vector<pair<oclMat, oclMat> > lowResForwardMotions_;
        vector<pair<oclMat, oclMat> > lowResBackwardMotions_;

        vector<pair<oclMat, oclMat> > highResForwardMotions_;
        vector<pair<oclMat, oclMat> > highResBackwardMotions_;

        vector<pair<oclMat, oclMat> > forwardMaps_;
        vector<pair<oclMat, oclMat> > backwardMaps_;

        oclMat highRes_;

        vector<oclMat> diffTerms_;
        oclMat a_, b_, c_, d_;
        oclMat regTerm_;
    };

    BTVL1_OCL_Base::BTVL1_OCL_Base()
    {
        scale_ = 4;
        iterations_ = 180;
        lambda_ = 0.03;
        tau_ = 1.3;
        alpha_ = 0.7;
        btvKernelSize_ = 7;
        blurKernelSize_ = 5;
        blurSigma_ = 0.0;
        opticalFlow_ = createOptFlow_Farneback_OCL();

        curBlurKernelSize_ = -1;
        curBlurSigma_ = -1.0;
        curSrcType_ = -1;

        curBtvKernelSize_ = -1;
        curAlpha_ = -1.0;
    }

    void BTVL1_OCL_Base::process(const vector<oclMat>& src, oclMat& dst,
        const vector<pair<oclMat, oclMat> >& forwardMotions, const vector<pair<oclMat, oclMat> >& backwardMotions,
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
                filters_[i] = cv::ocl::createGaussianFilter_GPU(src[0].type(), Size(blurKernelSize_, blurKernelSize_), blurSigma_);
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

        calcRelativeMotions(forwardMotions, backwardMotions,
            lowResForwardMotions_, lowResBackwardMotions_,
            baseIdx, src[0].size());

        upscaleMotions(lowResForwardMotions_, highResForwardMotions_, scale_);
        upscaleMotions(lowResBackwardMotions_, highResBackwardMotions_, scale_);

        forwardMaps_.resize(highResForwardMotions_.size());
        backwardMaps_.resize(highResForwardMotions_.size());
        for (size_t i = 0; i < highResForwardMotions_.size(); ++i)
        {
            buildMotionMaps(highResForwardMotions_[i], highResBackwardMotions_[i], forwardMaps_[i], backwardMaps_[i]);
        }
        // initial estimation

        const Size lowResSize = src[0].size();
        const Size highResSize(lowResSize.width * scale_, lowResSize.height * scale_);

        ocl::resize(src[baseIdx], highRes_, highResSize, 0, 0, INTER_LINEAR);

        // iterations

        diffTerms_.resize(src.size());
        bool d_inited = false;
        a_.create(highRes_.size(), highRes_.type());
        b_.create(highRes_.size(), highRes_.type());
        c_.create(lowResSize, highRes_.type());
        d_.create(highRes_.rows, highRes_.cols, highRes_.type());
        for (int i = 0; i < iterations_; ++i)
        {
            if(!d_inited)
            {
                d_.setTo(0);
                d_inited = true;
            }
            for (size_t k = 0; k < src.size(); ++k)
            {
                diffTerms_[k].create(highRes_.size(), highRes_.type());
                // a = M * Ih
                ocl::remap(highRes_, a_, backwardMaps_[k].first, backwardMaps_[k].second, INTER_NEAREST, BORDER_CONSTANT, Scalar());
                // b = HM * Ih
                filters_[k]->apply(a_, b_, Rect(0,0,-1,-1));
                // c = DHF * Ih
                ocl::resize(b_, c_, lowResSize, 0, 0, INTER_NEAREST);

                diffSign(src[k], c_, c_);

                // a = Dt * diff
                upscale(c_, d_, scale_);
                // b = HtDt * diff
                filters_[k]->apply(d_, b_, Rect(0,0,-1,-1));
                // diffTerm = MtHtDt * diff
                ocl::remap(b_, diffTerms_[k], forwardMaps_[k].first, forwardMaps_[k].second, INTER_NEAREST, BORDER_CONSTANT, Scalar());
            }

            if (lambda_ > 0)
            {
                calcBtvRegularization(highRes_, regTerm_, btvKernelSize_);
                ocl::addWeighted(highRes_, 1.0, regTerm_, -tau_ * lambda_, 0.0, highRes_);
            }

            for (size_t k = 0; k < src.size(); ++k)
            {
                ocl::addWeighted(highRes_, 1.0, diffTerms_[k], tau_, 0.0, highRes_);
            }
        }

        Rect inner(btvKernelSize_, btvKernelSize_, highRes_.cols - 2 * btvKernelSize_, highRes_.rows - 2 * btvKernelSize_);
        highRes_(inner).copyTo(dst);
    }

    void BTVL1_OCL_Base::collectGarbage()
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
        a_.release();
        b_.release();
        c_.release();
        regTerm_.release();
        c_btvRegWeights.release();
    }

    ////////////////////////////////////////////////////////////

    class BTVL1_OCL : public SuperResolution, private BTVL1_OCL_Base
    {
    public:
        AlgorithmInfo* info() const;

        BTVL1_OCL();

        void collectGarbage();

    protected:
        void initImpl(Ptr<FrameSource>& frameSource);
        void processImpl(Ptr<FrameSource>& frameSource, OutputArray output);

    private:
        int temporalAreaRadius_;

        void readNextFrame(Ptr<FrameSource>& frameSource);
        void processFrame(int idx);

        oclMat curFrame_;
        oclMat prevFrame_;

        vector<oclMat> frames_;
        vector<pair<oclMat, oclMat> > forwardMotions_;
        vector<pair<oclMat, oclMat> > backwardMotions_;
        vector<oclMat> outputs_;

        int storePos_;
        int procPos_;
        int outPos_;

        vector<oclMat> srcFrames_;
        vector<pair<oclMat, oclMat> > srcForwardMotions_;
        vector<pair<oclMat, oclMat> > srcBackwardMotions_;
        oclMat finalOutput_;
    };

    CV_INIT_ALGORITHM(BTVL1_OCL, "SuperResolution.BTVL1_OCL",
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

    BTVL1_OCL::BTVL1_OCL()
    {
        temporalAreaRadius_ = 4;
    }

    void BTVL1_OCL::collectGarbage()
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
        BTVL1_OCL_Base::collectGarbage();
    }

    void BTVL1_OCL::initImpl(Ptr<FrameSource>& frameSource)
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

    void BTVL1_OCL::processImpl(Ptr<FrameSource>& frameSource, OutputArray _output)
    {
        if (outPos_ >= storePos_)
        {
            if(_output.kind() == _InputArray::OCL_MAT)
            {
                getOclMatRef(_output).release();
            }
            else
            {
                _output.release();
            }
            return;
        }

        readNextFrame(frameSource);

        if (procPos_ < storePos_)
        {
            ++procPos_;
            processFrame(procPos_);
        }

        ++outPos_;
        const oclMat& curOutput = at(outPos_, outputs_);

        if (_output.kind() == _InputArray::OCL_MAT)
            curOutput.convertTo(getOclMatRef(_output), CV_8U);
        else
        {
            curOutput.convertTo(finalOutput_, CV_8U);
            arrCopy(finalOutput_, _output);
        }
    }

    void BTVL1_OCL::readNextFrame(Ptr<FrameSource>& frameSource)
    {
        curFrame_.release();
        frameSource->nextFrame(curFrame_);

        if (curFrame_.empty())
            return;

        ++storePos_;
        curFrame_.convertTo(at(storePos_, frames_), CV_32F);

        if (storePos_ > 0)
        {
            pair<oclMat, oclMat>& forwardMotion = at(storePos_ - 1, forwardMotions_);
            pair<oclMat, oclMat>& backwardMotion = at(storePos_, backwardMotions_);

            opticalFlow_->calc(prevFrame_, curFrame_, forwardMotion.first, forwardMotion.second);
            opticalFlow_->calc(curFrame_, prevFrame_, backwardMotion.first, backwardMotion.second);
        }

        curFrame_.copyTo(prevFrame_);
    }

    void BTVL1_OCL::processFrame(int idx)
    {
        const int startIdx = max(idx - temporalAreaRadius_, 0);
        const int procIdx = idx;
        const int endIdx = min(startIdx + 2 * temporalAreaRadius_, storePos_);

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

Ptr<SuperResolution> cv::superres::createSuperResolution_BTVL1_OCL()
{
    return makePtr<BTVL1_OCL>();
}
#endif
