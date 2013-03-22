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

using namespace std;
using namespace cv;
using namespace cv::superres;
using namespace cv::superres::detail;

namespace
{
    void calcRelativeMotions(const vector<Mat>& forwardMotions, const vector<Mat>& backwardMotions,
                             vector<Mat>& relForwardMotions, vector<Mat>& relBackwardMotions,
                             int baseIdx, Size size)
    {
        const int count = static_cast<int>(forwardMotions.size());

        relForwardMotions.resize(count);
        relForwardMotions[baseIdx].create(size, CV_32FC2);
        relForwardMotions[baseIdx].setTo(Scalar::all(0));

        relBackwardMotions.resize(count);
        relBackwardMotions[baseIdx].create(size, CV_32FC2);
        relBackwardMotions[baseIdx].setTo(Scalar::all(0));

        for (int i = baseIdx - 1; i >= 0; --i)
        {
            add(relForwardMotions[i + 1], forwardMotions[i], relForwardMotions[i]);

            add(relBackwardMotions[i + 1], backwardMotions[i + 1], relBackwardMotions[i]);
        }

        for (int i = baseIdx + 1; i < count; ++i)
        {
            add(relForwardMotions[i - 1], backwardMotions[i], relForwardMotions[i]);

            add(relBackwardMotions[i - 1], forwardMotions[i - 1], relBackwardMotions[i]);
        }
    }

    void upscaleMotions(const vector<Mat>& lowResMotions, vector<Mat>& highResMotions, int scale)
    {
        highResMotions.resize(lowResMotions.size());

        for (size_t i = 0; i < lowResMotions.size(); ++i)
        {
            resize(lowResMotions[i], highResMotions[i], Size(), scale, scale, INTER_CUBIC);
            multiply(highResMotions[i], Scalar::all(scale), highResMotions[i]);
        }
    }

    void buildMotionMaps(const Mat& forwardMotion, const Mat& backwardMotion, Mat& forwardMap, Mat& backwardMap)
    {
        forwardMap.create(forwardMotion.size(), CV_32FC2);
        backwardMap.create(forwardMotion.size(), CV_32FC2);

        for (int y = 0; y < forwardMotion.rows; ++y)
        {
            const Point2f* forwardMotionRow = forwardMotion.ptr<Point2f>(y);
            const Point2f* backwardMotionRow = backwardMotion.ptr<Point2f>(y);
            Point2f* forwardMapRow = forwardMap.ptr<Point2f>(y);
            Point2f* backwardMapRow = backwardMap.ptr<Point2f>(y);

            for (int x = 0; x < forwardMotion.cols; ++x)
            {
                Point2f base(static_cast<float>(x), static_cast<float>(y));

                forwardMapRow[x] = base + backwardMotionRow[x];
                backwardMapRow[x] = base + forwardMotionRow[x];
            }
        }
    }

    template <typename T>
    void upscaleImpl(const Mat& src, Mat& dst, int scale)
    {
        dst.create(src.rows * scale, src.cols * scale, src.type());
        dst.setTo(Scalar::all(0));

        for (int y = 0, Y = 0; y < src.rows; ++y, Y += scale)
        {
            const T* srcRow = src.ptr<T>(y);
            T* dstRow = dst.ptr<T>(Y);

            for (int x = 0, X = 0; x < src.cols; ++x, X += scale)
                dstRow[X] = srcRow[x];
        }
    }

    void upscale(const Mat& src, Mat& dst, int scale)
    {
        typedef void (*func_t)(const Mat& src, Mat& dst, int scale);
        static const func_t funcs[] =
        {
            0, upscaleImpl<float>, 0, upscaleImpl<Point3f>
        };

        CV_Assert( src.channels() == 1 || src.channels() == 3 || src.channels() == 4 );

        const func_t func = funcs[src.channels()];

        func(src, dst, scale);
    }

    float diffSign(float a, float b)
    {
        return a > b ? 1.0f : a < b ? -1.0f : 0.0f;
    }
    Point3f diffSign(Point3f a, Point3f b)
    {
        return Point3f(
            a.x > b.x ? 1.0f : a.x < b.x ? -1.0f : 0.0f,
            a.y > b.y ? 1.0f : a.y < b.y ? -1.0f : 0.0f,
            a.z > b.z ? 1.0f : a.z < b.z ? -1.0f : 0.0f
        );
    }

    void diffSign(const Mat& src1, const Mat& src2, Mat& dst)
    {
        const int count = src1.cols * src1.channels();

        dst.create(src1.size(), src1.type());

        for (int y = 0; y < src1.rows; ++y)
        {
            const float* src1Ptr = src1.ptr<float>(y);
            const float* src2Ptr = src2.ptr<float>(y);
            float* dstPtr = dst.ptr<float>(y);

            for (int x = 0; x < count; ++x)
                dstPtr[x] = diffSign(src1Ptr[x], src2Ptr[x]);
        }
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
    }

    template <typename T>
    struct BtvRegularizationBody : ParallelLoopBody
    {
        void operator ()(const Range& range) const;

        Mat src;
        mutable Mat dst;
        int ksize;
        const float* btvWeights;
    };

    template <typename T>
    void BtvRegularizationBody<T>::operator ()(const Range& range) const
    {
        for (int i = range.start; i < range.end; ++i)
        {
            const T* srcRow = src.ptr<T>(i);
            T* dstRow = dst.ptr<T>(i);

            for(int j = ksize; j < src.cols - ksize; ++j)
            {
                const T srcVal = srcRow[j];

                for (int m = 0, ind = 0; m <= ksize; ++m)
                {
                    const T* srcRow2 = src.ptr<T>(i - m);
                    const T* srcRow3 = src.ptr<T>(i + m);

                    for (int l = ksize; l + m >= 0; --l, ++ind)
                    {
                        dstRow[j] += btvWeights[ind] * (diffSign(srcVal, srcRow3[j + l]) - diffSign(srcRow2[j - l], srcVal));
                    }
                }
            }
        }
    }

    template <typename T>
    void calcBtvRegularizationImpl(const Mat& src, Mat& dst, int btvKernelSize, const vector<float>& btvWeights)
    {
        dst.create(src.size(), src.type());
        dst.setTo(Scalar::all(0));

        const int ksize = (btvKernelSize - 1) / 2;

        BtvRegularizationBody<T> body;

        body.src = src;
        body.dst = dst;
        body.ksize = ksize;
        body.btvWeights = &btvWeights[0];

        parallel_for_(Range(ksize, src.rows - ksize), body);
    }

    void calcBtvRegularization(const Mat& src, Mat& dst, int btvKernelSize, const vector<float>& btvWeights)
    {
        typedef void (*func_t)(const Mat& src, Mat& dst, int btvKernelSize, const vector<float>& btvWeights);
        static const func_t funcs[] =
        {
            0, calcBtvRegularizationImpl<float>, 0, calcBtvRegularizationImpl<Point3f>
        };

        const func_t func = funcs[src.channels()];

        func(src, dst, btvKernelSize, btvWeights);
    }

    class BTVL1_Base
    {
    public:
        BTVL1_Base();

        void process(const vector<Mat>& src, Mat& dst,
                     const vector<Mat>& forwardMotions, const vector<Mat>& backwardMotions,
                     int baseIdx);

        void collectGarbage();

    protected:
        int scale_;
        int iterations_;
        double tau_;
        double lambda_;
        double alpha_;
        int btvKernelSize_;
        int blurKernelSize_;
        double blurSigma_;
        Ptr<DenseOpticalFlowExt> opticalFlow_;

    private:
        Ptr<FilterEngine> filter_;
        int curBlurKernelSize_;
        double curBlurSigma_;
        int curSrcType_;

        vector<float> btvWeights_;
        int curBtvKernelSize_;
        double curAlpha_;

        vector<Mat> lowResForwardMotions_;
        vector<Mat> lowResBackwardMotions_;

        vector<Mat> highResForwardMotions_;
        vector<Mat> highResBackwardMotions_;

        vector<Mat> forwardMaps_;
        vector<Mat> backwardMaps_;

        Mat highRes_;

        Mat diffTerm_, regTerm_;
        Mat a_, b_, c_;
    };

    BTVL1_Base::BTVL1_Base()
    {
        scale_ = 4;
        iterations_ = 180;
        lambda_ = 0.03;
        tau_ = 1.3;
        alpha_ = 0.7;
        btvKernelSize_ = 7;
        blurKernelSize_ = 5;
        blurSigma_ = 0.0;
        opticalFlow_ = createOptFlow_Farneback();

        curBlurKernelSize_ = -1;
        curBlurSigma_ = -1.0;
        curSrcType_ = -1;

        curBtvKernelSize_ = -1;
        curAlpha_ = -1.0;
    }

    void BTVL1_Base::process(const vector<Mat>& src, Mat& dst, const vector<Mat>& forwardMotions, const vector<Mat>& backwardMotions, int baseIdx)
    {
        CV_Assert( scale_ > 1 );
        CV_Assert( iterations_ > 0 );
        CV_Assert( tau_ > 0.0 );
        CV_Assert( alpha_ > 0.0 );
        CV_Assert( btvKernelSize_ > 0 );
        CV_Assert( blurKernelSize_ > 0 );
        CV_Assert( blurSigma_ >= 0.0 );

        // update blur filter and btv weights

        if (filter_.empty() || blurKernelSize_ != curBlurKernelSize_ || blurSigma_ != curBlurSigma_ || src[0].type() != curSrcType_)
        {
            filter_ = createGaussianFilter(src[0].type(), Size(blurKernelSize_, blurKernelSize_), blurSigma_);
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

        // calc high res motions

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

        resize(src[baseIdx], highRes_, highResSize, 0, 0, INTER_CUBIC);

        // iterations

        diffTerm_.create(highResSize, highRes_.type());
        a_.create(highResSize, highRes_.type());
        b_.create(highResSize, highRes_.type());
        c_.create(lowResSize, highRes_.type());

        for (int i = 0; i < iterations_; ++i)
        {
            diffTerm_.setTo(Scalar::all(0));

            for (size_t k = 0; k < src.size(); ++k)
            {
                // a = M * Ih
                remap(highRes_, a_, backwardMaps_[k], noArray(), INTER_NEAREST);
                // b = HM * Ih
                filter_->apply(a_, b_);
                // c = DHM * Ih
                resize(b_, c_, lowResSize, 0, 0, INTER_NEAREST);

                diffSign(src[k], c_, c_);

                // a = Dt * diff
                upscale(c_, a_, scale_);
                // b = HtDt * diff
                filter_->apply(a_, b_);
                // a = MtHtDt * diff
                remap(b_, a_, forwardMaps_[k], noArray(), INTER_NEAREST);

                add(diffTerm_, a_, diffTerm_);
            }

            if (lambda_ > 0)
            {
                calcBtvRegularization(highRes_, regTerm_, btvKernelSize_, btvWeights_);
                addWeighted(diffTerm_, 1.0, regTerm_, -lambda_, 0.0, diffTerm_);
            }

            addWeighted(highRes_, 1.0, diffTerm_, tau_, 0.0, highRes_);
        }

        Rect inner(btvKernelSize_, btvKernelSize_, highRes_.cols - 2 * btvKernelSize_, highRes_.rows - 2 * btvKernelSize_);
        highRes_(inner).copyTo(dst);
    }

    void BTVL1_Base::collectGarbage()
    {
        filter_.release();

        lowResForwardMotions_.clear();
        lowResBackwardMotions_.clear();

        highResForwardMotions_.clear();
        highResBackwardMotions_.clear();

        forwardMaps_.clear();
        backwardMaps_.clear();

        highRes_.release();

        diffTerm_.release();
        regTerm_.release();
        a_.release();
        b_.release();
        c_.release();
    }

////////////////////////////////////////////////////////////////////

    class BTVL1 : public SuperResolution, private BTVL1_Base
    {
    public:
        AlgorithmInfo* info() const;

        BTVL1();

        void collectGarbage();

    protected:
        void initImpl(Ptr<FrameSource>& frameSource);
        void processImpl(Ptr<FrameSource>& frameSource, OutputArray output);

    private:
        int temporalAreaRadius_;

        void readNextFrame(Ptr<FrameSource>& frameSource);
        void processFrame(int idx);

        Mat curFrame_;
        Mat prevFrame_;

        vector<Mat> frames_;
        vector<Mat> forwardMotions_;
        vector<Mat> backwardMotions_;
        vector<Mat> outputs_;

        int storePos_;
        int procPos_;
        int outPos_;

        vector<Mat> srcFrames_;
        vector<Mat> srcForwardMotions_;
        vector<Mat> srcBackwardMotions_;
        Mat finalOutput_;
    };

    CV_INIT_ALGORITHM(BTVL1, "SuperResolution.BTVL1",
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

    BTVL1::BTVL1()
    {
        temporalAreaRadius_ = 4;
    }

    void BTVL1::collectGarbage()
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
        BTVL1_Base::collectGarbage();
    }

    void BTVL1::initImpl(Ptr<FrameSource>& frameSource)
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

    void BTVL1::processImpl(Ptr<FrameSource>& frameSource, OutputArray _output)
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
        const Mat& curOutput = at(outPos_, outputs_);

        if (_output.kind() < _InputArray::OPENGL_BUFFER)
            curOutput.convertTo(_output, CV_8U);
        else
        {
            curOutput.convertTo(finalOutput_, CV_8U);
            arrCopy(finalOutput_, _output);
        }
    }

    void BTVL1::readNextFrame(Ptr<FrameSource>& frameSource)
    {
        frameSource->nextFrame(curFrame_);

        if (curFrame_.empty())
            return;

        ++storePos_;
        curFrame_.convertTo(at(storePos_, frames_), CV_32F);

        if (storePos_ > 0)
        {
            opticalFlow_->calc(prevFrame_, curFrame_, at(storePos_ - 1, forwardMotions_));
            opticalFlow_->calc(curFrame_, prevFrame_, at(storePos_, backwardMotions_));
        }

        curFrame_.copyTo(prevFrame_);
    }

    void BTVL1::processFrame(int idx)
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

Ptr<SuperResolution> cv::superres::createSuperResolution_BTVL1()
{
    return new BTVL1;
}
