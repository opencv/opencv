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
#include "opencl_kernels_superres.hpp"

using namespace cv;
using namespace cv::superres;
using namespace cv::superres::detail;

namespace
{
#ifdef HAVE_OPENCL

    bool ocl_calcRelativeMotions(InputArrayOfArrays _forwardMotions, InputArrayOfArrays _backwardMotions,
                                 OutputArrayOfArrays _relForwardMotions, OutputArrayOfArrays _relBackwardMotions,
                                 int baseIdx, const Size & size)
    {
        std::vector<UMat> & forwardMotions = *(std::vector<UMat> *)_forwardMotions.getObj(),
                & backwardMotions = *(std::vector<UMat> *)_backwardMotions.getObj(),
                & relForwardMotions = *(std::vector<UMat> *)_relForwardMotions.getObj(),
                & relBackwardMotions = *(std::vector<UMat> *)_relBackwardMotions.getObj();

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

        return true;
    }

#endif

    void calcRelativeMotions(InputArrayOfArrays _forwardMotions, InputArrayOfArrays _backwardMotions,
                             OutputArrayOfArrays _relForwardMotions, OutputArrayOfArrays _relBackwardMotions,
                             int baseIdx, const Size & size)
    {
        CV_OCL_RUN(_forwardMotions.isUMatVector() && _backwardMotions.isUMatVector() &&
                   _relForwardMotions.isUMatVector() && _relBackwardMotions.isUMatVector(),
                   ocl_calcRelativeMotions(_forwardMotions, _backwardMotions, _relForwardMotions,
                                           _relBackwardMotions, baseIdx, size))

        std::vector<Mat> & forwardMotions = *(std::vector<Mat> *)_forwardMotions.getObj(),
                & backwardMotions = *(std::vector<Mat> *)_backwardMotions.getObj(),
                & relForwardMotions = *(std::vector<Mat> *)_relForwardMotions.getObj(),
                & relBackwardMotions = *(std::vector<Mat> *)_relBackwardMotions.getObj();

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
#ifdef HAVE_OPENCL

    bool ocl_upscaleMotions(InputArrayOfArrays _lowResMotions, OutputArrayOfArrays _highResMotions, int scale)
    {
        std::vector<UMat> & lowResMotions = *(std::vector<UMat> *)_lowResMotions.getObj(),
                & highResMotions = *(std::vector<UMat> *)_highResMotions.getObj();

        highResMotions.resize(lowResMotions.size());

        for (size_t i = 0; i < lowResMotions.size(); ++i)
        {
            resize(lowResMotions[i], highResMotions[i], Size(), scale, scale, INTER_LINEAR); // TODO
            multiply(highResMotions[i], Scalar::all(scale), highResMotions[i]);
        }

        return true;
    }

#endif

    void upscaleMotions(InputArrayOfArrays _lowResMotions, OutputArrayOfArrays _highResMotions, int scale)
    {
        CV_OCL_RUN(_lowResMotions.isUMatVector() && _highResMotions.isUMatVector(),
                   ocl_upscaleMotions(_lowResMotions, _highResMotions, scale))

        std::vector<Mat> & lowResMotions = *(std::vector<Mat> *)_lowResMotions.getObj(),
                & highResMotions = *(std::vector<Mat> *)_highResMotions.getObj();

        highResMotions.resize(lowResMotions.size());

        for (size_t i = 0; i < lowResMotions.size(); ++i)
        {
            resize(lowResMotions[i], highResMotions[i], Size(), scale, scale, INTER_CUBIC);
            multiply(highResMotions[i], Scalar::all(scale), highResMotions[i]);
        }
    }

#ifdef HAVE_OPENCL

    bool ocl_buildMotionMaps(InputArray _forwardMotion, InputArray _backwardMotion,
                             OutputArray _forwardMap, OutputArray _backwardMap)
    {
        ocl::Kernel k("buildMotionMaps", ocl::superres::superres_btvl1_oclsrc);
        if (k.empty())
            return false;

        UMat forwardMotion = _forwardMotion.getUMat(), backwardMotion = _backwardMotion.getUMat();
        Size size = forwardMotion.size();

        _forwardMap.create(size, CV_32FC2);
        _backwardMap.create(size, CV_32FC2);

        UMat forwardMap = _forwardMap.getUMat(), backwardMap = _backwardMap.getUMat();

        k.args(ocl::KernelArg::ReadOnlyNoSize(forwardMotion),
               ocl::KernelArg::ReadOnlyNoSize(backwardMotion),
               ocl::KernelArg::WriteOnlyNoSize(forwardMap),
               ocl::KernelArg::WriteOnly(backwardMap));

        size_t globalsize[2] = { (size_t)size.width, (size_t)size.height };
        return k.run(2, globalsize, NULL, false);
    }

#endif

    void buildMotionMaps(InputArray _forwardMotion, InputArray _backwardMotion,
                         OutputArray _forwardMap, OutputArray _backwardMap)
    {
        CV_OCL_RUN(_forwardMap.isUMat() && _backwardMap.isUMat(),
                   ocl_buildMotionMaps(_forwardMotion, _backwardMotion, _forwardMap,
                                       _backwardMap));

        Mat forwardMotion = _forwardMotion.getMat(), backwardMotion = _backwardMotion.getMat();

        _forwardMap.create(forwardMotion.size(), CV_32FC2);
        _backwardMap.create(forwardMotion.size(), CV_32FC2);

        Mat forwardMap = _forwardMap.getMat(), backwardMap = _backwardMap.getMat();

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
    void upscaleImpl(InputArray _src, OutputArray _dst, int scale)
    {
        Mat src = _src.getMat();
        _dst.create(src.rows * scale, src.cols * scale, src.type());
        _dst.setTo(Scalar::all(0));
        Mat dst = _dst.getMat();

        for (int y = 0, Y = 0; y < src.rows; ++y, Y += scale)
        {
            const T * const srcRow = src.ptr<T>(y);
            T * const dstRow = dst.ptr<T>(Y);

            for (int x = 0, X = 0; x < src.cols; ++x, X += scale)
                dstRow[X] = srcRow[x];
        }
    }

#ifdef HAVE_OPENCL

    static bool ocl_upscale(InputArray _src, OutputArray _dst, int scale)
    {
        int type = _src.type(), cn = CV_MAT_CN(type);
        ocl::Kernel k("upscale", ocl::superres::superres_btvl1_oclsrc,
                      format("-D cn=%d", cn));
        if (k.empty())
            return false;

        UMat src = _src.getUMat();
        _dst.create(src.rows * scale, src.cols * scale, type);
        _dst.setTo(Scalar::all(0));
        UMat dst = _dst.getUMat();

        k.args(ocl::KernelArg::ReadOnly(src),
               ocl::KernelArg::ReadWriteNoSize(dst), scale);

        size_t globalsize[2] = { (size_t)src.cols, (size_t)src.rows };
        return k.run(2, globalsize, NULL, false);
    }

#endif

    typedef struct _Point4f { float ar[4]; } Point4f;

    void upscale(InputArray _src, OutputArray _dst, int scale)
    {
        int cn = _src.channels();
        CV_Assert( cn == 1 || cn == 3 || cn == 4 );

        CV_OCL_RUN(_dst.isUMat(),
                   ocl_upscale(_src, _dst, scale))

        typedef void (*func_t)(InputArray src, OutputArray dst, int scale);
        static const func_t funcs[] =
        {
            0, upscaleImpl<float>, 0, upscaleImpl<Point3f>, upscaleImpl<Point4f>
        };

        const func_t func = funcs[cn];
        CV_Assert(func != 0);
        func(_src, _dst, scale);
    }

    inline float diffSign(float a, float b)
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

#ifdef HAVE_OPENCL

    static bool ocl_diffSign(InputArray _src1, OutputArray _src2, OutputArray _dst)
    {
        ocl::Kernel k("diffSign", ocl::superres::superres_btvl1_oclsrc);
        if (k.empty())
            return false;

        UMat src1 = _src1.getUMat(), src2 = _src2.getUMat();
        _dst.create(src1.size(), src1.type());
        UMat dst = _dst.getUMat();

        int cn = src1.channels();
        k.args(ocl::KernelArg::ReadOnlyNoSize(src1),
               ocl::KernelArg::ReadOnlyNoSize(src2),
               ocl::KernelArg::WriteOnly(dst, cn));

        size_t globalsize[2] = { (size_t)src1.cols * cn, (size_t)src1.rows };
        return k.run(2, globalsize, NULL, false);
    }

#endif

    void diffSign(InputArray _src1, OutputArray _src2, OutputArray _dst)
    {
        CV_OCL_RUN(_dst.isUMat(),
                   ocl_diffSign(_src1, _src2, _dst))

        Mat src1 = _src1.getMat(), src2 = _src2.getMat();
        _dst.create(src1.size(), src1.type());
        Mat dst = _dst.getMat();

        const int count = src1.cols * src1.channels();

        for (int y = 0; y < src1.rows; ++y)
        {
            const float * const src1Ptr = src1.ptr<float>(y);
            const float * const src2Ptr = src2.ptr<float>(y);
            float* dstPtr = dst.ptr<float>(y);

            for (int x = 0; x < count; ++x)
                dstPtr[x] = diffSign(src1Ptr[x], src2Ptr[x]);
        }
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
            const T * const srcRow = src.ptr<T>(i);
            T * const dstRow = dst.ptr<T>(i);

            for(int j = ksize; j < src.cols - ksize; ++j)
            {
                const T srcVal = srcRow[j];

                for (int m = 0, ind = 0; m <= ksize; ++m)
                {
                    const T* srcRow2 = src.ptr<T>(i - m);
                    const T* srcRow3 = src.ptr<T>(i + m);

                    for (int l = ksize; l + m >= 0; --l, ++ind)
                        dstRow[j] += btvWeights[ind] * (diffSign(srcVal, srcRow3[j + l])
                                                        - diffSign(srcRow2[j - l], srcVal));
                }
            }
        }
    }

    template <typename T>
    void calcBtvRegularizationImpl(InputArray _src, OutputArray _dst, int btvKernelSize, const std::vector<float>& btvWeights)
    {
        Mat src = _src.getMat();
        _dst.create(src.size(), src.type());
        _dst.setTo(Scalar::all(0));
        Mat dst = _dst.getMat();

        const int ksize = (btvKernelSize - 1) / 2;

        BtvRegularizationBody<T> body;

        body.src = src;
        body.dst = dst;
        body.ksize = ksize;
        body.btvWeights = &btvWeights[0];

        parallel_for_(Range(ksize, src.rows - ksize), body);
    }

#ifdef HAVE_OPENCL

    static bool ocl_calcBtvRegularization(InputArray _src, OutputArray _dst, int btvKernelSize, const UMat & ubtvWeights)
    {
        int cn = _src.channels();
        ocl::Kernel k("calcBtvRegularization", ocl::superres::superres_btvl1_oclsrc,
                      format("-D cn=%d", cn));
        if (k.empty())
            return false;

        UMat src = _src.getUMat();
        _dst.create(src.size(), src.type());
        _dst.setTo(Scalar::all(0));
        UMat dst = _dst.getUMat();

        const int ksize = (btvKernelSize - 1) / 2;

        k.args(ocl::KernelArg::ReadOnlyNoSize(src), ocl::KernelArg::WriteOnly(dst),
              ksize, ocl::KernelArg::PtrReadOnly(ubtvWeights));

        size_t globalsize[2] = { (size_t)src.cols, (size_t)src.rows };
        return k.run(2, globalsize, NULL, false);
    }

#endif

    void calcBtvRegularization(InputArray _src, OutputArray _dst, int btvKernelSize,
                               const std::vector<float>& btvWeights, const UMat & ubtvWeights)
    {
        CV_OCL_RUN(_dst.isUMat(),
                   ocl_calcBtvRegularization(_src, _dst, btvKernelSize, ubtvWeights))
        (void)ubtvWeights;

        typedef void (*func_t)(InputArray _src, OutputArray _dst, int btvKernelSize, const std::vector<float>& btvWeights);
        static const func_t funcs[] =
        {
            0, calcBtvRegularizationImpl<float>, 0, calcBtvRegularizationImpl<Point3f>, 0
        };

        const func_t func = funcs[_src.channels()];
        CV_Assert(func != 0);
        func(_src, _dst, btvKernelSize, btvWeights);
    }

    class BTVL1_Base : public cv::superres::SuperResolution
    {
    public:
        BTVL1_Base();

        void process(InputArrayOfArrays src, OutputArray dst, InputArrayOfArrays forwardMotions,
                     InputArrayOfArrays backwardMotions, int baseIdx);

        void collectGarbage();

        CV_IMPL_PROPERTY(int, Scale, scale_)
        CV_IMPL_PROPERTY(int, Iterations, iterations_)
        CV_IMPL_PROPERTY(double, Tau, tau_)
        CV_IMPL_PROPERTY(double, Labmda, lambda_)
        CV_IMPL_PROPERTY(double, Alpha, alpha_)
        CV_IMPL_PROPERTY(int, KernelSize, btvKernelSize_)
        CV_IMPL_PROPERTY(int, BlurKernelSize, blurKernelSize_)
        CV_IMPL_PROPERTY(double, BlurSigma, blurSigma_)
        CV_IMPL_PROPERTY(int, TemporalAreaRadius, temporalAreaRadius_)
        CV_IMPL_PROPERTY_S(Ptr<cv::superres::DenseOpticalFlowExt>, OpticalFlow, opticalFlow_)

    protected:
        int scale_;
        int iterations_;
        double tau_;
        double lambda_;
        double alpha_;
        int btvKernelSize_;
        int blurKernelSize_;
        double blurSigma_;
        int temporalAreaRadius_; // not used in some implementations
        Ptr<cv::superres::DenseOpticalFlowExt> opticalFlow_;

    private:
        bool ocl_process(InputArrayOfArrays src, OutputArray dst, InputArrayOfArrays forwardMotions,
                         InputArrayOfArrays backwardMotions, int baseIdx);

        //Ptr<FilterEngine> filter_;
        int curBlurKernelSize_;
        double curBlurSigma_;
        int curSrcType_;

        std::vector<float> btvWeights_;
        UMat ubtvWeights_;

        int curBtvKernelSize_;
        double curAlpha_;

        // Mat
        std::vector<Mat> lowResForwardMotions_;
        std::vector<Mat> lowResBackwardMotions_;

        std::vector<Mat> highResForwardMotions_;
        std::vector<Mat> highResBackwardMotions_;

        std::vector<Mat> forwardMaps_;
        std::vector<Mat> backwardMaps_;

        Mat highRes_;

        Mat diffTerm_, regTerm_;
        Mat a_, b_, c_;

#ifdef HAVE_OPENCL
        // UMat
        std::vector<UMat> ulowResForwardMotions_;
        std::vector<UMat> ulowResBackwardMotions_;

        std::vector<UMat> uhighResForwardMotions_;
        std::vector<UMat> uhighResBackwardMotions_;

        std::vector<UMat> uforwardMaps_;
        std::vector<UMat> ubackwardMaps_;

        UMat uhighRes_;

        UMat udiffTerm_, uregTerm_;
        UMat ua_, ub_, uc_;
#endif
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
        temporalAreaRadius_ = 0;
        opticalFlow_ = createOptFlow_Farneback();

        curBlurKernelSize_ = -1;
        curBlurSigma_ = -1.0;
        curSrcType_ = -1;

        curBtvKernelSize_ = -1;
        curAlpha_ = -1.0;
    }

#ifdef HAVE_OPENCL

    bool BTVL1_Base::ocl_process(InputArrayOfArrays _src, OutputArray _dst, InputArrayOfArrays _forwardMotions,
                                 InputArrayOfArrays _backwardMotions, int baseIdx)
    {
        std::vector<UMat> & src = *(std::vector<UMat> *)_src.getObj(),
                & forwardMotions = *(std::vector<UMat> *)_forwardMotions.getObj(),
                & backwardMotions = *(std::vector<UMat> *)_backwardMotions.getObj();

        // update blur filter and btv weights
        if (blurKernelSize_ != curBlurKernelSize_ || blurSigma_ != curBlurSigma_ || src[0].type() != curSrcType_)
        {
            //filter_ = createGaussianFilter(src[0].type(), Size(blurKernelSize_, blurKernelSize_), blurSigma_);
            curBlurKernelSize_ = blurKernelSize_;
            curBlurSigma_ = blurSigma_;
            curSrcType_ = src[0].type();
        }

        if (btvWeights_.empty() || btvKernelSize_ != curBtvKernelSize_ || alpha_ != curAlpha_)
        {
            calcBtvWeights(btvKernelSize_, alpha_, btvWeights_);
            Mat(btvWeights_, true).copyTo(ubtvWeights_);

            curBtvKernelSize_ = btvKernelSize_;
            curAlpha_ = alpha_;
        }

        // calc high res motions
        calcRelativeMotions(forwardMotions, backwardMotions, ulowResForwardMotions_, ulowResBackwardMotions_, baseIdx, src[0].size());

        upscaleMotions(ulowResForwardMotions_, uhighResForwardMotions_, scale_);
        upscaleMotions(ulowResBackwardMotions_, uhighResBackwardMotions_, scale_);

        uforwardMaps_.resize(uhighResForwardMotions_.size());
        ubackwardMaps_.resize(uhighResForwardMotions_.size());
        for (size_t i = 0; i < uhighResForwardMotions_.size(); ++i)
            buildMotionMaps(uhighResForwardMotions_[i], uhighResBackwardMotions_[i], uforwardMaps_[i], ubackwardMaps_[i]);

        // initial estimation
        const Size lowResSize = src[0].size();
        const Size highResSize(lowResSize.width * scale_, lowResSize.height * scale_);

        resize(src[baseIdx], uhighRes_, highResSize, 0, 0, INTER_LINEAR); // TODO

        // iterations
        udiffTerm_.create(highResSize, uhighRes_.type());
        ua_.create(highResSize, uhighRes_.type());
        ub_.create(highResSize, uhighRes_.type());
        uc_.create(lowResSize, uhighRes_.type());

        for (int i = 0; i < iterations_; ++i)
        {
            udiffTerm_.setTo(Scalar::all(0));

            for (size_t k = 0; k < src.size(); ++k)
            {
                // a = M * Ih
                remap(uhighRes_, ua_, ubackwardMaps_[k], noArray(), INTER_NEAREST);
                // b = HM * Ih
                GaussianBlur(ua_, ub_, Size(blurKernelSize_, blurKernelSize_), blurSigma_);
                // c = DHM * Ih
                resize(ub_, uc_, lowResSize, 0, 0, INTER_NEAREST);

                diffSign(src[k], uc_, uc_);

                // a = Dt * diff
                upscale(uc_, ua_, scale_);

                // b = HtDt * diff
                GaussianBlur(ua_, ub_, Size(blurKernelSize_, blurKernelSize_), blurSigma_);
                // a = MtHtDt * diff
                remap(ub_, ua_, uforwardMaps_[k], noArray(), INTER_NEAREST);

                add(udiffTerm_, ua_, udiffTerm_);
            }

            if (lambda_ > 0)
            {
                calcBtvRegularization(uhighRes_, uregTerm_, btvKernelSize_, btvWeights_, ubtvWeights_);
                addWeighted(udiffTerm_, 1.0, uregTerm_, -lambda_, 0.0, udiffTerm_);
            }

            addWeighted(uhighRes_, 1.0, udiffTerm_, tau_, 0.0, uhighRes_);
        }

        Rect inner(btvKernelSize_, btvKernelSize_, uhighRes_.cols - 2 * btvKernelSize_, uhighRes_.rows - 2 * btvKernelSize_);
        uhighRes_(inner).copyTo(_dst);

        return true;
    }

#endif

    void BTVL1_Base::process(InputArrayOfArrays _src, OutputArray _dst, InputArrayOfArrays _forwardMotions,
                             InputArrayOfArrays _backwardMotions, int baseIdx)
    {
        CV_INSTRUMENT_REGION()

        CV_Assert( scale_ > 1 );
        CV_Assert( iterations_ > 0 );
        CV_Assert( tau_ > 0.0 );
        CV_Assert( alpha_ > 0.0 );
        CV_Assert( btvKernelSize_ > 0 );
        CV_Assert( blurKernelSize_ > 0 );
        CV_Assert( blurSigma_ >= 0.0 );

        CV_OCL_RUN(_src.isUMatVector() && _dst.isUMat() && _forwardMotions.isUMatVector() &&
                   _backwardMotions.isUMatVector(),
                   ocl_process(_src, _dst, _forwardMotions, _backwardMotions, baseIdx))

        std::vector<Mat> & src = *(std::vector<Mat> *)_src.getObj(),
                & forwardMotions = *(std::vector<Mat> *)_forwardMotions.getObj(),
                & backwardMotions = *(std::vector<Mat> *)_backwardMotions.getObj();

        // update blur filter and btv weights
        if (blurKernelSize_ != curBlurKernelSize_ || blurSigma_ != curBlurSigma_ || src[0].type() != curSrcType_)
        {
            //filter_ = createGaussianFilter(src[0].type(), Size(blurKernelSize_, blurKernelSize_), blurSigma_);
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
                GaussianBlur(a_, b_, Size(blurKernelSize_, blurKernelSize_), blurSigma_);
                // c = DHM * Ih
                resize(b_, c_, lowResSize, 0, 0, INTER_NEAREST);

                diffSign(src[k], c_, c_);

                // a = Dt * diff
                upscale(c_, a_, scale_);
                // b = HtDt * diff
                GaussianBlur(a_, b_, Size(blurKernelSize_, blurKernelSize_), blurSigma_);
                // a = MtHtDt * diff
                remap(b_, a_, forwardMaps_[k], noArray(), INTER_NEAREST);

                add(diffTerm_, a_, diffTerm_);
            }

            if (lambda_ > 0)
            {
                calcBtvRegularization(highRes_, regTerm_, btvKernelSize_, btvWeights_, ubtvWeights_);
                addWeighted(diffTerm_, 1.0, regTerm_, -lambda_, 0.0, diffTerm_);
            }

            addWeighted(highRes_, 1.0, diffTerm_, tau_, 0.0, highRes_);
        }

        Rect inner(btvKernelSize_, btvKernelSize_, highRes_.cols - 2 * btvKernelSize_, highRes_.rows - 2 * btvKernelSize_);
        highRes_(inner).copyTo(_dst);
    }

    void BTVL1_Base::collectGarbage()
    {
        // Mat
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

#ifdef HAVE_OPENCL
        // UMat
        ulowResForwardMotions_.clear();
        ulowResBackwardMotions_.clear();

        uhighResForwardMotions_.clear();
        uhighResBackwardMotions_.clear();

        uforwardMaps_.clear();
        ubackwardMaps_.clear();

        uhighRes_.release();

        udiffTerm_.release();
        uregTerm_.release();
        ua_.release();
        ub_.release();
        uc_.release();
#endif
    }

////////////////////////////////////////////////////////////////////

    class BTVL1 : public BTVL1_Base
    {
    public:
        BTVL1();

        void collectGarbage();

    protected:
        void initImpl(Ptr<FrameSource>& frameSource);
        bool ocl_initImpl(Ptr<FrameSource>& frameSource);

        void processImpl(Ptr<FrameSource>& frameSource, OutputArray output);
        bool ocl_processImpl(Ptr<FrameSource>& frameSource, OutputArray output);

    private:
        void readNextFrame(Ptr<FrameSource>& frameSource);
        bool ocl_readNextFrame(Ptr<FrameSource>& frameSource);

        void processFrame(int idx);
        bool ocl_processFrame(int idx);

        int storePos_;
        int procPos_;
        int outPos_;

        // Mat
        Mat curFrame_;
        Mat prevFrame_;

        std::vector<Mat> frames_;
        std::vector<Mat> forwardMotions_;
        std::vector<Mat> backwardMotions_;
        std::vector<Mat> outputs_;

        std::vector<Mat> srcFrames_;
        std::vector<Mat> srcForwardMotions_;
        std::vector<Mat> srcBackwardMotions_;
        Mat finalOutput_;

#ifdef HAVE_OPENCL
        // UMat
        UMat ucurFrame_;
        UMat uprevFrame_;

        std::vector<UMat> uframes_;
        std::vector<UMat> uforwardMotions_;
        std::vector<UMat> ubackwardMotions_;
        std::vector<UMat> uoutputs_;

        std::vector<UMat> usrcFrames_;
        std::vector<UMat> usrcForwardMotions_;
        std::vector<UMat> usrcBackwardMotions_;
#endif
    };

    BTVL1::BTVL1()
    {
        temporalAreaRadius_ = 4;
    }

    void BTVL1::collectGarbage()
    {
        // Mat
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

#ifdef HAVE_OPENCL
        // UMat
        ucurFrame_.release();
        uprevFrame_.release();

        uframes_.clear();
        uforwardMotions_.clear();
        ubackwardMotions_.clear();
        uoutputs_.clear();

        usrcFrames_.clear();
        usrcForwardMotions_.clear();
        usrcBackwardMotions_.clear();
#endif

        SuperResolution::collectGarbage();
        BTVL1_Base::collectGarbage();
    }

#ifdef HAVE_OPENCL

    bool BTVL1::ocl_initImpl(Ptr<FrameSource>& frameSource)
    {
        const int cacheSize = 2 * temporalAreaRadius_ + 1;

        uframes_.resize(cacheSize);
        uforwardMotions_.resize(cacheSize);
        ubackwardMotions_.resize(cacheSize);
        uoutputs_.resize(cacheSize);

        storePos_ = -1;

        for (int t = -temporalAreaRadius_; t <= temporalAreaRadius_; ++t)
            readNextFrame(frameSource);

        for (int i = 0; i <= temporalAreaRadius_; ++i)
            processFrame(i);

        procPos_ = temporalAreaRadius_;
        outPos_ = -1;

        return true;
    }

#endif

    void BTVL1::initImpl(Ptr<FrameSource>& frameSource)
    {
        const int cacheSize = 2 * temporalAreaRadius_ + 1;

        frames_.resize(cacheSize);
        forwardMotions_.resize(cacheSize);
        backwardMotions_.resize(cacheSize);
        outputs_.resize(cacheSize);

        CV_OCL_RUN(isUmat_,
                   ocl_initImpl(frameSource))

        storePos_ = -1;

        for (int t = -temporalAreaRadius_; t <= temporalAreaRadius_; ++t)
            readNextFrame(frameSource);

        for (int i = 0; i <= temporalAreaRadius_; ++i)
            processFrame(i);

        procPos_ = temporalAreaRadius_;
        outPos_ = -1;
    }

#ifdef HAVE_OPENCL

    bool BTVL1::ocl_processImpl(Ptr<FrameSource>& /*frameSource*/, OutputArray _output)
    {
        const UMat& curOutput = at(outPos_, uoutputs_);
        curOutput.convertTo(_output, CV_8U);

        return true;
    }

#endif

    void BTVL1::processImpl(Ptr<FrameSource>& frameSource, OutputArray _output)
    {
        CV_INSTRUMENT_REGION()

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

        CV_OCL_RUN(isUmat_,
                   ocl_processImpl(frameSource, _output))

        const Mat& curOutput = at(outPos_, outputs_);

        if (_output.kind() < _InputArray::OPENGL_BUFFER || _output.isUMat())
            curOutput.convertTo(_output, CV_8U);
        else
        {
            curOutput.convertTo(finalOutput_, CV_8U);
            arrCopy(finalOutput_, _output);
        }
    }

#ifdef HAVE_OPENCL

    bool BTVL1::ocl_readNextFrame(Ptr<FrameSource>& /*frameSource*/)
    {
        ucurFrame_.convertTo(at(storePos_, uframes_), CV_32F);

        if (storePos_ > 0)
        {
            opticalFlow_->calc(uprevFrame_, ucurFrame_, at(storePos_ - 1, uforwardMotions_));
            opticalFlow_->calc(ucurFrame_, uprevFrame_, at(storePos_, ubackwardMotions_));
        }

        ucurFrame_.copyTo(uprevFrame_);
        return true;
    }

#endif

    void BTVL1::readNextFrame(Ptr<FrameSource>& frameSource)
    {
        CV_INSTRUMENT_REGION()

        frameSource->nextFrame(curFrame_);
        if (curFrame_.empty())
            return;

#ifdef HAVE_OPENCL
        if (isUmat_)
            curFrame_.copyTo(ucurFrame_);
#endif
        ++storePos_;

        CV_OCL_RUN(isUmat_,
                   ocl_readNextFrame(frameSource))

        curFrame_.convertTo(at(storePos_, frames_), CV_32F);

        if (storePos_ > 0)
        {
            opticalFlow_->calc(prevFrame_, curFrame_, at(storePos_ - 1, forwardMotions_));
            opticalFlow_->calc(curFrame_, prevFrame_, at(storePos_, backwardMotions_));
        }

        curFrame_.copyTo(prevFrame_);
    }

#ifdef HAVE_OPENCL

    bool BTVL1::ocl_processFrame(int idx)
    {
        const int startIdx = std::max(idx - temporalAreaRadius_, 0);
        const int procIdx = idx;
        const int endIdx = std::min(startIdx + 2 * temporalAreaRadius_, storePos_);

        const int count = endIdx - startIdx + 1;

        usrcFrames_.resize(count);
        usrcForwardMotions_.resize(count);
        usrcBackwardMotions_.resize(count);

        int baseIdx = -1;

        for (int i = startIdx, k = 0; i <= endIdx; ++i, ++k)
        {
            if (i == procIdx)
                baseIdx = k;

            usrcFrames_[k] = at(i, uframes_);

            if (i < endIdx)
                usrcForwardMotions_[k] = at(i, uforwardMotions_);
            if (i > startIdx)
                usrcBackwardMotions_[k] = at(i, ubackwardMotions_);
        }

        process(usrcFrames_, at(idx, uoutputs_), usrcForwardMotions_, usrcBackwardMotions_, baseIdx);

        return true;
    }

#endif

    void BTVL1::processFrame(int idx)
    {
        CV_INSTRUMENT_REGION()

        CV_OCL_RUN(isUmat_,
                   ocl_processFrame(idx))

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

Ptr<cv::superres::SuperResolution> cv::superres::createSuperResolution_BTVL1()
{
    return makePtr<BTVL1>();
}
