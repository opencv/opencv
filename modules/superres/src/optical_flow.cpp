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

using namespace std;
using namespace cv;
using namespace cv::gpu;
using namespace cv::superres;
using namespace cv::superres::detail;

///////////////////////////////////////////////////////////////////
// CpuOpticalFlow

namespace
{
    class CpuOpticalFlow : public DenseOpticalFlowExt
    {
    public:
        explicit CpuOpticalFlow(int work_type);

        void calc(InputArray frame0, InputArray frame1, OutputArray flow1, OutputArray flow2);
        void collectGarbage();

    protected:
        virtual void impl(const Mat& input0, const Mat& input1, OutputArray dst) = 0;

    private:
        int work_type_;
        Mat buf_[6];
        Mat flow_;
        Mat flows_[2];
    };

    CpuOpticalFlow::CpuOpticalFlow(int work_type) : work_type_(work_type)
    {
    }

    void CpuOpticalFlow::calc(InputArray _frame0, InputArray _frame1, OutputArray _flow1, OutputArray _flow2)
    {
        Mat frame0 = arrGetMat(_frame0, buf_[0]);
        Mat frame1 = arrGetMat(_frame1, buf_[1]);

        CV_Assert( frame1.type() == frame0.type() );
        CV_Assert( frame1.size() == frame0.size() );

        Mat input0 = convertToType(frame0, work_type_, buf_[2], buf_[3]);
        Mat input1 = convertToType(frame1, work_type_, buf_[4], buf_[5]);

        if (!_flow2.needed() && _flow1.kind() < _InputArray::OPENGL_BUFFER)
        {
            impl(input0, input1, _flow1);
            return;
        }

        impl(input0, input1, flow_);

        if (!_flow2.needed())
        {
            arrCopy(flow_, _flow1);
        }
        else
        {
            split(flow_, flows_);

            arrCopy(flows_[0], _flow1);
            arrCopy(flows_[1], _flow2);
        }
    }

    void CpuOpticalFlow::collectGarbage()
    {
        for (int i = 0; i < 6; ++i)
            buf_[i].release();
        flow_.release();
        flows_[0].release();
        flows_[1].release();
    }
}

///////////////////////////////////////////////////////////////////
// Farneback

namespace
{
    class Farneback : public CpuOpticalFlow
    {
    public:
        AlgorithmInfo* info() const;

        Farneback();

    protected:
        void impl(const Mat& input0, const Mat& input1, OutputArray dst);

    private:
        double pyrScale_;
        int numLevels_;
        int winSize_;
        int numIters_;
        int polyN_;
        double polySigma_;
        int flags_;
    };

    CV_INIT_ALGORITHM(Farneback, "DenseOpticalFlowExt.Farneback",
                      obj.info()->addParam(obj, "pyrScale", obj.pyrScale_);
                      obj.info()->addParam(obj, "numLevels", obj.numLevels_);
                      obj.info()->addParam(obj, "winSize", obj.winSize_);
                      obj.info()->addParam(obj, "numIters", obj.numIters_);
                      obj.info()->addParam(obj, "polyN", obj.polyN_);
                      obj.info()->addParam(obj, "polySigma", obj.polySigma_);
                      obj.info()->addParam(obj, "flags", obj.flags_));

    Farneback::Farneback() : CpuOpticalFlow(CV_8UC1)
    {
        pyrScale_ = 0.5;
        numLevels_ = 5;
        winSize_ = 13;
        numIters_ = 10;
        polyN_ = 5;
        polySigma_ = 1.1;
        flags_ = 0;
    }

    void Farneback::impl(const Mat& input0, const Mat& input1, OutputArray dst)
    {
        calcOpticalFlowFarneback(input0, input1, dst, pyrScale_, numLevels_, winSize_, numIters_, polyN_, polySigma_, flags_);
    }
}

Ptr<DenseOpticalFlowExt> cv::superres::createOptFlow_Farneback()
{
    return new Farneback;
}

///////////////////////////////////////////////////////////////////
// Simple

namespace
{
    class Simple : public CpuOpticalFlow
    {
    public:
        AlgorithmInfo* info() const;

        Simple();

    protected:
        void impl(const Mat& input0, const Mat& input1, OutputArray dst);

    private:
        int layers_;
        int averagingBlockSize_;
        int maxFlow_;
        double sigmaDist_;
        double sigmaColor_;
        int postProcessWindow_;
        double sigmaDistFix_;
        double sigmaColorFix_;
        double occThr_;
        int upscaleAveragingRadius_;
        double upscaleSigmaDist_;
        double upscaleSigmaColor_;
        double speedUpThr_;
    };

    CV_INIT_ALGORITHM(Simple, "DenseOpticalFlowExt.Simple",
                      obj.info()->addParam(obj, "layers", obj.layers_);
                      obj.info()->addParam(obj, "averagingBlockSize", obj.averagingBlockSize_);
                      obj.info()->addParam(obj, "maxFlow", obj.maxFlow_);
                      obj.info()->addParam(obj, "sigmaDist", obj.sigmaDist_);
                      obj.info()->addParam(obj, "sigmaColor", obj.sigmaColor_);
                      obj.info()->addParam(obj, "postProcessWindow", obj.postProcessWindow_);
                      obj.info()->addParam(obj, "sigmaDistFix", obj.sigmaDistFix_);
                      obj.info()->addParam(obj, "sigmaColorFix", obj.sigmaColorFix_);
                      obj.info()->addParam(obj, "occThr", obj.occThr_);
                      obj.info()->addParam(obj, "upscaleAveragingRadius", obj.upscaleAveragingRadius_);
                      obj.info()->addParam(obj, "upscaleSigmaDist", obj.upscaleSigmaDist_);
                      obj.info()->addParam(obj, "upscaleSigmaColor", obj.upscaleSigmaColor_);
                      obj.info()->addParam(obj, "speedUpThr", obj.speedUpThr_));

    Simple::Simple() : CpuOpticalFlow(CV_8UC3)
    {
        layers_ = 3;
        averagingBlockSize_ = 2;
        maxFlow_ = 4;
        sigmaDist_ = 4.1;
        sigmaColor_ = 25.5;
        postProcessWindow_ = 18;
        sigmaDistFix_ = 55.0;
        sigmaColorFix_ = 25.5;
        occThr_ = 0.35;
        upscaleAveragingRadius_ = 18;
        upscaleSigmaDist_ = 55.0;
        upscaleSigmaColor_ = 25.5;
        speedUpThr_ = 10;
    }

    void Simple::impl(const Mat& _input0, const Mat& _input1, OutputArray dst)
    {
        Mat input0 = _input0;
        Mat input1 = _input1;
        calcOpticalFlowSF(input0, input1, dst.getMatRef(),
                          layers_,
                          averagingBlockSize_,
                          maxFlow_,
                          sigmaDist_,
                          sigmaColor_,
                          postProcessWindow_,
                          sigmaDistFix_,
                          sigmaColorFix_,
                          occThr_,
                          upscaleAveragingRadius_,
                          upscaleSigmaDist_,
                          upscaleSigmaColor_,
                          speedUpThr_);
    }
}

Ptr<DenseOpticalFlowExt> cv::superres::createOptFlow_Simple()
{
    return new Simple;
}

///////////////////////////////////////////////////////////////////
// DualTVL1

namespace
{
    class DualTVL1 : public CpuOpticalFlow
    {
    public:
        AlgorithmInfo* info() const;

        DualTVL1();

        void collectGarbage();

    protected:
        void impl(const Mat& input0, const Mat& input1, OutputArray dst);

    private:
        double tau_;
        double lambda_;
        double theta_;
        int nscales_;
        int warps_;
        double epsilon_;
        int iterations_;
        bool useInitialFlow_;

        Ptr<DenseOpticalFlow> alg_;
    };

    CV_INIT_ALGORITHM(DualTVL1, "DenseOpticalFlowExt.DualTVL1",
                      obj.info()->addParam(obj, "tau", obj.tau_);
                      obj.info()->addParam(obj, "lambda", obj.lambda_);
                      obj.info()->addParam(obj, "theta", obj.theta_);
                      obj.info()->addParam(obj, "nscales", obj.nscales_);
                      obj.info()->addParam(obj, "warps", obj.warps_);
                      obj.info()->addParam(obj, "epsilon", obj.epsilon_);
                      obj.info()->addParam(obj, "iterations", obj.iterations_);
                      obj.info()->addParam(obj, "useInitialFlow", obj.useInitialFlow_));

    DualTVL1::DualTVL1() : CpuOpticalFlow(CV_8UC1)
    {
        alg_ = cv::createOptFlow_DualTVL1();
        tau_ = alg_->getDouble("tau");
        lambda_ = alg_->getDouble("lambda");
        theta_ = alg_->getDouble("theta");
        nscales_ = alg_->getInt("nscales");
        warps_ = alg_->getInt("warps");
        epsilon_ = alg_->getDouble("epsilon");
        iterations_ = alg_->getInt("iterations");
        useInitialFlow_ = alg_->getBool("useInitialFlow");
    }

    void DualTVL1::impl(const Mat& input0, const Mat& input1, OutputArray dst)
    {
        alg_->set("tau", tau_);
        alg_->set("lambda", lambda_);
        alg_->set("theta", theta_);
        alg_->set("nscales", nscales_);
        alg_->set("warps", warps_);
        alg_->set("epsilon", epsilon_);
        alg_->set("iterations", iterations_);
        alg_->set("useInitialFlow", useInitialFlow_);

        alg_->calc(input0, input1, dst);
    }

    void DualTVL1::collectGarbage()
    {
        alg_->collectGarbage();
        CpuOpticalFlow::collectGarbage();
    }
}

Ptr<DenseOpticalFlowExt> cv::superres::createOptFlow_DualTVL1()
{
    return new DualTVL1;
}

///////////////////////////////////////////////////////////////////
// GpuOpticalFlow

#ifndef HAVE_OPENCV_GPU

Ptr<DenseOpticalFlowExt> cv::superres::createOptFlow_Farneback_GPU()
{
    CV_Error(CV_StsNotImplemented, "The called functionality is disabled for current build or platform");
    return Ptr<DenseOpticalFlowExt>();
}

Ptr<DenseOpticalFlowExt> cv::superres::createOptFlow_DualTVL1_GPU()
{
    CV_Error(CV_StsNotImplemented, "The called functionality is disabled for current build or platform");
    return Ptr<DenseOpticalFlowExt>();
}

Ptr<DenseOpticalFlowExt> cv::superres::createOptFlow_Brox_GPU()
{
    CV_Error(CV_StsNotImplemented, "The called functionality is disabled for current build or platform");
    return Ptr<DenseOpticalFlowExt>();
}

Ptr<DenseOpticalFlowExt> cv::superres::createOptFlow_PyrLK_GPU()
{
    CV_Error(CV_StsNotImplemented, "The called functionality is disabled for current build or platform");
    return Ptr<DenseOpticalFlowExt>();
}

#else // HAVE_OPENCV_GPU

namespace
{
    class GpuOpticalFlow : public DenseOpticalFlowExt
    {
    public:
        explicit GpuOpticalFlow(int work_type);

        void calc(InputArray frame0, InputArray frame1, OutputArray flow1, OutputArray flow2);
        void collectGarbage();

    protected:
        virtual void impl(const GpuMat& input0, const GpuMat& input1, GpuMat& dst1, GpuMat& dst2) = 0;

    private:
        int work_type_;
        GpuMat buf_[6];
        GpuMat u_, v_, flow_;
    };

    GpuOpticalFlow::GpuOpticalFlow(int work_type) : work_type_(work_type)
    {
    }

    void GpuOpticalFlow::calc(InputArray _frame0, InputArray _frame1, OutputArray _flow1, OutputArray _flow2)
    {
        GpuMat frame0 = arrGetGpuMat(_frame0, buf_[0]);
        GpuMat frame1 = arrGetGpuMat(_frame1, buf_[1]);

        CV_Assert( frame1.type() == frame0.type() );
        CV_Assert( frame1.size() == frame0.size() );

        GpuMat input0 = convertToType(frame0, work_type_, buf_[2], buf_[3]);
        GpuMat input1 = convertToType(frame1, work_type_, buf_[4], buf_[5]);

        if (_flow2.needed() && _flow1.kind() == _InputArray::GPU_MAT && _flow2.kind() == _InputArray::GPU_MAT)
        {
            impl(input0, input1, _flow1.getGpuMatRef(), _flow2.getGpuMatRef());
            return;
        }

        impl(input0, input1, u_, v_);

        if (_flow2.needed())
        {
            arrCopy(u_, _flow1);
            arrCopy(v_, _flow2);
        }
        else
        {
            GpuMat src[] = {u_, v_};
            merge(src, 2, flow_);
            arrCopy(flow_, _flow1);
        }
    }

    void GpuOpticalFlow::collectGarbage()
    {
        for (int i = 0; i < 6; ++i)
            buf_[i].release();
        u_.release();
        v_.release();
        flow_.release();
    }
}

///////////////////////////////////////////////////////////////////
// Brox_GPU

namespace
{
    class Brox_GPU : public GpuOpticalFlow
    {
    public:
        AlgorithmInfo* info() const;

        Brox_GPU();

        void collectGarbage();

    protected:
        void impl(const GpuMat& input0, const GpuMat& input1, GpuMat& dst1, GpuMat& dst2);

    private:
        double alpha_;
        double gamma_;
        double scaleFactor_;
        int innerIterations_;
        int outerIterations_;
        int solverIterations_;

        BroxOpticalFlow alg_;
    };

    CV_INIT_ALGORITHM(Brox_GPU, "DenseOpticalFlowExt.Brox_GPU",
                      obj.info()->addParam(obj, "alpha", obj.alpha_, false, 0, 0, "Flow smoothness");
                      obj.info()->addParam(obj, "gamma", obj.gamma_, false, 0, 0, "Gradient constancy importance");
                      obj.info()->addParam(obj, "scaleFactor", obj.scaleFactor_, false, 0, 0, "Pyramid scale factor");
                      obj.info()->addParam(obj, "innerIterations", obj.innerIterations_, false, 0, 0, "Number of lagged non-linearity iterations (inner loop)");
                      obj.info()->addParam(obj, "outerIterations", obj.outerIterations_, false, 0, 0, "Number of warping iterations (number of pyramid levels)");
                      obj.info()->addParam(obj, "solverIterations", obj.solverIterations_, false, 0, 0, "Number of linear system solver iterations"));

    Brox_GPU::Brox_GPU() : GpuOpticalFlow(CV_32FC1), alg_(0.197f, 50.0f, 0.8f, 10, 77, 10)
    {
        alpha_ = alg_.alpha;
        gamma_ = alg_.gamma;
        scaleFactor_ = alg_.scale_factor;
        innerIterations_ = alg_.inner_iterations;
        outerIterations_ = alg_.outer_iterations;
        solverIterations_ = alg_.solver_iterations;
    }

    void Brox_GPU::impl(const GpuMat& input0, const GpuMat& input1, GpuMat& dst1, GpuMat& dst2)
    {
        alg_.alpha = static_cast<float>(alpha_);
        alg_.gamma = static_cast<float>(gamma_);
        alg_.scale_factor = static_cast<float>(scaleFactor_);
        alg_.inner_iterations = innerIterations_;
        alg_.outer_iterations = outerIterations_;
        alg_.solver_iterations = solverIterations_;

        alg_(input0, input1, dst1, dst2);
    }

    void Brox_GPU::collectGarbage()
    {
        alg_.buf.release();
        GpuOpticalFlow::collectGarbage();
    }
}

Ptr<DenseOpticalFlowExt> cv::superres::createOptFlow_Brox_GPU()
{
    return new Brox_GPU;
}

///////////////////////////////////////////////////////////////////
// PyrLK_GPU

namespace
{
    class PyrLK_GPU : public GpuOpticalFlow
    {
    public:
        AlgorithmInfo* info() const;

        PyrLK_GPU();

        void collectGarbage();

    protected:
        void impl(const GpuMat& input0, const GpuMat& input1, GpuMat& dst1, GpuMat& dst2);

    private:
        int winSize_;
        int maxLevel_;
        int iterations_;

        PyrLKOpticalFlow alg_;
    };

    CV_INIT_ALGORITHM(PyrLK_GPU, "DenseOpticalFlowExt.PyrLK_GPU",
                      obj.info()->addParam(obj, "winSize", obj.winSize_);
                      obj.info()->addParam(obj, "maxLevel", obj.maxLevel_);
                      obj.info()->addParam(obj, "iterations", obj.iterations_));

    PyrLK_GPU::PyrLK_GPU() : GpuOpticalFlow(CV_8UC1)
    {
        winSize_ = alg_.winSize.width;
        maxLevel_ = alg_.maxLevel;
        iterations_ = alg_.iters;
    }

    void PyrLK_GPU::impl(const GpuMat& input0, const GpuMat& input1, GpuMat& dst1, GpuMat& dst2)
    {
        alg_.winSize.width = winSize_;
        alg_.winSize.height = winSize_;
        alg_.maxLevel = maxLevel_;
        alg_.iters = iterations_;

        alg_.dense(input0, input1, dst1, dst2);
    }

    void PyrLK_GPU::collectGarbage()
    {
        alg_.releaseMemory();
        GpuOpticalFlow::collectGarbage();
    }
}

Ptr<DenseOpticalFlowExt> cv::superres::createOptFlow_PyrLK_GPU()
{
    return new PyrLK_GPU;
}

///////////////////////////////////////////////////////////////////
// Farneback_GPU

namespace
{
    class Farneback_GPU : public GpuOpticalFlow
    {
    public:
        AlgorithmInfo* info() const;

        Farneback_GPU();

        void collectGarbage();

    protected:
        void impl(const GpuMat& input0, const GpuMat& input1, GpuMat& dst1, GpuMat& dst2);

    private:
        double pyrScale_;
        int numLevels_;
        int winSize_;
        int numIters_;
        int polyN_;
        double polySigma_;
        int flags_;

        FarnebackOpticalFlow alg_;
    };

    CV_INIT_ALGORITHM(Farneback_GPU, "DenseOpticalFlowExt.Farneback_GPU",
                      obj.info()->addParam(obj, "pyrScale", obj.pyrScale_);
                      obj.info()->addParam(obj, "numLevels", obj.numLevels_);
                      obj.info()->addParam(obj, "winSize", obj.winSize_);
                      obj.info()->addParam(obj, "numIters", obj.numIters_);
                      obj.info()->addParam(obj, "polyN", obj.polyN_);
                      obj.info()->addParam(obj, "polySigma", obj.polySigma_);
                      obj.info()->addParam(obj, "flags", obj.flags_));

    Farneback_GPU::Farneback_GPU() : GpuOpticalFlow(CV_8UC1)
    {
        pyrScale_ = alg_.pyrScale;
        numLevels_ = alg_.numLevels;
        winSize_ = alg_.winSize;
        numIters_ = alg_.numIters;
        polyN_ = alg_.polyN;
        polySigma_ = alg_.polySigma;
        flags_ = alg_.flags;
    }

    void Farneback_GPU::impl(const GpuMat& input0, const GpuMat& input1, GpuMat& dst1, GpuMat& dst2)
    {
        alg_.pyrScale = pyrScale_;
        alg_.numLevels = numLevels_;
        alg_.winSize = winSize_;
        alg_.numIters = numIters_;
        alg_.polyN = polyN_;
        alg_.polySigma = polySigma_;
        alg_.flags = flags_;

        alg_(input0, input1, dst1, dst2);
    }

    void Farneback_GPU::collectGarbage()
    {
        alg_.releaseMemory();
        GpuOpticalFlow::collectGarbage();
    }
}

Ptr<DenseOpticalFlowExt> cv::superres::createOptFlow_Farneback_GPU()
{
    return new Farneback_GPU;
}

///////////////////////////////////////////////////////////////////
// DualTVL1_GPU

namespace
{
    class DualTVL1_GPU : public GpuOpticalFlow
    {
    public:
        AlgorithmInfo* info() const;

        DualTVL1_GPU();

        void collectGarbage();

    protected:
        void impl(const GpuMat& input0, const GpuMat& input1, GpuMat& dst1, GpuMat& dst2);

    private:
        double tau_;
        double lambda_;
        double theta_;
        int nscales_;
        int warps_;
        double epsilon_;
        int iterations_;
        bool useInitialFlow_;

        OpticalFlowDual_TVL1_GPU alg_;
    };

    CV_INIT_ALGORITHM(DualTVL1_GPU, "DenseOpticalFlowExt.DualTVL1_GPU",
                      obj.info()->addParam(obj, "tau", obj.tau_);
                      obj.info()->addParam(obj, "lambda", obj.lambda_);
                      obj.info()->addParam(obj, "theta", obj.theta_);
                      obj.info()->addParam(obj, "nscales", obj.nscales_);
                      obj.info()->addParam(obj, "warps", obj.warps_);
                      obj.info()->addParam(obj, "epsilon", obj.epsilon_);
                      obj.info()->addParam(obj, "iterations", obj.iterations_);
                      obj.info()->addParam(obj, "useInitialFlow", obj.useInitialFlow_));

    DualTVL1_GPU::DualTVL1_GPU() : GpuOpticalFlow(CV_8UC1)
    {
        tau_ = alg_.tau;
        lambda_ = alg_.lambda;
        theta_ = alg_.theta;
        nscales_ = alg_.nscales;
        warps_ = alg_.warps;
        epsilon_ = alg_.epsilon;
        iterations_ = alg_.iterations;
        useInitialFlow_ = alg_.useInitialFlow;
    }

    void DualTVL1_GPU::impl(const GpuMat& input0, const GpuMat& input1, GpuMat& dst1, GpuMat& dst2)
    {
        alg_.tau = tau_;
        alg_.lambda = lambda_;
        alg_.theta = theta_;
        alg_.nscales = nscales_;
        alg_.warps = warps_;
        alg_.epsilon = epsilon_;
        alg_.iterations = iterations_;
        alg_.useInitialFlow = useInitialFlow_;

        alg_(input0, input1, dst1, dst2);
    }

    void DualTVL1_GPU::collectGarbage()
    {
        alg_.collectGarbage();
        GpuOpticalFlow::collectGarbage();
    }
}

Ptr<DenseOpticalFlowExt> cv::superres::createOptFlow_DualTVL1_GPU()
{
    return new DualTVL1_GPU;
}

#endif // HAVE_OPENCV_GPU
#ifdef HAVE_OPENCV_OCL

namespace
{
    class oclOpticalFlow : public DenseOpticalFlowExt
    {
    public:
        explicit oclOpticalFlow(int work_type);

        void calc(InputArray frame0, InputArray frame1, OutputArray flow1, OutputArray flow2);
        void collectGarbage();

    protected:
        virtual void impl(const cv::ocl::oclMat& input0, const cv::ocl::oclMat& input1, cv::ocl::oclMat& dst1, cv::ocl::oclMat& dst2) = 0;

    private:
        int work_type_;
        cv::ocl::oclMat buf_[6];
        cv::ocl::oclMat u_, v_, flow_;
    };

    oclOpticalFlow::oclOpticalFlow(int work_type) : work_type_(work_type)
    {
    }

    void oclOpticalFlow::calc(InputArray frame0, InputArray frame1, OutputArray flow1, OutputArray flow2)
    {
        ocl::oclMat& _frame0 = ocl::getOclMatRef(frame0);
        ocl::oclMat& _frame1 = ocl::getOclMatRef(frame1);
        ocl::oclMat& _flow1  = ocl::getOclMatRef(flow1);
        ocl::oclMat& _flow2  = ocl::getOclMatRef(flow2);

        CV_Assert( _frame1.type() == _frame0.type() );
        CV_Assert( _frame1.size() == _frame0.size() );

        cv::ocl::oclMat input0_ = convertToType(_frame0, work_type_, buf_[2], buf_[3]);
        cv::ocl::oclMat input1_ = convertToType(_frame1, work_type_, buf_[4], buf_[5]);

        impl(input0_, input1_, u_, v_);//go to tvl1 algorithm

        u_.copyTo(_flow1);
        v_.copyTo(_flow2);
    }

    void oclOpticalFlow::collectGarbage()
    {
        for (int i = 0; i < 6; ++i)
            buf_[i].release();
        u_.release();
        v_.release();
        flow_.release();
    }
}
///////////////////////////////////////////////////////////////////
// PyrLK_OCL

namespace
{
    class PyrLK_OCL : public oclOpticalFlow
    {
    public:
        AlgorithmInfo* info() const;

        PyrLK_OCL();

        void collectGarbage();

    protected:
        void impl(const ocl::oclMat& input0, const ocl::oclMat& input1, ocl::oclMat& dst1, ocl::oclMat& dst2);

    private:
        int winSize_;
        int maxLevel_;
        int iterations_;

        ocl::PyrLKOpticalFlow alg_;
    };

    CV_INIT_ALGORITHM(PyrLK_OCL, "DenseOpticalFlowExt.PyrLK_OCL",
        obj.info()->addParam(obj, "winSize", obj.winSize_);
    obj.info()->addParam(obj, "maxLevel", obj.maxLevel_);
    obj.info()->addParam(obj, "iterations", obj.iterations_));

    PyrLK_OCL::PyrLK_OCL() : oclOpticalFlow(CV_8UC1)
    {
        winSize_ = alg_.winSize.width;
        maxLevel_ = alg_.maxLevel;
        iterations_ = alg_.iters;
    }

    void PyrLK_OCL::impl(const cv::ocl::oclMat& input0, const cv::ocl::oclMat& input1, cv::ocl::oclMat& dst1, cv::ocl::oclMat& dst2)
    {
        alg_.winSize.width = winSize_;
        alg_.winSize.height = winSize_;
        alg_.maxLevel = maxLevel_;
        alg_.iters = iterations_;

        alg_.dense(input0, input1, dst1, dst2);
    }

    void PyrLK_OCL::collectGarbage()
    {
        alg_.releaseMemory();
        oclOpticalFlow::collectGarbage();
    }
}

Ptr<DenseOpticalFlowExt> cv::superres::createOptFlow_PyrLK_OCL()
{
    return new PyrLK_OCL;
}

///////////////////////////////////////////////////////////////////
// DualTVL1_OCL

namespace
{
    class DualTVL1_OCL : public oclOpticalFlow
    {
    public:
        AlgorithmInfo* info() const;

        DualTVL1_OCL();

        void collectGarbage();

    protected:
        void impl(const cv::ocl::oclMat& input0, const cv::ocl::oclMat& input1, cv::ocl::oclMat& dst1, cv::ocl::oclMat& dst2);

    private:
        double tau_;
        double lambda_;
        double theta_;
        int nscales_;
        int warps_;
        double epsilon_;
        int iterations_;
        bool useInitialFlow_;

        ocl::OpticalFlowDual_TVL1_OCL alg_;
    };

    CV_INIT_ALGORITHM(DualTVL1_OCL, "DenseOpticalFlowExt.DualTVL1_OCL",
    obj.info()->addParam(obj, "tau", obj.tau_);
    obj.info()->addParam(obj, "lambda", obj.lambda_);
    obj.info()->addParam(obj, "theta", obj.theta_);
    obj.info()->addParam(obj, "nscales", obj.nscales_);
    obj.info()->addParam(obj, "warps", obj.warps_);
    obj.info()->addParam(obj, "epsilon", obj.epsilon_);
    obj.info()->addParam(obj, "iterations", obj.iterations_);
    obj.info()->addParam(obj, "useInitialFlow", obj.useInitialFlow_));

    DualTVL1_OCL::DualTVL1_OCL() : oclOpticalFlow(CV_8UC1)
    {
        tau_ = alg_.tau;
        lambda_ = alg_.lambda;
        theta_ = alg_.theta;
        nscales_ = alg_.nscales;
        warps_ = alg_.warps;
        epsilon_ = alg_.epsilon;
        iterations_ = alg_.iterations;
        useInitialFlow_ = alg_.useInitialFlow;
    }

    void DualTVL1_OCL::impl(const cv::ocl::oclMat& input0, const cv::ocl::oclMat& input1, cv::ocl::oclMat& dst1, cv::ocl::oclMat& dst2)
    {
        alg_.tau = tau_;
        alg_.lambda = lambda_;
        alg_.theta = theta_;
        alg_.nscales = nscales_;
        alg_.warps = warps_;
        alg_.epsilon = epsilon_;
        alg_.iterations = iterations_;
        alg_.useInitialFlow = useInitialFlow_;

        alg_(input0, input1, dst1, dst2);

    }

    void DualTVL1_OCL::collectGarbage()
    {
        alg_.collectGarbage();
        oclOpticalFlow::collectGarbage();
    }
}

Ptr<DenseOpticalFlowExt> cv::superres::createOptFlow_DualTVL1_OCL()
{
    return new DualTVL1_OCL;
}

///////////////////////////////////////////////////////////////////
// FarneBack

namespace
{
    class FarneBack_OCL : public oclOpticalFlow
    {
    public:
        AlgorithmInfo* info() const;

        FarneBack_OCL();

        void collectGarbage();

    protected:
        void impl(const cv::ocl::oclMat& input0, const cv::ocl::oclMat& input1, cv::ocl::oclMat& dst1, cv::ocl::oclMat& dst2);

    private:
        double pyrScale_;
        int numLevels_;
        int winSize_;
        int numIters_;
        int polyN_;
        double polySigma_;
        int flags_;

        ocl::FarnebackOpticalFlow alg_;
    };

    CV_INIT_ALGORITHM(FarneBack_OCL, "DenseOpticalFlowExt.FarneBack_OCL",
        obj.info()->addParam(obj, "pyrScale", obj.pyrScale_);
    obj.info()->addParam(obj, "numLevels", obj.numLevels_);
    obj.info()->addParam(obj, "winSize", obj.winSize_);
    obj.info()->addParam(obj, "numIters", obj.numIters_);
    obj.info()->addParam(obj, "polyN", obj.polyN_);
    obj.info()->addParam(obj, "polySigma", obj.polySigma_);
    obj.info()->addParam(obj, "flags", obj.flags_));

    FarneBack_OCL::FarneBack_OCL() : oclOpticalFlow(CV_8UC1)
    {
        pyrScale_ = alg_.pyrScale;
        numLevels_ = alg_.numLevels;
        winSize_ = alg_.winSize;
        numIters_ = alg_.numIters;
        polyN_ = alg_.polyN;
        polySigma_ = alg_.polySigma;
        flags_ = alg_.flags;
    }

    void FarneBack_OCL::impl(const cv::ocl::oclMat& input0, const cv::ocl::oclMat& input1, cv::ocl::oclMat& dst1, cv::ocl::oclMat& dst2)
    {
        alg_.pyrScale = pyrScale_;
        alg_.numLevels = numLevels_;
        alg_.winSize = winSize_;
        alg_.numIters = numIters_;
        alg_.polyN = polyN_;
        alg_.polySigma = polySigma_;
        alg_.flags = flags_;

        alg_(input0, input1, dst1, dst2);
    }

    void FarneBack_OCL::collectGarbage()
    {
        alg_.releaseMemory();
        oclOpticalFlow::collectGarbage();
    }
}

Ptr<DenseOpticalFlowExt> cv::superres::createOptFlow_Farneback_OCL()
{
    return new FarneBack_OCL;
}

#endif
