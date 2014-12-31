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

#if !defined HAVE_CUDA || defined(CUDA_DISABLER)

Ptr<OpticalFlowDual_TVL1> cv::cuda::OpticalFlowDual_TVL1::create(double, double, double, int, int, double, int, double, double, bool) { throw_no_cuda(); return Ptr<OpticalFlowDual_TVL1>(); }

#else

using namespace cv;
using namespace cv::cuda;

namespace tvl1flow
{
    void centeredGradient(PtrStepSzf src, PtrStepSzf dx, PtrStepSzf dy, cudaStream_t stream);
    void warpBackward(PtrStepSzf I0, PtrStepSzf I1, PtrStepSzf I1x, PtrStepSzf I1y,
                      PtrStepSzf u1, PtrStepSzf u2,
                      PtrStepSzf I1w, PtrStepSzf I1wx, PtrStepSzf I1wy,
                      PtrStepSzf grad, PtrStepSzf rho,
                      cudaStream_t stream);
    void estimateU(PtrStepSzf I1wx, PtrStepSzf I1wy,
                   PtrStepSzf grad, PtrStepSzf rho_c,
                   PtrStepSzf p11, PtrStepSzf p12, PtrStepSzf p21, PtrStepSzf p22, PtrStepSzf p31, PtrStepSzf p32,
                   PtrStepSzf u1, PtrStepSzf u2, PtrStepSzf u3, PtrStepSzf error,
                   float l_t, float theta, float gamma, bool calcError,
                   cudaStream_t stream);
    void estimateDualVariables(PtrStepSzf u1, PtrStepSzf u2, PtrStepSzf u3,
                               PtrStepSzf p11, PtrStepSzf p12, PtrStepSzf p21, PtrStepSzf p22, PtrStepSzf p31, PtrStepSzf p32,
                               float taut, float gamma,
                               cudaStream_t stream);
}

namespace
{
    class OpticalFlowDual_TVL1_Impl : public OpticalFlowDual_TVL1
    {
    public:
        OpticalFlowDual_TVL1_Impl(double tau, double lambda, double theta, int nscales, int warps, double epsilon,
                                  int iterations, double scaleStep, double gamma, bool useInitialFlow) :
            tau_(tau), lambda_(lambda), gamma_(gamma), theta_(theta), nscales_(nscales), warps_(warps),
            epsilon_(epsilon), iterations_(iterations), scaleStep_(scaleStep), useInitialFlow_(useInitialFlow)
        {
        }

        virtual double getTau() const { return tau_; }
        virtual void setTau(double tau) { tau_ = tau; }

        virtual double getLambda() const { return lambda_; }
        virtual void setLambda(double lambda) { lambda_ = lambda; }

        virtual double getGamma() const { return gamma_; }
        virtual void setGamma(double gamma) { gamma_ = gamma; }

        virtual double getTheta() const { return theta_; }
        virtual void setTheta(double theta) { theta_ = theta; }

        virtual int getNumScales() const { return nscales_; }
        virtual void setNumScales(int nscales) { nscales_ = nscales; }

        virtual int getNumWarps() const { return warps_; }
        virtual void setNumWarps(int warps) { warps_ = warps; }

        virtual double getEpsilon() const { return epsilon_; }
        virtual void setEpsilon(double epsilon) { epsilon_ = epsilon; }

        virtual int getNumIterations() const { return iterations_; }
        virtual void setNumIterations(int iterations) { iterations_ = iterations; }

        virtual double getScaleStep() const { return scaleStep_; }
        virtual void setScaleStep(double scaleStep) { scaleStep_ = scaleStep; }

        virtual bool getUseInitialFlow() const { return useInitialFlow_; }
        virtual void setUseInitialFlow(bool useInitialFlow) { useInitialFlow_ = useInitialFlow; }

        virtual void calc(InputArray I0, InputArray I1, InputOutputArray flow, Stream& stream);

    private:
        double tau_;
        double lambda_;
        double gamma_;
        double theta_;
        int nscales_;
        int warps_;
        double epsilon_;
        int iterations_;
        double scaleStep_;
        bool useInitialFlow_;

    private:
        void calcImpl(const GpuMat& I0, const GpuMat& I1, GpuMat& flowx, GpuMat& flowy, Stream& stream);
        void procOneScale(const GpuMat& I0, const GpuMat& I1, GpuMat& u1, GpuMat& u2, GpuMat& u3, Stream& stream);

        std::vector<GpuMat> I0s;
        std::vector<GpuMat> I1s;
        std::vector<GpuMat> u1s;
        std::vector<GpuMat> u2s;
        std::vector<GpuMat> u3s;

        GpuMat I1x_buf;
        GpuMat I1y_buf;

        GpuMat I1w_buf;
        GpuMat I1wx_buf;
        GpuMat I1wy_buf;

        GpuMat grad_buf;
        GpuMat rho_c_buf;

        GpuMat p11_buf;
        GpuMat p12_buf;
        GpuMat p21_buf;
        GpuMat p22_buf;
        GpuMat p31_buf;
        GpuMat p32_buf;

        GpuMat diff_buf;
        GpuMat norm_buf;
    };

    void OpticalFlowDual_TVL1_Impl::calc(InputArray _frame0, InputArray _frame1, InputOutputArray _flow, Stream& stream)
    {
        const GpuMat frame0 = _frame0.getGpuMat();
        const GpuMat frame1 = _frame1.getGpuMat();

        BufferPool pool(stream);
        GpuMat flowx = pool.getBuffer(frame0.size(), CV_32FC1);
        GpuMat flowy = pool.getBuffer(frame0.size(), CV_32FC1);

        calcImpl(frame0, frame1, flowx, flowy, stream);

        GpuMat flows[] = {flowx, flowy};
        cuda::merge(flows, 2, _flow, stream);
    }

    void OpticalFlowDual_TVL1_Impl::calcImpl(const GpuMat& I0, const GpuMat& I1, GpuMat& flowx, GpuMat& flowy, Stream& stream)
    {
        CV_Assert( I0.type() == CV_8UC1 || I0.type() == CV_32FC1 );
        CV_Assert( I0.size() == I1.size() );
        CV_Assert( I0.type() == I1.type() );
        CV_Assert( !useInitialFlow_ || (flowx.size() == I0.size() && flowx.type() == CV_32FC1 && flowy.size() == flowx.size() && flowy.type() == flowx.type()) );
        CV_Assert( nscales_ > 0 );

        // allocate memory for the pyramid structure
        I0s.resize(nscales_);
        I1s.resize(nscales_);
        u1s.resize(nscales_);
        u2s.resize(nscales_);
        u3s.resize(nscales_);

        I0.convertTo(I0s[0], CV_32F, I0.depth() == CV_8U ? 1.0 : 255.0, stream);
        I1.convertTo(I1s[0], CV_32F, I1.depth() == CV_8U ? 1.0 : 255.0, stream);

        if (!useInitialFlow_)
        {
            flowx.create(I0.size(), CV_32FC1);
            flowy.create(I0.size(), CV_32FC1);
        }

        u1s[0] = flowx;
        u2s[0] = flowy;
        if (gamma_)
        {
            u3s[0].create(I0.size(), CV_32FC1);
        }

        I1x_buf.create(I0.size(), CV_32FC1);
        I1y_buf.create(I0.size(), CV_32FC1);

        I1w_buf.create(I0.size(), CV_32FC1);
        I1wx_buf.create(I0.size(), CV_32FC1);
        I1wy_buf.create(I0.size(), CV_32FC1);

        grad_buf.create(I0.size(), CV_32FC1);
        rho_c_buf.create(I0.size(), CV_32FC1);

        p11_buf.create(I0.size(), CV_32FC1);
        p12_buf.create(I0.size(), CV_32FC1);
        p21_buf.create(I0.size(), CV_32FC1);
        p22_buf.create(I0.size(), CV_32FC1);
        if (gamma_)
        {
            p31_buf.create(I0.size(), CV_32FC1);
            p32_buf.create(I0.size(), CV_32FC1);
        }
        diff_buf.create(I0.size(), CV_32FC1);

        // create the scales
        for (int s = 1; s < nscales_; ++s)
        {
            cuda::resize(I0s[s-1], I0s[s], Size(), scaleStep_, scaleStep_, INTER_LINEAR, stream);
            cuda::resize(I1s[s-1], I1s[s], Size(), scaleStep_, scaleStep_, INTER_LINEAR, stream);

            if (I0s[s].cols < 16 || I0s[s].rows < 16)
            {
                nscales_ = s;
                break;
            }

            if (useInitialFlow_)
            {
                cuda::resize(u1s[s-1], u1s[s], Size(), scaleStep_, scaleStep_, INTER_LINEAR, stream);
                cuda::resize(u2s[s-1], u2s[s], Size(), scaleStep_, scaleStep_, INTER_LINEAR, stream);

                cuda::multiply(u1s[s], Scalar::all(scaleStep_), u1s[s], 1, -1, stream);
                cuda::multiply(u2s[s], Scalar::all(scaleStep_), u2s[s], 1, -1, stream);
            }
            else
            {
                u1s[s].create(I0s[s].size(), CV_32FC1);
                u2s[s].create(I0s[s].size(), CV_32FC1);
            }
            if (gamma_)
            {
                u3s[s].create(I0s[s].size(), CV_32FC1);
            }
        }

        if (!useInitialFlow_)
        {
            u1s[nscales_-1].setTo(Scalar::all(0), stream);
            u2s[nscales_-1].setTo(Scalar::all(0), stream);
        }
        if (gamma_)
        {
            u3s[nscales_ - 1].setTo(Scalar::all(0), stream);
        }

        // pyramidal structure for computing the optical flow
        for (int s = nscales_ - 1; s >= 0; --s)
        {
            // compute the optical flow at the current scale
            procOneScale(I0s[s], I1s[s], u1s[s], u2s[s], u3s[s], stream);

            // if this was the last scale, finish now
            if (s == 0)
                break;

            // otherwise, upsample the optical flow

            // zoom the optical flow for the next finer scale
            cuda::resize(u1s[s], u1s[s - 1], I0s[s - 1].size(), 0, 0, INTER_LINEAR, stream);
            cuda::resize(u2s[s], u2s[s - 1], I0s[s - 1].size(), 0, 0, INTER_LINEAR, stream);
            if (gamma_)
            {
                cuda::resize(u3s[s], u3s[s - 1], I0s[s - 1].size(), 0, 0, INTER_LINEAR, stream);
            }

            // scale the optical flow with the appropriate zoom factor
            cuda::multiply(u1s[s - 1], Scalar::all(1/scaleStep_), u1s[s - 1], 1, -1, stream);
            cuda::multiply(u2s[s - 1], Scalar::all(1/scaleStep_), u2s[s - 1], 1, -1, stream);
        }
    }

    void OpticalFlowDual_TVL1_Impl::procOneScale(const GpuMat& I0, const GpuMat& I1, GpuMat& u1, GpuMat& u2, GpuMat& u3, Stream& _stream)
    {
        using namespace tvl1flow;

        cudaStream_t stream = StreamAccessor::getStream(_stream);

        const double scaledEpsilon = epsilon_ * epsilon_ * I0.size().area();

        CV_DbgAssert( I1.size() == I0.size() );
        CV_DbgAssert( I1.type() == I0.type() );
        CV_DbgAssert( u1.size() == I0.size() );
        CV_DbgAssert( u2.size() == u1.size() );

        GpuMat I1x = I1x_buf(Rect(0, 0, I0.cols, I0.rows));
        GpuMat I1y = I1y_buf(Rect(0, 0, I0.cols, I0.rows));
        centeredGradient(I1, I1x, I1y, stream);

        GpuMat I1w = I1w_buf(Rect(0, 0, I0.cols, I0.rows));
        GpuMat I1wx = I1wx_buf(Rect(0, 0, I0.cols, I0.rows));
        GpuMat I1wy = I1wy_buf(Rect(0, 0, I0.cols, I0.rows));

        GpuMat grad = grad_buf(Rect(0, 0, I0.cols, I0.rows));
        GpuMat rho_c = rho_c_buf(Rect(0, 0, I0.cols, I0.rows));

        GpuMat p11 = p11_buf(Rect(0, 0, I0.cols, I0.rows));
        GpuMat p12 = p12_buf(Rect(0, 0, I0.cols, I0.rows));
        GpuMat p21 = p21_buf(Rect(0, 0, I0.cols, I0.rows));
        GpuMat p22 = p22_buf(Rect(0, 0, I0.cols, I0.rows));
        GpuMat p31, p32;
        if (gamma_)
        {
            p31 = p31_buf(Rect(0, 0, I0.cols, I0.rows));
            p32 = p32_buf(Rect(0, 0, I0.cols, I0.rows));
        }
        p11.setTo(Scalar::all(0), _stream);
        p12.setTo(Scalar::all(0), _stream);
        p21.setTo(Scalar::all(0), _stream);
        p22.setTo(Scalar::all(0), _stream);
        if (gamma_)
        {
            p31.setTo(Scalar::all(0), _stream);
            p32.setTo(Scalar::all(0), _stream);
        }

        GpuMat diff = diff_buf(Rect(0, 0, I0.cols, I0.rows));

        const float l_t = static_cast<float>(lambda_ * theta_);
        const float taut = static_cast<float>(tau_ / theta_);

        for (int warpings = 0; warpings < warps_; ++warpings)
        {
            warpBackward(I0, I1, I1x, I1y, u1, u2, I1w, I1wx, I1wy, grad, rho_c, stream);

            double error = std::numeric_limits<double>::max();
            double prevError = 0.0;
            for (int n = 0; error > scaledEpsilon && n < iterations_; ++n)
            {
                // some tweaks to make sum operation less frequently
                bool calcError = (epsilon_ > 0) && (n & 0x1) && (prevError < scaledEpsilon);
                estimateU(I1wx, I1wy, grad, rho_c, p11, p12, p21, p22, p31, p32, u1, u2, u3, diff, l_t, static_cast<float>(theta_), gamma_, calcError, stream);
                if (calcError)
                {
                    _stream.waitForCompletion();
                    error = cuda::sum(diff, norm_buf)[0];
                    prevError = error;
                }
                else
                {
                    error = std::numeric_limits<double>::max();
                    prevError -= scaledEpsilon;
                }

                estimateDualVariables(u1, u2, u3, p11, p12, p21, p22, p31, p32, taut, gamma_, stream);
            }
        }
    }
}

Ptr<OpticalFlowDual_TVL1> cv::cuda::OpticalFlowDual_TVL1::create(
            double tau, double lambda, double theta, int nscales, int warps,
            double epsilon, int iterations, double scaleStep, double gamma, bool useInitialFlow)
{
    return makePtr<OpticalFlowDual_TVL1_Impl>(tau, lambda, theta, nscales, warps,
                                              epsilon, iterations, scaleStep, gamma, useInitialFlow);
}

#endif // !defined HAVE_CUDA || defined(CUDA_DISABLER)
