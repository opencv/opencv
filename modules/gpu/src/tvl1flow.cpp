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

cv::gpu::OpticalFlowDual_TVL1_GPU::OpticalFlowDual_TVL1_GPU() { throw_nogpu(); }
void cv::gpu::OpticalFlowDual_TVL1_GPU::operator ()(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&) { throw_nogpu(); }
void cv::gpu::OpticalFlowDual_TVL1_GPU::collectGarbage() {}
void cv::gpu::OpticalFlowDual_TVL1_GPU::procOneScale(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&) { throw_nogpu(); }

#else

using namespace std;
using namespace cv;
using namespace cv::gpu;

cv::gpu::OpticalFlowDual_TVL1_GPU::OpticalFlowDual_TVL1_GPU()
{
    tau            = 0.25;
    lambda         = 0.15;
    theta          = 0.3;
    nscales        = 5;
    warps          = 5;
    epsilon        = 0.01;
    iterations     = 300;
    useInitialFlow = false;
}

void cv::gpu::OpticalFlowDual_TVL1_GPU::operator ()(const GpuMat& I0, const GpuMat& I1, GpuMat& flowx, GpuMat& flowy)
{
    CV_Assert( I0.type() == CV_8UC1 || I0.type() == CV_32FC1 );
    CV_Assert( I0.size() == I1.size() );
    CV_Assert( I0.type() == I1.type() );
    CV_Assert( !useInitialFlow || (flowx.size() == I0.size() && flowx.type() == CV_32FC1 && flowy.size() == flowx.size() && flowy.type() == flowx.type()) );
    CV_Assert( nscales > 0 );

    // allocate memory for the pyramid structure
    I0s.resize(nscales);
    I1s.resize(nscales);
    u1s.resize(nscales);
    u2s.resize(nscales);

    I0.convertTo(I0s[0], CV_32F, I0.depth() == CV_8U ? 1.0 : 255.0);
    I1.convertTo(I1s[0], CV_32F, I1.depth() == CV_8U ? 1.0 : 255.0);

    if (!useInitialFlow)
    {
        flowx.create(I0.size(), CV_32FC1);
        flowy.create(I0.size(), CV_32FC1);
    }

    u1s[0] = flowx;
    u2s[0] = flowy;

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

    diff_buf.create(I0.size(), CV_32FC1);

    // create the scales
    for (int s = 1; s < nscales; ++s)
    {
        gpu::pyrDown(I0s[s - 1], I0s[s]);
        gpu::pyrDown(I1s[s - 1], I1s[s]);

        if (I0s[s].cols < 16 || I0s[s].rows < 16)
        {
            nscales = s;
            break;
        }

        if (useInitialFlow)
        {
            gpu::pyrDown(u1s[s - 1], u1s[s]);
            gpu::pyrDown(u2s[s - 1], u2s[s]);

            gpu::multiply(u1s[s], Scalar::all(0.5), u1s[s]);
            gpu::multiply(u2s[s], Scalar::all(0.5), u2s[s]);
        }
        else
        {
            u1s[s].create(I0s[s].size(), CV_32FC1);
            u2s[s].create(I0s[s].size(), CV_32FC1);
        }
    }

    if (!useInitialFlow)
    {
        u1s[nscales-1].setTo(Scalar::all(0));
        u2s[nscales-1].setTo(Scalar::all(0));
    }

    // pyramidal structure for computing the optical flow
    for (int s = nscales - 1; s >= 0; --s)
    {
        // compute the optical flow at the current scale
        procOneScale(I0s[s], I1s[s], u1s[s], u2s[s]);

        // if this was the last scale, finish now
        if (s == 0)
            break;

        // otherwise, upsample the optical flow

        // zoom the optical flow for the next finer scale
        gpu::resize(u1s[s], u1s[s - 1], I0s[s - 1].size());
        gpu::resize(u2s[s], u2s[s - 1], I0s[s - 1].size());

        // scale the optical flow with the appropriate zoom factor
        gpu::multiply(u1s[s - 1], Scalar::all(2), u1s[s - 1]);
        gpu::multiply(u2s[s - 1], Scalar::all(2), u2s[s - 1]);
    }
}

namespace tvl1flow
{
    void centeredGradient(PtrStepSzf src, PtrStepSzf dx, PtrStepSzf dy);
    void warpBackward(PtrStepSzf I0, PtrStepSzf I1, PtrStepSzf I1x, PtrStepSzf I1y, PtrStepSzf u1, PtrStepSzf u2, PtrStepSzf I1w, PtrStepSzf I1wx, PtrStepSzf I1wy, PtrStepSzf grad, PtrStepSzf rho);
    void estimateU(PtrStepSzf I1wx, PtrStepSzf I1wy,
                   PtrStepSzf grad, PtrStepSzf rho_c,
                   PtrStepSzf p11, PtrStepSzf p12, PtrStepSzf p21, PtrStepSzf p22,
                   PtrStepSzf u1, PtrStepSzf u2, PtrStepSzf error,
                   float l_t, float theta, bool calcError);
    void estimateDualVariables(PtrStepSzf u1, PtrStepSzf u2, PtrStepSzf p11, PtrStepSzf p12, PtrStepSzf p21, PtrStepSzf p22, float taut);
}

void cv::gpu::OpticalFlowDual_TVL1_GPU::procOneScale(const GpuMat& I0, const GpuMat& I1, GpuMat& u1, GpuMat& u2)
{
    using namespace tvl1flow;

    const double scaledEpsilon = epsilon * epsilon * I0.size().area();

    CV_DbgAssert( I1.size() == I0.size() );
    CV_DbgAssert( I1.type() == I0.type() );
    CV_DbgAssert( u1.size() == I0.size() );
    CV_DbgAssert( u2.size() == u1.size() );

    GpuMat I1x = I1x_buf(Rect(0, 0, I0.cols, I0.rows));
    GpuMat I1y = I1y_buf(Rect(0, 0, I0.cols, I0.rows));
    centeredGradient(I1, I1x, I1y);

    GpuMat I1w = I1w_buf(Rect(0, 0, I0.cols, I0.rows));
    GpuMat I1wx = I1wx_buf(Rect(0, 0, I0.cols, I0.rows));
    GpuMat I1wy = I1wy_buf(Rect(0, 0, I0.cols, I0.rows));

    GpuMat grad = grad_buf(Rect(0, 0, I0.cols, I0.rows));
    GpuMat rho_c = rho_c_buf(Rect(0, 0, I0.cols, I0.rows));

    GpuMat p11 = p11_buf(Rect(0, 0, I0.cols, I0.rows));
    GpuMat p12 = p12_buf(Rect(0, 0, I0.cols, I0.rows));
    GpuMat p21 = p21_buf(Rect(0, 0, I0.cols, I0.rows));
    GpuMat p22 = p22_buf(Rect(0, 0, I0.cols, I0.rows));
    p11.setTo(Scalar::all(0));
    p12.setTo(Scalar::all(0));
    p21.setTo(Scalar::all(0));
    p22.setTo(Scalar::all(0));

    GpuMat diff = diff_buf(Rect(0, 0, I0.cols, I0.rows));

    const float l_t = static_cast<float>(lambda * theta);
    const float taut = static_cast<float>(tau / theta);

    for (int warpings = 0; warpings < warps; ++warpings)
    {
        warpBackward(I0, I1, I1x, I1y, u1, u2, I1w, I1wx, I1wy, grad, rho_c);

        double error = numeric_limits<double>::max();
        double prevError = 0.0;
        for (int n = 0; error > scaledEpsilon && n < iterations; ++n)
        {
            // some tweaks to make sum operation less frequently
            bool calcError = (epsilon > 0) && (n & 0x1) && (prevError < scaledEpsilon);

            estimateU(I1wx, I1wy, grad, rho_c, p11, p12, p21, p22, u1, u2, diff, l_t, static_cast<float>(theta), calcError);

            if (calcError)
            {
                error = gpu::sum(diff, norm_buf)[0];
                prevError = error;
            }
            else
            {
                error = numeric_limits<double>::max();
                prevError -= scaledEpsilon;
            }

            estimateDualVariables(u1, u2, p11, p12, p21, p22, taut);
        }
    }
}

void cv::gpu::OpticalFlowDual_TVL1_GPU::collectGarbage()
{
    I0s.clear();
    I1s.clear();
    u1s.clear();
    u2s.clear();

    I1x_buf.release();
    I1y_buf.release();

    I1w_buf.release();
    I1wx_buf.release();
    I1wy_buf.release();

    grad_buf.release();
    rho_c_buf.release();

    p11_buf.release();
    p12_buf.release();
    p21_buf.release();
    p22_buf.release();

    diff_buf.release();
    norm_buf.release();
}

#endif // !defined HAVE_CUDA || defined(CUDA_DISABLER)
