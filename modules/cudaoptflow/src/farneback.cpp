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

#define MIN_SIZE 32

#define S(x) StreamAccessor::getStream(x)

// CUDA resize() is fast, but it differs from the CPU analog. Disabling this flag
// leads to an inefficient code. It's for debug purposes only.
#define ENABLE_CUDA_RESIZE 1

using namespace cv;
using namespace cv::cuda;

#if !defined HAVE_CUDA || defined(CUDA_DISABLER)

void cv::cuda::FarnebackOpticalFlow::operator ()(const GpuMat&, const GpuMat&, GpuMat&, GpuMat&, Stream&) { throw_no_cuda(); }

#else

namespace cv { namespace cuda { namespace device { namespace optflow_farneback
{
    void setPolynomialExpansionConsts(
            int polyN, const float *g, const float *xg, const float *xxg,
            float ig11, float ig03, float ig33, float ig55);

    void polynomialExpansionGpu(const PtrStepSzf &src, int polyN, PtrStepSzf dst, cudaStream_t stream);

    void setUpdateMatricesConsts();

    void updateMatricesGpu(
            const PtrStepSzf flowx, const PtrStepSzf flowy, const PtrStepSzf R0, const PtrStepSzf R1,
            PtrStepSzf M, cudaStream_t stream);

    void updateFlowGpu(
            const PtrStepSzf M, PtrStepSzf flowx, PtrStepSzf flowy, cudaStream_t stream);

    /*void boxFilterGpu(const PtrStepSzf src, int ksizeHalf, PtrStepSzf dst, cudaStream_t stream);*/

    void boxFilter5Gpu(const PtrStepSzf src, int ksizeHalf, PtrStepSzf dst, cudaStream_t stream);

    void boxFilter5Gpu_CC11(const PtrStepSzf src, int ksizeHalf, PtrStepSzf dst, cudaStream_t stream);

    void setGaussianBlurKernel(const float *gKer, int ksizeHalf);

    void gaussianBlurGpu(
            const PtrStepSzf src, int ksizeHalf, PtrStepSzf dst, int borderType, cudaStream_t stream);

    void gaussianBlur5Gpu(
            const PtrStepSzf src, int ksizeHalf, PtrStepSzf dst, int borderType, cudaStream_t stream);

    void gaussianBlur5Gpu_CC11(
            const PtrStepSzf src, int ksizeHalf, PtrStepSzf dst, int borderType, cudaStream_t stream);

}}}} // namespace cv { namespace cuda { namespace cudev { namespace optflow_farneback


void cv::cuda::FarnebackOpticalFlow::prepareGaussian(
        int n, double sigma, float *g, float *xg, float *xxg,
        double &ig11, double &ig03, double &ig33, double &ig55)
{
    double s = 0.;
    for (int x = -n; x <= n; x++)
    {
        g[x] = (float)std::exp(-x*x/(2*sigma*sigma));
        s += g[x];
    }

    s = 1./s;
    for (int x = -n; x <= n; x++)
    {
        g[x] = (float)(g[x]*s);
        xg[x] = (float)(x*g[x]);
        xxg[x] = (float)(x*x*g[x]);
    }

    Mat_<double> G(6, 6);
    G.setTo(0);

    for (int y = -n; y <= n; y++)
    {
        for (int x = -n; x <= n; x++)
        {
            G(0,0) += g[y]*g[x];
            G(1,1) += g[y]*g[x]*x*x;
            G(3,3) += g[y]*g[x]*x*x*x*x;
            G(5,5) += g[y]*g[x]*x*x*y*y;
        }
    }

    //G[0][0] = 1.;
    G(2,2) = G(0,3) = G(0,4) = G(3,0) = G(4,0) = G(1,1);
    G(4,4) = G(3,3);
    G(3,4) = G(4,3) = G(5,5);

    // invG:
    // [ x        e  e    ]
    // [    y             ]
    // [       y          ]
    // [ e        z       ]
    // [ e           z    ]
    // [                u ]
    Mat_<double> invG = G.inv(DECOMP_CHOLESKY);

    ig11 = invG(1,1);
    ig03 = invG(0,3);
    ig33 = invG(3,3);
    ig55 = invG(5,5);
}


void cv::cuda::FarnebackOpticalFlow::setPolynomialExpansionConsts(int n, double sigma)
{
    std::vector<float> buf(n*6 + 3);
    float* g = &buf[0] + n;
    float* xg = g + n*2 + 1;
    float* xxg = xg + n*2 + 1;

    if (sigma < FLT_EPSILON)
        sigma = n*0.3;

    double ig11, ig03, ig33, ig55;
    prepareGaussian(n, sigma, g, xg, xxg, ig11, ig03, ig33, ig55);

    device::optflow_farneback::setPolynomialExpansionConsts(n, g, xg, xxg, static_cast<float>(ig11), static_cast<float>(ig03), static_cast<float>(ig33), static_cast<float>(ig55));
}


void cv::cuda::FarnebackOpticalFlow::updateFlow_boxFilter(
        const GpuMat& R0, const GpuMat& R1, GpuMat& flowx, GpuMat &flowy,
        GpuMat& M, GpuMat &bufM, int blockSize, bool updateMatrices, Stream streams[])
{
    if (deviceSupports(FEATURE_SET_COMPUTE_12))
        device::optflow_farneback::boxFilter5Gpu(M, blockSize/2, bufM, S(streams[0]));
    else
        device::optflow_farneback::boxFilter5Gpu_CC11(M, blockSize/2, bufM, S(streams[0]));
    swap(M, bufM);

    for (int i = 1; i < 5; ++i)
        streams[i].waitForCompletion();
    device::optflow_farneback::updateFlowGpu(M, flowx, flowy, S(streams[0]));

    if (updateMatrices)
        device::optflow_farneback::updateMatricesGpu(flowx, flowy, R0, R1, M, S(streams[0]));
}


void cv::cuda::FarnebackOpticalFlow::updateFlow_gaussianBlur(
        const GpuMat& R0, const GpuMat& R1, GpuMat& flowx, GpuMat& flowy,
        GpuMat& M, GpuMat &bufM, int blockSize, bool updateMatrices, Stream streams[])
{
    if (deviceSupports(FEATURE_SET_COMPUTE_12))
        device::optflow_farneback::gaussianBlur5Gpu(
                    M, blockSize/2, bufM, BORDER_REPLICATE, S(streams[0]));
    else
        device::optflow_farneback::gaussianBlur5Gpu_CC11(
                    M, blockSize/2, bufM, BORDER_REPLICATE, S(streams[0]));
    swap(M, bufM);

    device::optflow_farneback::updateFlowGpu(M, flowx, flowy, S(streams[0]));

    if (updateMatrices)
        device::optflow_farneback::updateMatricesGpu(flowx, flowy, R0, R1, M, S(streams[0]));
}


void cv::cuda::FarnebackOpticalFlow::operator ()(
        const GpuMat &frame0, const GpuMat &frame1, GpuMat &flowx, GpuMat &flowy, Stream &s)
{
    CV_Assert(frame0.channels() == 1 && frame1.channels() == 1);
    CV_Assert(frame0.size() == frame1.size());
    CV_Assert(polyN == 5 || polyN == 7);
    CV_Assert(!fastPyramids || std::abs(pyrScale - 0.5) < 1e-6);

    Stream streams[5];
    if (S(s))
        streams[0] = s;

    Size size = frame0.size();
    GpuMat prevFlowX, prevFlowY, curFlowX, curFlowY;

    flowx.create(size, CV_32F);
    flowy.create(size, CV_32F);
    GpuMat flowx0 = flowx;
    GpuMat flowy0 = flowy;

    // Crop unnecessary levels
    double scale = 1;
    int numLevelsCropped = 0;
    for (; numLevelsCropped < numLevels; numLevelsCropped++)
    {
        scale *= pyrScale;
        if (size.width*scale < MIN_SIZE || size.height*scale < MIN_SIZE)
            break;
    }

    frame0.convertTo(frames_[0], CV_32F, streams[0]);
    frame1.convertTo(frames_[1], CV_32F, streams[1]);

    if (fastPyramids)
    {
        // Build Gaussian pyramids using pyrDown()
        pyramid0_.resize(numLevelsCropped + 1);
        pyramid1_.resize(numLevelsCropped + 1);
        pyramid0_[0] = frames_[0];
        pyramid1_[0] = frames_[1];
        for (int i = 1; i <= numLevelsCropped; ++i)
        {
            cuda::pyrDown(pyramid0_[i - 1], pyramid0_[i], streams[0]);
            cuda::pyrDown(pyramid1_[i - 1], pyramid1_[i], streams[1]);
        }
    }

    setPolynomialExpansionConsts(polyN, polySigma);
    device::optflow_farneback::setUpdateMatricesConsts();

    for (int k = numLevelsCropped; k >= 0; k--)
    {
        streams[0].waitForCompletion();

        scale = 1;
        for (int i = 0; i < k; i++)
            scale *= pyrScale;

        double sigma = (1./scale - 1) * 0.5;
        int smoothSize = cvRound(sigma*5) | 1;
        smoothSize = std::max(smoothSize, 3);

        int width = cvRound(size.width*scale);
        int height = cvRound(size.height*scale);

        if (fastPyramids)
        {
            width = pyramid0_[k].cols;
            height = pyramid0_[k].rows;
        }

        if (k > 0)
        {
            curFlowX.create(height, width, CV_32F);
            curFlowY.create(height, width, CV_32F);
        }
        else
        {
            curFlowX = flowx0;
            curFlowY = flowy0;
        }

        if (!prevFlowX.data)
        {
            if (flags & OPTFLOW_USE_INITIAL_FLOW)
            {
                cuda::resize(flowx0, curFlowX, Size(width, height), 0, 0, INTER_LINEAR, streams[0]);
                cuda::resize(flowy0, curFlowY, Size(width, height), 0, 0, INTER_LINEAR, streams[1]);
                curFlowX.convertTo(curFlowX, curFlowX.depth(), scale, streams[0]);
                curFlowY.convertTo(curFlowY, curFlowY.depth(), scale, streams[1]);
            }
            else
            {
                curFlowX.setTo(0, streams[0]);
                curFlowY.setTo(0, streams[1]);
            }
        }
        else
        {
            cuda::resize(prevFlowX, curFlowX, Size(width, height), 0, 0, INTER_LINEAR, streams[0]);
            cuda::resize(prevFlowY, curFlowY, Size(width, height), 0, 0, INTER_LINEAR, streams[1]);
            curFlowX.convertTo(curFlowX, curFlowX.depth(), 1./pyrScale, streams[0]);
            curFlowY.convertTo(curFlowY, curFlowY.depth(), 1./pyrScale, streams[1]);
        }

        GpuMat M = allocMatFromBuf(5*height, width, CV_32F, M_);
        GpuMat bufM = allocMatFromBuf(5*height, width, CV_32F, bufM_);
        GpuMat R[2] =
        {
            allocMatFromBuf(5*height, width, CV_32F, R_[0]),
            allocMatFromBuf(5*height, width, CV_32F, R_[1])
        };

        if (fastPyramids)
        {
            device::optflow_farneback::polynomialExpansionGpu(pyramid0_[k], polyN, R[0], S(streams[0]));
            device::optflow_farneback::polynomialExpansionGpu(pyramid1_[k], polyN, R[1], S(streams[1]));
        }
        else
        {
            GpuMat blurredFrame[2] =
            {
                allocMatFromBuf(size.height, size.width, CV_32F, blurredFrame_[0]),
                allocMatFromBuf(size.height, size.width, CV_32F, blurredFrame_[1])
            };
            GpuMat pyrLevel[2] =
            {
                allocMatFromBuf(height, width, CV_32F, pyrLevel_[0]),
                allocMatFromBuf(height, width, CV_32F, pyrLevel_[1])
            };

            Mat g = getGaussianKernel(smoothSize, sigma, CV_32F);
            device::optflow_farneback::setGaussianBlurKernel(g.ptr<float>(smoothSize/2), smoothSize/2);

            for (int i = 0; i < 2; i++)
            {
                device::optflow_farneback::gaussianBlurGpu(
                        frames_[i], smoothSize/2, blurredFrame[i], BORDER_REFLECT101, S(streams[i]));
                cuda::resize(blurredFrame[i], pyrLevel[i], Size(width, height), 0.0, 0.0, INTER_LINEAR, streams[i]);
                device::optflow_farneback::polynomialExpansionGpu(pyrLevel[i], polyN, R[i], S(streams[i]));
            }
        }

        streams[1].waitForCompletion();
        device::optflow_farneback::updateMatricesGpu(curFlowX, curFlowY, R[0], R[1], M, S(streams[0]));

        if (flags & OPTFLOW_FARNEBACK_GAUSSIAN)
        {
            Mat g = getGaussianKernel(winSize, winSize/2*0.3f, CV_32F);
            device::optflow_farneback::setGaussianBlurKernel(g.ptr<float>(winSize/2), winSize/2);
        }
        for (int i = 0; i < numIters; i++)
        {
            if (flags & OPTFLOW_FARNEBACK_GAUSSIAN)
                updateFlow_gaussianBlur(R[0], R[1], curFlowX, curFlowY, M, bufM, winSize, i < numIters-1, streams);
            else
                updateFlow_boxFilter(R[0], R[1], curFlowX, curFlowY, M, bufM, winSize, i < numIters-1, streams);
        }

        prevFlowX = curFlowX;
        prevFlowY = curFlowY;
    }

    flowx = curFlowX;
    flowy = curFlowY;

    if (!S(s))
        streams[0].waitForCompletion();
}

#endif
