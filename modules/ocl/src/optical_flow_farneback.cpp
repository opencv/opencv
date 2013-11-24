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
//      Sen Liu, swjtuls1987@126.com
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
#include "opencl_kernels.hpp"
#include "opencv2/video/tracking.hpp"

using namespace cv;
using namespace cv::ocl;

#define MIN_SIZE 32

namespace cv {
namespace ocl {
namespace optflow_farneback
{
oclMat g;
oclMat xg;
oclMat xxg;
oclMat gKer;

float ig[4];

inline void setGaussianBlurKernel(const float *c_gKer, int ksizeHalf)
{
    cv::Mat t_gKer(1, ksizeHalf + 1, CV_32FC1, const_cast<float *>(c_gKer));
    gKer.upload(t_gKer);
}

static void gaussianBlurOcl(const oclMat &src, int ksizeHalf, oclMat &dst)
{
    String kernelName("gaussianBlur");
    size_t localThreads[3] = { 256, 1, 1 };
    size_t globalThreads[3] = { src.cols, src.rows, 1 };
    int smem_size = (localThreads[0] + 2*ksizeHalf) * sizeof(float);

    CV_Assert(dst.size() == src.size());
    std::vector< std::pair<size_t, const void *> > args;
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&dst.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&src.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&gKer.data));
    args.push_back(std::make_pair(smem_size, (void *)NULL));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.rows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.cols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.step));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.step));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&ksizeHalf));

    openCLExecuteKernel(Context::getContext(), &optical_flow_farneback, kernelName,
                        globalThreads, localThreads, args, -1, -1);
}

static void polynomialExpansionOcl(const oclMat &src, int polyN, oclMat &dst)
{
    String kernelName("polynomialExpansion");
    size_t localThreads[3] = { 256, 1, 1 };
    size_t globalThreads[3] = { divUp(src.cols, localThreads[0] - 2*polyN) * localThreads[0], src.rows, 1 };
    int smem_size = 3 * localThreads[0] * sizeof(float);

    std::vector< std::pair<size_t, const void *> > args;
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&dst.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&src.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&g.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&xg.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&xxg.data));
    args.push_back(std::make_pair(smem_size, (void *)NULL));
    args.push_back(std::make_pair(sizeof(cl_float4), (void *)&ig));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.rows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.cols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.step));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.step));

    char opt [128];
    sprintf(opt, "-D polyN=%d", polyN);

    openCLExecuteKernel(Context::getContext(), &optical_flow_farneback, kernelName,
                        globalThreads, localThreads, args, -1, -1, opt);
}

static void updateMatricesOcl(const oclMat &flowx, const oclMat &flowy, const oclMat &R0, const oclMat &R1, oclMat &M)
{
    String kernelName("updateMatrices");
    size_t localThreads[3] = { 32, 8, 1 };
    size_t globalThreads[3] = { flowx.cols, flowx.rows, 1 };

    std::vector< std::pair<size_t, const void *> > args;
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&M.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&flowx.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&flowy.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&R0.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&R1.data));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&flowx.rows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&flowx.cols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&M.step));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&flowx.step));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&flowy.step));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&R0.step));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&R1.step));

    openCLExecuteKernel(Context::getContext(), &optical_flow_farneback, kernelName,
                        globalThreads, localThreads, args, -1, -1);
}

static void boxFilter5Ocl(const oclMat &src, int ksizeHalf, oclMat &dst)
{
    String kernelName("boxFilter5");
    int height = src.rows / 5;
    size_t localThreads[3] = { 256, 1, 1 };
    size_t globalThreads[3] = { src.cols, height, 1 };
    int smem_size = (localThreads[0] + 2*ksizeHalf) * 5 * sizeof(float);

    std::vector< std::pair<size_t, const void *> > args;
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&dst.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&src.data));
    args.push_back(std::make_pair(smem_size, (void *)NULL));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&height));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.cols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.step));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.step));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&ksizeHalf));

    openCLExecuteKernel(Context::getContext(), &optical_flow_farneback, kernelName,
                        globalThreads, localThreads, args, -1, -1);
}

static void updateFlowOcl(const oclMat &M, oclMat &flowx, oclMat &flowy)
{
    String kernelName("updateFlow");
    int cols = divUp(flowx.cols, 4);
    size_t localThreads[3] = { 32, 8, 1 };
    size_t globalThreads[3] = { cols, flowx.rows, 1 };

    std::vector< std::pair<size_t, const void *> > args;
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&flowx.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&flowy.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&M.data));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&flowx.rows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&cols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&flowx.step));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&flowy.step));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&M.step));

    openCLExecuteKernel(Context::getContext(), &optical_flow_farneback, kernelName,
                        globalThreads, localThreads, args, -1, -1);
}

static void gaussianBlur5Ocl(const oclMat &src, int ksizeHalf, oclMat &dst)
{
    String kernelName("gaussianBlur5");
    int height = src.rows / 5;
    size_t localThreads[3] = { 256, 1, 1 };
    size_t globalThreads[3] = { src.cols, height, 1 };
    int smem_size = (localThreads[0] + 2*ksizeHalf) * 5 * sizeof(float);

    std::vector< std::pair<size_t, const void *> > args;
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&dst.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&src.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&gKer.data));
    args.push_back(std::make_pair(smem_size, (void *)NULL));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&height));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.cols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&dst.step));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&src.step));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&ksizeHalf));

    openCLExecuteKernel(Context::getContext(), &optical_flow_farneback, kernelName,
                        globalThreads, localThreads, args, -1, -1);
}
}
}
} // namespace cv { namespace ocl { namespace optflow_farneback

static oclMat allocMatFromBuf(int rows, int cols, int type, oclMat &mat)
{
    if (!mat.empty() && mat.type() == type && mat.rows >= rows && mat.cols >= cols)
        return mat(Rect(0, 0, cols, rows));
    return mat = oclMat(rows, cols, type);
}

cv::ocl::FarnebackOpticalFlow::FarnebackOpticalFlow()
{
    numLevels = 5;
    pyrScale = 0.5;
    fastPyramids = false;
    winSize = 13;
    numIters = 10;
    polyN = 5;
    polySigma = 1.1;
    flags = 0;
}

void cv::ocl::FarnebackOpticalFlow::releaseMemory()
{
    frames_[0].release();
    frames_[1].release();
    pyrLevel_[0].release();
    pyrLevel_[1].release();
    M_.release();
    bufM_.release();
    R_[0].release();
    R_[1].release();
    blurredFrame_[0].release();
    blurredFrame_[1].release();
    pyramid0_.clear();
    pyramid1_.clear();
}

void cv::ocl::FarnebackOpticalFlow::prepareGaussian(
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

void cv::ocl::FarnebackOpticalFlow::setPolynomialExpansionConsts(int n, double sigma)
{
    std::vector<float> buf(n*6 + 3);
    float* g = &buf[0] + n;
    float* xg = g + n*2 + 1;
    float* xxg = xg + n*2 + 1;

    if (sigma < FLT_EPSILON)
        sigma = n*0.3;

    double ig11, ig03, ig33, ig55;
    prepareGaussian(n, sigma, g, xg, xxg, ig11, ig03, ig33, ig55);

    cv::Mat t_g(1, n + 1, CV_32FC1, g);
    cv::Mat t_xg(1, n + 1, CV_32FC1, xg);
    cv::Mat t_xxg(1, n + 1, CV_32FC1, xxg);

    optflow_farneback::g.upload(t_g);
    optflow_farneback::xg.upload(t_xg);
    optflow_farneback::xxg.upload(t_xxg);

    optflow_farneback::ig[0] = static_cast<float>(ig11);
    optflow_farneback::ig[1] = static_cast<float>(ig03);
    optflow_farneback::ig[2] = static_cast<float>(ig33);
    optflow_farneback::ig[3] = static_cast<float>(ig55);
}

void cv::ocl::FarnebackOpticalFlow::updateFlow_boxFilter(
    const oclMat& R0, const oclMat& R1, oclMat& flowx, oclMat &flowy,
    oclMat& M, oclMat &bufM, int blockSize, bool updateMatrices)
{
    optflow_farneback::boxFilter5Ocl(M, blockSize/2, bufM);

    swap(M, bufM);

    optflow_farneback::updateFlowOcl(M, flowx, flowy);

    if (updateMatrices)
        optflow_farneback::updateMatricesOcl(flowx, flowy, R0, R1, M);
}


void cv::ocl::FarnebackOpticalFlow::updateFlow_gaussianBlur(
    const oclMat& R0, const oclMat& R1, oclMat& flowx, oclMat& flowy,
    oclMat& M, oclMat &bufM, int blockSize, bool updateMatrices)
{
    optflow_farneback::gaussianBlur5Ocl(M, blockSize/2, bufM);

    swap(M, bufM);

    optflow_farneback::updateFlowOcl(M, flowx, flowy);

    if (updateMatrices)
        optflow_farneback::updateMatricesOcl(flowx, flowy, R0, R1, M);
}


void cv::ocl::FarnebackOpticalFlow::operator ()(
    const oclMat &frame0, const oclMat &frame1, oclMat &flowx, oclMat &flowy)
{
    CV_Assert(frame0.channels() == 1 && frame1.channels() == 1);
    CV_Assert(frame0.size() == frame1.size());
    CV_Assert(polyN == 5 || polyN == 7);
    CV_Assert(!fastPyramids || std::abs(pyrScale - 0.5) < 1e-6);

    Size size = frame0.size();
    oclMat prevFlowX, prevFlowY, curFlowX, curFlowY;

    flowx.create(size, CV_32F);
    flowy.create(size, CV_32F);
    oclMat flowx0 = flowx;
    oclMat flowy0 = flowy;

    // Crop unnecessary levels
    double scale = 1;
    int numLevelsCropped = 0;
    for (; numLevelsCropped < numLevels; numLevelsCropped++)
    {
        scale *= pyrScale;
        if (size.width*scale < MIN_SIZE || size.height*scale < MIN_SIZE)
            break;
    }

    frame0.convertTo(frames_[0], CV_32F);
    frame1.convertTo(frames_[1], CV_32F);

    if (fastPyramids)
    {
        // Build Gaussian pyramids using pyrDown()
        pyramid0_.resize(numLevelsCropped + 1);
        pyramid1_.resize(numLevelsCropped + 1);
        pyramid0_[0] = frames_[0];
        pyramid1_[0] = frames_[1];
        for (int i = 1; i <= numLevelsCropped; ++i)
        {
            pyrDown(pyramid0_[i - 1], pyramid0_[i]);
            pyrDown(pyramid1_[i - 1], pyramid1_[i]);
        }
    }

    setPolynomialExpansionConsts(polyN, polySigma);

    for (int k = numLevelsCropped; k >= 0; k--)
    {
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
            if (flags & cv::OPTFLOW_USE_INITIAL_FLOW)
            {
                resize(flowx0, curFlowX, Size(width, height), 0, 0, INTER_LINEAR);
                resize(flowy0, curFlowY, Size(width, height), 0, 0, INTER_LINEAR);
                multiply(scale, curFlowX, curFlowX);
                multiply(scale, curFlowY, curFlowY);
            }
            else
            {
                curFlowX.setTo(0);
                curFlowY.setTo(0);
            }
        }
        else
        {
            resize(prevFlowX, curFlowX, Size(width, height), 0, 0, INTER_LINEAR);
            resize(prevFlowY, curFlowY, Size(width, height), 0, 0, INTER_LINEAR);
            multiply(1./pyrScale, curFlowX, curFlowX);
            multiply(1./pyrScale, curFlowY, curFlowY);
        }

        oclMat M = allocMatFromBuf(5*height, width, CV_32F, M_);
        oclMat bufM = allocMatFromBuf(5*height, width, CV_32F, bufM_);
        oclMat R[2] =
        {
            allocMatFromBuf(5*height, width, CV_32F, R_[0]),
            allocMatFromBuf(5*height, width, CV_32F, R_[1])
        };

        if (fastPyramids)
        {
            optflow_farneback::polynomialExpansionOcl(pyramid0_[k], polyN, R[0]);
            optflow_farneback::polynomialExpansionOcl(pyramid1_[k], polyN, R[1]);
        }
        else
        {
            oclMat blurredFrame[2] =
            {
                allocMatFromBuf(size.height, size.width, CV_32F, blurredFrame_[0]),
                allocMatFromBuf(size.height, size.width, CV_32F, blurredFrame_[1])
            };
            oclMat pyrLevel[2] =
            {
                allocMatFromBuf(height, width, CV_32F, pyrLevel_[0]),
                allocMatFromBuf(height, width, CV_32F, pyrLevel_[1])
            };

            Mat g = getGaussianKernel(smoothSize, sigma, CV_32F);
            optflow_farneback::setGaussianBlurKernel(g.ptr<float>(smoothSize/2), smoothSize/2);

            for (int i = 0; i < 2; i++)
            {
                optflow_farneback::gaussianBlurOcl(frames_[i], smoothSize/2, blurredFrame[i]);
                resize(blurredFrame[i], pyrLevel[i], Size(width, height), INTER_LINEAR);
                optflow_farneback::polynomialExpansionOcl(pyrLevel[i], polyN, R[i]);
            }
        }

        optflow_farneback::updateMatricesOcl(curFlowX, curFlowY, R[0], R[1], M);

        if (flags & OPTFLOW_FARNEBACK_GAUSSIAN)
        {
            Mat g = getGaussianKernel(winSize, winSize/2*0.3f, CV_32F);
            optflow_farneback::setGaussianBlurKernel(g.ptr<float>(winSize/2), winSize/2);
        }
        for (int i = 0; i < numIters; i++)
        {
            if (flags & OPTFLOW_FARNEBACK_GAUSSIAN)
                updateFlow_gaussianBlur(R[0], R[1], curFlowX, curFlowY, M, bufM, winSize, i < numIters-1);
            else
                updateFlow_boxFilter(R[0], R[1], curFlowX, curFlowY, M, bufM, winSize, i < numIters-1);
        }

        prevFlowX = curFlowX;
        prevFlowY = curFlowY;
    }

    flowx = curFlowX;
    flowy = curFlowY;
}
