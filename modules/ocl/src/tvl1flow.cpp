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
//        Jin Ma, jin@multicorewareinc.com
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
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

using namespace cv;
using namespace cv::ocl;

cv::ocl::OpticalFlowDual_TVL1_OCL::OpticalFlowDual_TVL1_OCL()
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

void cv::ocl::OpticalFlowDual_TVL1_OCL::operator()(const oclMat& I0, const oclMat& I1, oclMat& flowx, oclMat& flowy)
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
    //I0s_step == I1s_step
    I0.convertTo(I0s[0], CV_32F, I0.depth() == CV_8U ? 1.0 : 255.0);
    I1.convertTo(I1s[0], CV_32F, I1.depth() == CV_8U ? 1.0 : 255.0);


    if (!useInitialFlow)
    {
        flowx.create(I0.size(), CV_32FC1);
        flowy.create(I0.size(), CV_32FC1);
    }
    //u1s_step != u2s_step
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
        ocl::pyrDown(I0s[s - 1], I0s[s]);
        ocl::pyrDown(I1s[s - 1], I1s[s]);

        if (I0s[s].cols < 16 || I0s[s].rows < 16)
        {
            nscales = s;
            break;
        }

        if (useInitialFlow)
        {
            ocl::pyrDown(u1s[s - 1], u1s[s]);
            ocl::pyrDown(u2s[s - 1], u2s[s]);

            //ocl::multiply(u1s[s], Scalar::all(0.5), u1s[s]);
            multiply(0.5, u1s[s], u1s[s]);
            //ocl::multiply(u2s[s], Scalar::all(0.5), u2s[s]);
            multiply(0.5, u1s[s], u2s[s]);
        }
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
        ocl::resize(u1s[s], u1s[s - 1], I0s[s - 1].size());
        ocl::resize(u2s[s], u2s[s - 1], I0s[s - 1].size());

        // scale the optical flow with the appropriate zoom factor
        multiply(2, u1s[s - 1], u1s[s - 1]);
        multiply(2, u2s[s - 1], u2s[s - 1]);

    }

}

namespace ocl_tvl1flow
{
    void centeredGradient(const oclMat &src, oclMat &dx, oclMat &dy);

    void warpBackward(const oclMat &I0, const oclMat &I1, oclMat &I1x, oclMat &I1y,
        oclMat &u1, oclMat &u2, oclMat &I1w, oclMat &I1wx, oclMat &I1wy,
        oclMat &grad, oclMat &rho);

    void estimateU(oclMat &I1wx, oclMat &I1wy, oclMat &grad,
        oclMat &rho_c, oclMat &p11, oclMat &p12,
        oclMat &p21, oclMat &p22, oclMat &u1,
        oclMat &u2, oclMat &error, float l_t, float theta, char calc_error);

    void estimateDualVariables(oclMat &u1, oclMat &u2,
        oclMat &p11, oclMat &p12, oclMat &p21, oclMat &p22, float taut);
}

void cv::ocl::OpticalFlowDual_TVL1_OCL::procOneScale(const oclMat &I0, const oclMat &I1, oclMat &u1, oclMat &u2)
{
    using namespace ocl_tvl1flow;

    const double scaledEpsilon = epsilon * epsilon * I0.size().area();

    CV_DbgAssert( I1.size() == I0.size() );
    CV_DbgAssert( I1.type() == I0.type() );
    CV_DbgAssert( u1.empty() || u1.size() == I0.size() );
    CV_DbgAssert( u2.size() == u1.size() );

    if (u1.empty())
    {
        u1.create(I0.size(), CV_32FC1);
        u1.setTo(Scalar::all(0));

        u2.create(I0.size(), CV_32FC1);
        u2.setTo(Scalar::all(0));
    }

    oclMat I1x = I1x_buf(Rect(0, 0, I0.cols, I0.rows));
    oclMat I1y = I1y_buf(Rect(0, 0, I0.cols, I0.rows));

    centeredGradient(I1, I1x, I1y);

    oclMat I1w = I1w_buf(Rect(0, 0, I0.cols, I0.rows));
    oclMat I1wx = I1wx_buf(Rect(0, 0, I0.cols, I0.rows));
    oclMat I1wy = I1wy_buf(Rect(0, 0, I0.cols, I0.rows));

    oclMat grad = grad_buf(Rect(0, 0, I0.cols, I0.rows));
    oclMat rho_c = rho_c_buf(Rect(0, 0, I0.cols, I0.rows));

    oclMat p11 = p11_buf(Rect(0, 0, I0.cols, I0.rows));
    oclMat p12 = p12_buf(Rect(0, 0, I0.cols, I0.rows));
    oclMat p21 = p21_buf(Rect(0, 0, I0.cols, I0.rows));
    oclMat p22 = p22_buf(Rect(0, 0, I0.cols, I0.rows));
    p11.setTo(Scalar::all(0));
    p12.setTo(Scalar::all(0));
    p21.setTo(Scalar::all(0));
    p22.setTo(Scalar::all(0));

    oclMat diff = diff_buf(Rect(0, 0, I0.cols, I0.rows));

    const float l_t = static_cast<float>(lambda * theta);
    const float taut = static_cast<float>(tau / theta);

    for (int warpings = 0; warpings < warps; ++warpings)
    {
        warpBackward(I0, I1, I1x, I1y, u1, u2, I1w, I1wx, I1wy, grad, rho_c);

        double error = std::numeric_limits<double>::max();
        double prev_error = 0;
        for (int n = 0; error > scaledEpsilon && n < iterations; ++n)
        {
            // some tweaks to make sum operation less frequently
            char calc_error = (n & 0x1) && (prev_error < scaledEpsilon);
            estimateU(I1wx, I1wy, grad, rho_c, p11, p12, p21, p22,
                      u1, u2, diff, l_t, static_cast<float>(theta), calc_error);
            if(calc_error)
            {
                error = ocl::sum(diff)[0];
                prev_error = error;
            }
            else
            {
                error = std::numeric_limits<double>::max();
                prev_error -= scaledEpsilon;
            }
            estimateDualVariables(u1, u2, p11, p12, p21, p22, taut);

        }
    }


}

void cv::ocl::OpticalFlowDual_TVL1_OCL::collectGarbage()
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

void ocl_tvl1flow::centeredGradient(const oclMat &src, oclMat &dx, oclMat &dy)
{
    Context  *clCxt = src.clCxt;
    size_t localThreads[3] = {32, 8, 1};
    size_t globalThreads[3] = {src.cols, src.rows, 1};

    int srcElementSize = src.elemSize();
    int src_step = src.step/srcElementSize;

    int dElememntSize = dx.elemSize();
    int dx_step = dx.step/dElememntSize;

    String kernelName = "centeredGradientKernel";
    std::vector< std::pair<size_t, const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&src.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&src.cols));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&src.rows));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&src_step));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&dx.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&dy.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&dx_step));
    openCLExecuteKernel(clCxt, &tvl1flow, kernelName, globalThreads, localThreads, args, -1, -1);

}

void ocl_tvl1flow::estimateDualVariables(oclMat &u1, oclMat &u2, oclMat &p11, oclMat &p12, oclMat &p21, oclMat &p22, float taut)
{
    Context *clCxt = u1.clCxt;

    size_t localThread[] = {32, 8, 1};
    size_t globalThread[] =
    {
        u1.cols,
        u1.rows,
        1
    };

    int u1_element_size = u1.elemSize();
    int u1_step = u1.step/u1_element_size;

    int u2_element_size = u2.elemSize();
    int u2_step = u2.step/u2_element_size;

    int p11_element_size = p11.elemSize();
    int p11_step = p11.step/p11_element_size;

    int u1_offset_y = u1.offset/u1.step;
    int u1_offset_x = u1.offset%u1.step;
    u1_offset_x = u1_offset_x/u1.elemSize();

    int u2_offset_y = u2.offset/u2.step;
    int u2_offset_x = u2.offset%u2.step;
    u2_offset_x = u2_offset_x/u2.elemSize();

    String kernelName = "estimateDualVariablesKernel";
    std::vector< std::pair<size_t, const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&u1.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&u1.cols));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&u1.rows));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&u1_step));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&u2.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&p11.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&p11_step));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&p12.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&p21.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&p22.data));
    args.push_back( std::make_pair( sizeof(cl_float), (void*)&taut));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&u2_step));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&u1_offset_x));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&u1_offset_y));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&u2_offset_x));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&u2_offset_y));

    openCLExecuteKernel(clCxt, &tvl1flow, kernelName, globalThread, localThread, args, -1, -1);
}

void ocl_tvl1flow::estimateU(oclMat &I1wx, oclMat &I1wy, oclMat &grad,
    oclMat &rho_c, oclMat &p11, oclMat &p12,
    oclMat &p21, oclMat &p22, oclMat &u1,
    oclMat &u2, oclMat &error, float l_t, float theta, char calc_error)
{
    Context* clCxt = I1wx.clCxt;

    size_t localThread[] = {32, 8, 1};
    size_t globalThread[] =
    {
        I1wx.cols,
        I1wx.rows,
        1
    };

    int I1wx_element_size = I1wx.elemSize();
    int I1wx_step = I1wx.step/I1wx_element_size;

    int u1_element_size = u1.elemSize();
    int u1_step = u1.step/u1_element_size;

    int u2_element_size = u2.elemSize();
    int u2_step = u2.step/u2_element_size;

    int u1_offset_y = u1.offset/u1.step;
    int u1_offset_x = u1.offset%u1.step;
    u1_offset_x = u1_offset_x/u1.elemSize();

    int u2_offset_y = u2.offset/u2.step;
    int u2_offset_x = u2.offset%u2.step;
    u2_offset_x = u2_offset_x/u2.elemSize();

    String kernelName = "estimateUKernel";
    std::vector< std::pair<size_t, const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&I1wx.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&I1wx.cols));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&I1wx.rows));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&I1wx_step));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&I1wy.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&grad.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&rho_c.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&p11.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&p12.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&p21.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&p22.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&u1.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&u1_step));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&u2.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&error.data));
    args.push_back( std::make_pair( sizeof(cl_float), (void*)&l_t));
    args.push_back( std::make_pair( sizeof(cl_float), (void*)&theta));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&u2_step));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&u1_offset_x));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&u1_offset_y));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&u2_offset_x));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&u2_offset_y));
    args.push_back( std::make_pair( sizeof(cl_char), (void*)&calc_error));

    openCLExecuteKernel(clCxt, &tvl1flow, kernelName, globalThread, localThread, args, -1, -1);
}

void ocl_tvl1flow::warpBackward(const oclMat &I0, const oclMat &I1, oclMat &I1x, oclMat &I1y, oclMat &u1, oclMat &u2, oclMat &I1w, oclMat &I1wx, oclMat &I1wy, oclMat &grad, oclMat &rho)
{
    Context* clCxt = I0.clCxt;

    int u1ElementSize = u1.elemSize();
    int u1Step = u1.step/u1ElementSize;

    int u2ElementSize = u2.elemSize();
    int u2Step = u2.step/u2ElementSize;

    int I0ElementSize = I0.elemSize();
    int I0Step = I0.step/I0ElementSize;

    int I1w_element_size = I1w.elemSize();
    int I1w_step = I1w.step/I1w_element_size;

    int u1_offset_y = u1.offset/u1.step;
    int u1_offset_x = u1.offset%u1.step;
    u1_offset_x = u1_offset_x/u1.elemSize();

    int u2_offset_y = u2.offset/u2.step;
    int u2_offset_x = u2.offset%u2.step;
    u2_offset_x = u2_offset_x/u2.elemSize();

    size_t localThread[] = {32, 8, 1};
    size_t globalThread[] =
    {
        I0.cols,
        I0.rows,
        1
    };

    cl_mem I1_tex;
    cl_mem I1x_tex;
    cl_mem I1y_tex;
    I1_tex = bindTexture(I1);
    I1x_tex = bindTexture(I1x);
    I1y_tex = bindTexture(I1y);

    String kernelName = "warpBackwardKernel";
    std::vector< std::pair<size_t, const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&I0.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&I0Step));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&I0.cols));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&I0.rows));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&I1_tex));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&I1x_tex));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&I1y_tex));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&u1.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&u1Step));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&u2.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&I1w.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&I1wx.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&I1wy.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&grad.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void*)&rho.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&I1w_step));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&u2Step));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&u1_offset_x));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&u1_offset_y));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&u2_offset_x));
    args.push_back( std::make_pair( sizeof(cl_int), (void*)&u2_offset_y));

    openCLExecuteKernel(clCxt, &tvl1flow, kernelName, globalThread, localThread, args, -1, -1);

    releaseTexture(I1_tex);
    releaseTexture(I1x_tex);
    releaseTexture(I1y_tex);
}
