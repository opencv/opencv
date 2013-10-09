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
//                For Open Source Comuter Vision Library
//
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@multicorewareinc.com
//
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
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular urpose are disclaimed.
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

namespace cv
{
    namespace ocl
    {
        namespace interpolate
        {
            //The following are ported from NPP_staging.cu
            // As it is not valid to do pointer offset operations on host for default oclMat's native cl_mem pointer,
            // we may have to do this on kernel
            void memsetKernel(float val, oclMat &img, int height, int offset);
            void normalizeKernel(oclMat &buffer, int height, int factor_offset, int dst_offset);
            void forwardWarpKernel(const oclMat &src, oclMat &buffer, const oclMat &u, const oclMat &v, const float time_scale,
                                   int b_offset, int d_offset); // buffer, dst offset

            //OpenCL conversion of nppiStVectorWarp_PSF2x2_32f_C1
            void vectorWarp(const oclMat &src, const oclMat &u, const oclMat &v,
                            oclMat &buffer, int buf_offset, float timeScale, int dst_offset);
            //OpenCL conversion of BlendFrames
            void blendFrames(const oclMat &frame0, const oclMat &frame1, const oclMat &buffer,
                             float pos, oclMat &newFrame, cl_mem &, cl_mem &);

            // bind a buffer to an image
            void bindImgTex(const oclMat &img, cl_mem &tex);
        }
    }
}

void cv::ocl::interpolateFrames(const oclMat &frame0, const oclMat &frame1,
                                const oclMat &fu, const oclMat &fv,
                                const oclMat &bu, const oclMat &bv,
                                float pos, oclMat &newFrame, oclMat &buf)
{
    CV_Assert(frame0.type() == CV_32FC1);
    CV_Assert(frame1.size() == frame0.size() && frame1.type() == frame0.type());
    CV_Assert(fu.size() == frame0.size() && fu.type() == frame0.type());
    CV_Assert(fv.size() == frame0.size() && fv.type() == frame0.type());
    CV_Assert(bu.size() == frame0.size() && bu.type() == frame0.type());
    CV_Assert(bv.size() == frame0.size() && bv.type() == frame0.type());

    newFrame.create(frame0.size(), frame0.type());

    buf.create(6 * frame0.rows, frame0.cols, CV_32FC1);
    buf.setTo(Scalar::all(0));

    size_t step = frame0.step;

    CV_Assert(frame1.step == step && fu.step == step && fv.step == step && bu.step == step && bv.step == step && newFrame.step == step && buf.step == step);
    cl_mem tex_src0 = 0, tex_src1 = 0;

    // warp flow
    using namespace interpolate;

    bindImgTex(frame0, tex_src0);
    bindImgTex(frame1, tex_src1);

    // CUDA Offsets
    enum
    {
        cov0 = 0,
        cov1,
        fwdU,
        fwdV,
        bwdU,
        bwdV
    };

    vectorWarp(fu, fu, fv, buf, cov0, pos,        fwdU);
    vectorWarp(fv, fu, fv, buf, cov0, pos,        fwdV);
    vectorWarp(bu, bu, bv, buf, cov1, 1.0f - pos, bwdU);
    vectorWarp(bv, bu, bv, buf, cov1, 1.0f - pos, bwdU);

    blendFrames(frame0, frame1, buf, pos, newFrame, tex_src0, tex_src1);

    openCLFree(tex_src0);
    openCLFree(tex_src1);
}

void interpolate::memsetKernel(float val, oclMat &img, int height, int offset)
{
    Context *clCxt = Context::getContext();
    String kernelName = "memsetKernel";
    std::vector< std::pair<size_t, const void *> > args;
    int step = img.step / sizeof(float);
    offset = step * height * offset;

    args.push_back( std::make_pair( sizeof(cl_float), (void *)&val));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&img.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&img.cols));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&height));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&step));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&offset));

    size_t globalThreads[3] = {img.cols, height, 1};
    size_t localThreads[3]  = {16, 16, 1};
    openCLExecuteKernel(clCxt, &interpolate_frames, kernelName, globalThreads, localThreads, args, -1, -1);
}
void interpolate::normalizeKernel(oclMat &buffer, int height, int factor_offset, int dst_offset)
{
    Context *clCxt = Context::getContext();
    String kernelName = "normalizeKernel";
    std::vector< std::pair<size_t, const void *> > args;
    int step   = buffer.step / sizeof(float);
    factor_offset = step * height * factor_offset;
    dst_offset    = step * height * dst_offset;

    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&buffer.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&buffer.cols));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&height));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&step));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&factor_offset));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst_offset));

    size_t globalThreads[3] = {buffer.cols, height, 1};
    size_t localThreads[3]  = {16, 16, 1};
    openCLExecuteKernel(clCxt, &interpolate_frames, kernelName, globalThreads, localThreads, args, -1, -1);
}

void interpolate::forwardWarpKernel(const oclMat &src, oclMat &buffer, const oclMat &u, const oclMat &v, const float time_scale,
                                    int b_offset, int d_offset)
{
    Context *clCxt = Context::getContext();
    String kernelName = "forwardWarpKernel";
    std::vector< std::pair<size_t, const void *> > args;
    int f_step  = u.step / sizeof(float); // flow step
    int b_step  = buffer.step / sizeof(float);

    b_offset  = b_step * src.rows * b_offset;
    d_offset  = b_step * src.rows * d_offset;

    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&buffer.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&u.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&v.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.cols));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.rows));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&f_step));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&b_step));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&b_offset));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&d_offset));
    args.push_back( std::make_pair( sizeof(cl_float), (void *)&time_scale));

    size_t globalThreads[3] = {src.cols, src.rows, 1};
    size_t localThreads[3]  = {16, 16, 1};
    openCLExecuteKernel(clCxt, &interpolate_frames, kernelName, globalThreads, localThreads, args, -1, -1);
}

void interpolate::vectorWarp(const oclMat &src, const oclMat &u, const oclMat &v,
                             oclMat &buffer, int b_offset, float timeScale, int d_offset)
{
    memsetKernel(0, buffer, src.rows, b_offset);
    forwardWarpKernel(src, buffer, u, v, timeScale, b_offset, d_offset);
    normalizeKernel(buffer, src.rows, b_offset, d_offset);
}

void interpolate::blendFrames(const oclMat &frame0, const oclMat &/*frame1*/, const oclMat &buffer, float pos, oclMat &newFrame, cl_mem &tex_src0, cl_mem &tex_src1)
{
    int step = buffer.step / sizeof(float);

    Context *clCxt = Context::getContext();
    String kernelName = "blendFramesKernel";
    std::vector< std::pair<size_t, const void *> > args;

    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&tex_src0));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&tex_src1));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&buffer.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&newFrame.data));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&frame0.cols));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&frame0.rows));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&step));
    args.push_back( std::make_pair( sizeof(cl_float), (void *)&pos));

    size_t globalThreads[3] = {frame0.cols, frame0.rows, 1};
    size_t localThreads[3]  = {16, 16, 1};
    openCLExecuteKernel(clCxt, &interpolate_frames, kernelName, globalThreads, localThreads, args, -1, -1);
}

void interpolate::bindImgTex(const oclMat &img, cl_mem &texture)
{
    if(texture)
    {
        openCLFree(texture);
    }
    texture = bindTexture(img);
}
