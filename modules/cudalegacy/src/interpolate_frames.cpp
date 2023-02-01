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

using namespace cv;
using namespace cv::cuda;

#if !defined (HAVE_CUDA) || !defined (HAVE_OPENCV_CUDALEGACY) || defined (CUDA_DISABLER)

void cv::cuda::interpolateFrames(const GpuMat&, const GpuMat&, const GpuMat&, const GpuMat&, const GpuMat&, const GpuMat&, float, GpuMat&, GpuMat&, Stream&) { throw_no_cuda(); }

#else

void cv::cuda::interpolateFrames(const GpuMat& frame0, const GpuMat& frame1, const GpuMat& fu, const GpuMat& fv, const GpuMat& bu, const GpuMat& bv,
                                float pos, GpuMat& newFrame, GpuMat& buf, Stream& s)
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

    // occlusion masks
    GpuMat occ0 = buf.rowRange(0 * frame0.rows, 1 * frame0.rows);
    GpuMat occ1 = buf.rowRange(1 * frame0.rows, 2 * frame0.rows);

    // interpolated forward flow
    GpuMat fui = buf.rowRange(2 * frame0.rows, 3 * frame0.rows);
    GpuMat fvi = buf.rowRange(3 * frame0.rows, 4 * frame0.rows);

    // interpolated backward flow
    GpuMat bui = buf.rowRange(4 * frame0.rows, 5 * frame0.rows);
    GpuMat bvi = buf.rowRange(5 * frame0.rows, 6 * frame0.rows);

    size_t step = frame0.step;

    CV_Assert(frame1.step == step && fu.step == step && fv.step == step && bu.step == step && bv.step == step && newFrame.step == step && buf.step == step);

    cudaStream_t stream = StreamAccessor::getStream(s);
    NppStStreamHandler h(stream);

    NppStInterpolationState state;

    state.size         = NcvSize32u(frame0.cols, frame0.rows);
    state.nStep        = static_cast<Ncv32u>(step);
    state.pSrcFrame0   = const_cast<Ncv32f*>(frame0.ptr<Ncv32f>());
    state.pSrcFrame1   = const_cast<Ncv32f*>(frame1.ptr<Ncv32f>());
    state.pFU          = const_cast<Ncv32f*>(fu.ptr<Ncv32f>());
    state.pFV          = const_cast<Ncv32f*>(fv.ptr<Ncv32f>());
    state.pBU          = const_cast<Ncv32f*>(bu.ptr<Ncv32f>());
    state.pBV          = const_cast<Ncv32f*>(bv.ptr<Ncv32f>());
    state.pos          = pos;
    state.pNewFrame    = newFrame.ptr<Ncv32f>();
    state.ppBuffers[0] = occ0.ptr<Ncv32f>();
    state.ppBuffers[1] = occ1.ptr<Ncv32f>();
    state.ppBuffers[2] = fui.ptr<Ncv32f>();
    state.ppBuffers[3] = fvi.ptr<Ncv32f>();
    state.ppBuffers[4] = bui.ptr<Ncv32f>();
    state.ppBuffers[5] = bvi.ptr<Ncv32f>();

    ncvSafeCall( nppiStInterpolateFrames(&state) );

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}

#endif /* HAVE_CUDA */
